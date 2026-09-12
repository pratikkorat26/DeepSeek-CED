"""Dense multi-head attention primitives for the CED LLM.

Both modules are dense (full, non-sparse) multi-head attention backed by
``torch.nn.functional.scaled_dot_product_attention``. There is no MoE, no
GQA/MLA, no sparse pattern, and no RoPE in v1 (positions come from learned
embeddings in ``model.py``).
"""

from typing import Optional, Tuple

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SDPA backend selection (FORGE-MODEL: MPS measurement)
# ---------------------------------------------------------------------------
# On Apple MPS all SDPBackend choices lower to the same MPS kernel (measured
# ~0.039ms tiny-causal and ~0.025ms decode shapes across MATH/FLASH/
# EFFICIENT/CUDNN within noise; SDPA itself is <1% of the ~3.9ms/token
# decode cost, proving overhead-bound). Default is therefore direct
# ``scaled_dot_product_attention`` with NO per-forward context overhead
# (bit-identical). An explicit backend can be forced via env
# ``CED_SDPA_BACKEND=math|flash|efficient|mem|cudnn|auto`` or per-call
# override for experimentation.
def preferred_sdpa_backend(device=None):
    """Return the preferred SDPBackend for ``device`` or ``None`` for auto.

    ``None`` means "call SDPA directly, no context" (fastest, bit-identical).
    On MPS this is MATH-equivalent; explicit env override still honored.
    """
    try:
        env = str(os.environ.get("CED_SDPA_BACKEND", "auto")).strip().lower()
    except Exception:
        env = "auto"
    if env in ("math",):
        try:
            from torch.nn.attention import SDPBackend as _B

            return _B.MATH
        except Exception:
            return None
    if env in ("flash", "flash_attention"):
        try:
            from torch.nn.attention import SDPBackend as _B

            return _B.FLASH_ATTENTION
        except Exception:
            return None
    if env in ("efficient", "mem", "mem_efficient", "memory"):
        try:
            from torch.nn.attention import SDPBackend as _B

            return _B.EFFICIENT_ATTENTION
        except Exception:
            return None
    if env in ("cudnn", "cudnn_attention"):
        try:
            from torch.nn.attention import SDPBackend as _B

            return _B.CUDNN_ATTENTION
        except Exception:
            return None
    # "auto" (default): no context -> direct call (zero overhead).
    return None


def _sdpa(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    """SDPA with optional backend context (default: direct, no overhead).

    Extra kwarg ``_backend`` may hold an SDPBackend to force; otherwise the
    env-selected ``preferred_sdpa_backend`` is consulted once per call (cheap
    string check, no context when auto).
    """
    backend = kwargs.pop("_backend", None)
    if backend is None:
        backend = preferred_sdpa_backend(getattr(q, "device", None))
    if backend is None:
        return F.scaled_dot_product_attention(q, k, v, **kwargs)
    try:
        from torch.nn.attention import sdpa_kernel as _ctx

        with _ctx(backend):
            return F.scaled_dot_product_attention(q, k, v, **kwargs)
    except Exception:
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


def _qkv_param_key(*projs) -> tuple:
    """Cache key for fused-QKV weights: (ptr, version, device, dtype)."""
    key = []
    for p in projs:
        try:
            w = p.weight
        except Exception:
            key.append((0, 0, "", ""))
            continue
        try:
            ptr = w.data_ptr()
        except Exception:
            ptr = 0
        try:
            ver = int(getattr(w, "_version", 0))
        except Exception:
            ver = 0
        try:
            dev = str(w.device)
        except Exception:
            dev = ""
        try:
            dt = str(w.dtype)
        except Exception:
            dt = ""
        key.append((ptr, ver, dev, dt))
    return tuple(key)


def split_heads(x: Tensor, nhead: int) -> Tensor:
    """Split the model dim into heads: ``[B, T, D]`` -> ``[B, H, T, Dh]``."""
    batch, seq_len, d_model = x.shape
    head_dim = d_model // nhead
    return x.view(batch, seq_len, nhead, head_dim).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """Merge heads back: ``[B, H, T, Dh]`` -> ``[B, T, D]``."""
    batch, nhead, seq_len, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, seq_len, nhead * head_dim)


def _is_no_pad_mask(key_padding_mask: Optional[Tensor]) -> bool:
    """True when the mask blocks nothing (None or all-False bool).

    Fast path: lets callers use the ``is_causal=True`` / ``attn_mask=None``
    SDPA kernels and skip materializing a ``[B, 1, T, T]`` mask.
    Non-bool dtypes are treated conservatively as "has pad" so exact
    semantics are preserved for unexpected inputs.
    """
    if key_padding_mask is None:
        return True
    try:
        if key_padding_mask.dtype == torch.bool:
            return not bool(key_padding_mask.any())
    except Exception:
        return False
    return False


# Cache of lower-triangular causal-ok masks keyed by (T, device-str).
# Avoids re-allocating ``torch.tril(torch.ones(T, T))`` on every layer/forward.
_CAUSAL_OK_CACHE: dict = {}


def _get_causal_ok(seq_len: int, device: torch.device) -> Tensor:
    try:
        key = (int(seq_len), str(device))
    except Exception:
        return torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )
    try:
        hit = _CAUSAL_OK_CACHE.get(key)
    except Exception:
        hit = None
    if hit is not None:
        try:
            if hit.device == device and hit.shape == (seq_len, seq_len):
                return hit
        except Exception:
            pass
    fresh = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    try:
        if len(_CAUSAL_OK_CACHE) < 32:
            _CAUSAL_OK_CACHE[key] = fresh
    except Exception:
        pass
    return fresh


def causal_attend_mask(key_padding_mask: Tensor) -> Tensor:
    """Build a fused causal + key-padding SDPA mask.

    Args:
        key_padding_mask: Bool tensor of shape ``[B, T]`` where ``True``
            marks pad (ignore) key positions.

    Returns:
        Bool tensor of shape ``[B, 1, T, T]`` following the
        ``scaled_dot_product_attention`` convention (``True`` = take part in
        attention): position ``i`` may attend to ``j <= i`` when key ``j`` is
        not padding. Broadcastable over attention heads for use as
        ``attn_mask``.
    """
    batch, seq_len = key_padding_mask.shape
    device = key_padding_mask.device
    causal_ok = _get_causal_ok(seq_len, device)
    key_ok = (~key_padding_mask.to(dtype=torch.bool)).view(batch, 1, 1, seq_len)
    return causal_ok.view(1, 1, seq_len, seq_len) & key_ok


class CausalSelfAttention(nn.Module):
    """Strictly causal dense multi-head self-attention with padding support.

    Position ``i`` may attend to positions ``j <= i`` only (no future leak);
    key positions flagged by ``key_padding_mask`` are additionally blocked.
    """

    def __init__(
        self, d_model: int, nhead: int, dropout: float = 0.0, fused_qkv: bool = True
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                "d_model (%d) must be divisible by nhead (%d)" % (d_model, nhead)
            )
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)
        # Fused QKV: single GEMM for q/k/v (3 launches -> 1). Bit-identical
        # (each output element is an independent dot product; batching the
        # output dim does not change reduction order). Disabled only via flag.
        self.use_fused_qkv = bool(fused_qkv)
        self._fused_weight: Optional[Tensor] = None
        self._fused_key: Optional[tuple] = None

    def _fused_qkv_weight(self) -> Optional[Tensor]:
        """Stacked [3D, D] QKV weight, cached in inference only."""
        if not self.use_fused_qkv:
            return None
        # Training / grad-enabled: use the separate projections (original
        # path). A fresh fused cat per forward costs an extra [3D, D] alloc
        # with zero launch savings on tiny shapes, and the separate linears
        # carry identical grad flow. Fusion pays off in inference only.
        if self.training or torch.is_grad_enabled():
            return None
        # Inference (eval + no_grad): reuse cached stack until weights change.
        try:
            key = _qkv_param_key(self.q_proj, self.k_proj, self.v_proj)
        except Exception:
            key = None
        try:
            if (
                self._fused_weight is not None
                and self._fused_key is not None
                and key is not None
                and self._fused_key == key
            ):
                return self._fused_weight
        except Exception:
            pass
        try:
            fused = torch.cat(
                [self.q_proj.weight, self.k_proj.weight, self.v_proj.weight],
                dim=0,
            ).detach()
            # Keep on the same device/dtype as sources (cat already does).
            self._fused_weight = fused
            self._fused_key = key
            return fused
        except Exception:
            return None

    def _qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Project ``x`` to (q, k, v) un-split ``[B, T, D]`` (fused fast path)."""
        if self.use_fused_qkv:
            w = self._fused_qkv_weight()
            if w is not None:
                try:
                    qkv = F.linear(x, w)
                    q, k, v = qkv.chunk(3, dim=-1)
                    return q, k, v
                except Exception:
                    pass
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Attend causally over ``x``.

        Args:
            x: Hidden states of shape ``[B, T, D]``.
            key_padding_mask: Optional bool ``[B, T]`` with ``True`` = pad.

        Returns:
            Tensor of shape ``[B, T, D]``.
        """
        batch, seq_len, _ = x.shape
        q_u, k_u, v_u = self._qkv(x)
        q = split_heads(q_u, self.nhead)
        k = split_heads(k_u, self.nhead)
        v = split_heads(v_u, self.nhead)
        dropout_p = self.dropout if self.training else 0.0
        if _is_no_pad_mask(key_padding_mask):
            # Fast path: no pads -> fused causal kernel, no [B,1,T,T] alloc.
            y = _sdpa(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            if key_padding_mask.shape != (batch, seq_len):
                raise ValueError(
                    "key_padding_mask must have shape [B, T]=[%d, %d], got %s"
                    % (batch, seq_len, tuple(key_padding_mask.shape))
                )
            attn_mask = causal_attend_mask(key_padding_mask)
            y = _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        out = self.out_proj(merge_heads(y))
        # Skip dropout dispatch when it is identity (p==0 or eval).
        if self.training and self.dropout != 0.0:
            out = self.resid_dropout(out)
        return out


class GlobalCrossAttention(nn.Module):
    """Non-causal dense multi-head cross-attention over the full encoder context.

    CED KV-reuse: the keys/values arrive already projected into the shared
    global KV space by ``CEDForLM.kv_proj_k`` / ``kv_proj_v`` (computed once
    from the final encoder states), so this module projects only the queries
    and reuses ``k_glob``/``v_glob`` directly. Every query position may attend
    to every (non-pad) encoder position.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                "d_model (%d) must be divisible by nhead (%d)" % (d_model, nhead)
            )
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: Tensor,
        k_glob: Tensor,
        v_glob: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Cross-attend queries to the global encoder KV.

        Args:
            q: Decoder queries of shape ``[B, Tq, D]``.
            k_glob: Global keys of shape ``[B, Tk, D]`` (pre-projected once).
            v_glob: Global values of shape ``[B, Tk, D]`` (pre-projected once).
            key_padding_mask: Optional bool ``[B, Tk]`` with ``True`` = pad.

        Returns:
            Tensor of shape ``[B, Tq, D]``.
        """
        batch_q, len_q, dim_q = q.shape
        batch_k, len_k, dim_k = k_glob.shape
        if v_glob.shape != (batch_k, len_k, dim_k):
            raise ValueError("k_glob and v_glob shapes must match")
        if dim_q != self.d_model or dim_k != self.d_model:
            raise ValueError("last dim of q/k_glob/v_glob must equal d_model")
        if batch_q != batch_k:
            raise ValueError("batch size of q and k_glob/v_glob must match")
        queries = split_heads(self.q_proj(q), self.nhead)
        keys = split_heads(k_glob, self.nhead)
        values = split_heads(v_glob, self.nhead)
        attn_mask = None
        if not _is_no_pad_mask(key_padding_mask):
            if key_padding_mask.shape != (batch_k, len_k):
                raise ValueError(
                    "key_padding_mask must have shape [B, Tk]=[%d, %d], got %s"
                    % (batch_k, len_k, tuple(key_padding_mask.shape))
                )
            # SDPA bool convention: True = take part in attention.
            attn_mask = (~key_padding_mask.to(dtype=torch.bool)).view(
                batch_k, 1, 1, len_k
            )
        dropout_p = self.dropout if self.training else 0.0
        y = _sdpa(queries, keys, values, attn_mask=attn_mask, dropout_p=dropout_p)
        out = self.out_proj(merge_heads(y))
        if self.training and self.dropout != 0.0:
            out = self.resid_dropout(out)
        return out
