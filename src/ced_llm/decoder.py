"""Causal decoder: layers with causal self-attn, global cross-attn, and GELU FFN.

Each decoder layer cross-attends to the shared global ``K_g``/``V_g`` that the
model projects exactly once from the final encoder states (CED KV-reuse).
Self-attention supports incremental decoding via an explicit per-layer
``(K, V)`` cache so autoregressive steps never rerun the encoder.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import (
    GlobalCrossAttention,
    _is_no_pad_mask,
    _qkv_param_key,
    _sdpa,
    causal_attend_mask,
    merge_heads,
    split_heads,
)
from .config import CEDConfig

# A per-layer self-attention KV cache: head-split keys/values, (K, V) each
# of shape [B, H, T_past, Dh], or None for an empty cache.
KVCache = Optional[Tuple[Tensor, Tensor]]


class DecoderLayer(nn.Module):
    """One decoder layer: causal self-attn -> global cross-attn -> FFN (all pre-norm)."""

    def __init__(self, config: CEDConfig) -> None:
        super().__init__()
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.head_dim = config.d_model // config.nhead
        self.dropout_p = config.dropout
        self.self_q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.self_k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.self_v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.self_out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.cross_attn = GlobalCrossAttention(
            config.d_model, config.nhead, config.dropout
        )
        self.ln_self = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln_cross = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln_ff = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.dim_ff)
        self.fc2 = nn.Linear(config.dim_ff, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        # Fused self-QKV (3->1 GEMM) + inference weight-stack cache.
        self.use_fused_qkv = bool(getattr(config, "use_fused_qkv", True))
        self._fused_weight: Optional[Tensor] = None
        self._fused_key: Optional[tuple] = None

    def _fused_qkv_weight(self) -> Optional[Tensor]:
        if not self.use_fused_qkv:
            return None
        # Training / grad-enabled: separate projections (original path).
        # Fresh fused cat per forward is pure overhead on tiny shapes;
        # separate linears carry identical grad flow. Fusion is inference-only.
        if self.training or torch.is_grad_enabled():
            return None
        try:
            key = _qkv_param_key(
                self.self_q_proj, self.self_k_proj, self.self_v_proj
            )
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
                [
                    self.self_q_proj.weight,
                    self.self_k_proj.weight,
                    self.self_v_proj.weight,
                ],
                dim=0,
            ).detach()
            self._fused_weight = fused
            self._fused_key = key
            return fused
        except Exception:
            return None

    def _qkv_self(self, h_norm: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Self Q/K/V un-split ``[B, T, D]`` (fused single-GEMM fast path)."""
        if self.use_fused_qkv:
            w = self._fused_qkv_weight()
            if w is not None:
                try:
                    qkv = F.linear(h_norm, w)
                    q, k, v = qkv.chunk(3, dim=-1)
                    return q, k, v
                except Exception:
                    pass
        return (
            self.self_q_proj(h_norm),
            self.self_k_proj(h_norm),
            self.self_v_proj(h_norm),
        )

    def _self_full(self, h_norm: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        """Full-sequence causal self-attention over ``h_norm`` -> ``[B, T, D]``."""
        q_u, k_u, v_u = self._qkv_self(h_norm)
        q = split_heads(q_u, self.nhead)
        k = split_heads(k_u, self.nhead)
        v = split_heads(v_u, self.nhead)
        dropout_p = self.dropout_p if self.training else 0.0
        if _is_no_pad_mask(key_padding_mask):
            # Fast path: fused causal kernel, no [B,1,T,T] alloc.
            y = _sdpa(q, k, v, is_causal=True, dropout_p=dropout_p)
        else:
            attn_mask = causal_attend_mask(key_padding_mask)  # type: ignore[arg-type]
            y = _sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)
        return self.self_out_proj(merge_heads(y))

    def _self_step(
        self,
        h_norm: Tensor,
        cache: Tuple[Tensor, Tensor],
        history_key_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Single-token self-attention with KV cache.

        Args:
            h_norm: Normalised current-token hidden state ``[B, 1, D]``.
            cache: ``(K_past, V_past)`` head-split caches, each
                ``[B, H, T_past, Dh]`` (empty past allowed with ``T_past == 0``).
            history_key_mask: Optional bool ``[B, T_past + 1]`` (``True`` = pad)
                covering past keys plus the current token.

        Returns:
            ``(attn_out [B, 1, D], (K_full, V_full))`` with the new K/V appended.
        """
        k_past, v_past = cache
        q_u, k_u, v_u = self._qkv_self(h_norm)
        q_new = split_heads(q_u, self.nhead)
        k_new = split_heads(k_u, self.nhead)
        v_new = split_heads(v_u, self.nhead)
        if k_past is None or k_past.size(2) == 0:
            k_full, v_full = k_new, v_new
        else:
            k_full = torch.cat([k_past, k_new], dim=2)
            v_full = torch.cat([v_past, v_new], dim=2)
        attn_mask = None
        if history_key_mask is not None:
            if history_key_mask.shape != (h_norm.size(0), k_full.size(2)):
                raise ValueError(
                    "history key mask must have shape [B, T_past+1]=[%d, %d], got %s"
                    % (
                        h_norm.size(0),
                        k_full.size(2),
                        tuple(history_key_mask.shape),
                    )
                )
            # Fast path: all-keep history -> attn_mask=None (no alloc).
            if not _is_no_pad_mask(history_key_mask):
                # SDPA bool convention: True = take part in attention.
                attn_mask = (~history_key_mask.to(dtype=torch.bool)).view(
                    h_norm.size(0), 1, 1, k_full.size(2)
                )
        # The single query is the newest position, so all cached keys are valid
        # past context: no causal mask needed beyond the (optional) key mask.
        dropout_p = self.dropout_p if self.training else 0.0
        y = _sdpa(q_new, k_full, v_full, attn_mask=attn_mask, dropout_p=dropout_p)
        return self.self_out_proj(merge_heads(y)), (k_full, v_full)

    def _self_step_static(
        self,
        h_norm: Tensor,
        k_buf: Optional[Tensor],
        v_buf: Optional[Tensor],
        cur_len: int,
        capacity: int,
        history_key_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single-token self-attention with STATIC preallocated KV buffers.

        Fill-in-place, no per-step ``cat`` realloc. ``k_buf``/``v_buf`` are
        ``[B, H, S, Dh]`` (``S == capacity``) or ``None`` on the first step
        (allocated here with ``torch.empty``). Position ``cur_len`` is filled
        via ``narrow(...).copy_`` and the prefix ``narrow(2, 0, cur_len+1)``
        view is used for attention (no data copy, bit-identical values).

        Returns ``(attn_out, k_buf, v_buf, k_full_view, v_full_view)``.
        """
        q_u, k_u, v_u = self._qkv_self(h_norm)
        q_new = split_heads(q_u, self.nhead)
        k_new = split_heads(k_u, self.nhead)
        v_new = split_heads(v_u, self.nhead)
        b, h, _, dh = q_new.shape
        # Allocate lazily so dtype/device exactly match the projections.
        if k_buf is None or v_buf is None:
            k_buf = torch.empty(
                b, h, int(capacity), dh,
                device=k_new.device, dtype=k_new.dtype,
            )
            v_buf = torch.empty(
                b, h, int(capacity), dh,
                device=v_new.device, dtype=v_new.dtype,
            )
        else:
            # Mismatched episode (batch/device/dtype/capacity changed):
            # reallocate when starting fresh (cur_len==0); otherwise the
            # caller changed batch mid-episode, which is a contract error.
            try:
                need_new = (
                    k_buf.size(0) != b
                    or k_buf.size(1) != h
                    or k_buf.size(2) != int(capacity)
                    or k_buf.size(3) != dh
                    or k_buf.device != k_new.device
                    or k_buf.dtype != k_new.dtype
                    or v_buf.size(0) != b
                    or v_buf.size(1) != h
                    or v_buf.size(2) != int(capacity)
                    or v_buf.size(3) != dh
                    or v_buf.device != v_new.device
                    or v_buf.dtype != v_new.dtype
                )
            except Exception:
                need_new = True
            if need_new:
                if int(cur_len) != 0:
                    raise ValueError(
                        "batch/device/dtype changed mid-episode (static KV mismatch)"
                    )
                k_buf = torch.empty(
                    b, h, int(capacity), dh,
                    device=k_new.device, dtype=k_new.dtype,
                )
                v_buf = torch.empty(
                    b, h, int(capacity), dh,
                    device=v_new.device, dtype=v_new.dtype,
                )
        # Fill-in-place (no realloc, no O(T) copy).
        try:
            k_buf.narrow(2, int(cur_len), 1).copy_(k_new)
            v_buf.narrow(2, int(cur_len), 1).copy_(v_new)
        except Exception:
            # Index error (cur_len >= capacity) -> let caller raise cleanly.
            raise ValueError("decode position exceeds max_seq_len")
        k_full = k_buf.narrow(2, 0, int(cur_len) + 1)
        v_full = v_buf.narrow(2, 0, int(cur_len) + 1)
        attn_mask = None
        if history_key_mask is not None:
            if history_key_mask.shape != (h_norm.size(0), k_full.size(2)):
                raise ValueError(
                    "history key mask must have shape [B, T_past+1]=[%d, %d], got %s"
                    % (
                        h_norm.size(0),
                        k_full.size(2),
                        tuple(history_key_mask.shape),
                    )
                )
            if not _is_no_pad_mask(history_key_mask):
                attn_mask = (~history_key_mask.to(dtype=torch.bool)).view(
                    h_norm.size(0), 1, 1, k_full.size(2)
                )
        dropout_p = self.dropout_p if self.training else 0.0
        y = _sdpa(q_new, k_full, v_full, attn_mask=attn_mask, dropout_p=dropout_p)
        return self.self_out_proj(merge_heads(y)), k_buf, v_buf, k_full, v_full

    def _ffn(self, h_norm: Tensor) -> Tensor:
        """Token-wise GELU feed-forward network."""
        # Skip dropout dispatch when identity.
        if self.training and self.dropout.p != 0.0:
            return self.fc2(self.dropout(self.act(self.fc1(h_norm))))
        return self.fc2(self.act(self.fc1(h_norm)))

    def forward(
        self,
        h: Tensor,
        k_glob: Tensor,
        v_glob: Tensor,
        self_padding_mask: Optional[Tensor] = None,
        glob_padding_mask: Optional[Tensor] = None,
        self_kv_cache: KVCache = None,
    ) -> Tuple[Tensor, KVCache]:
        """Run the layer in full-sequence or single-token (cached) mode.

        Args:
            h: Decoder hidden states ``[B, T, D]`` (full mode) or ``[B, 1, D]``
                (incremental step mode).
            k_glob: Shared global keys ``[B, Tk, D]`` (projected once by the model).
            v_glob: Shared global values ``[B, Tk, D]``.
            self_padding_mask: Full mode: bool ``[B, T]`` (``True`` = pad).
                Step mode: optional bool history mask ``[B, T_past + 1]``.
            glob_padding_mask: Optional bool ``[B, Tk]`` (``True`` = pad encoder key).
            self_kv_cache: ``None`` for full-sequence training, else the
                ``(K_past, V_past)`` tuple for an incremental step
                (either entry may itself be ``None`` when empty).

        Returns:
            ``(out, new_kv)`` where ``out`` matches ``h``'s shape and ``new_kv``
            is ``None`` in full mode or the updated ``(K, V)`` cache in step mode.
        """
        # Local flag avoids repeated attribute/dispatch cost and keeps the
        # math identical (dropout with p==0 or in eval is identity).
        use_drop = self.training and self.dropout.p != 0.0
        if self_kv_cache is None:
            sa = self._self_full(self.ln_self(h), self_padding_mask)
            h = h + (self.dropout(sa) if use_drop else sa)
            ca = self.cross_attn(self.ln_cross(h), k_glob, v_glob, glob_padding_mask)
            h = h + (self.dropout(ca) if use_drop else ca)
            ff = self._ffn(self.ln_ff(h))
            h = h + (self.dropout(ff) if use_drop else ff)
            return h, None
        if h.size(1) != 1:
            raise ValueError(
                "incremental step mode requires h of shape [B, 1, D], got %s"
                % (tuple(h.shape),)
            )
        k_past, v_past = self_kv_cache
        sa, new_kv = self._self_step(self.ln_self(h), (k_past, v_past), self_padding_mask)
        h = h + (self.dropout(sa) if use_drop else sa)
        ca = self.cross_attn(self.ln_cross(h), k_glob, v_glob, glob_padding_mask)
        h = h + (self.dropout(ca) if use_drop else ca)
        ff = self._ffn(self.ln_ff(h))
        h = h + (self.dropout(ff) if use_drop else ff)
        return h, new_kv

    def forward_static(
        self,
        h: Tensor,
        k_glob: Tensor,
        v_glob: Tensor,
        self_padding_mask: Optional[Tensor],
        glob_padding_mask: Optional[Tensor],
        k_buf: Optional[Tensor],
        v_buf: Optional[Tensor],
        cur_len: int,
        capacity: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Single-token layer step with STATIC buffers (fill-in-place).

        Returns ``(out, k_buf, v_buf, k_view, v_view)``; buffers are mutated
        (allocated when ``None``) and views share their storage.
        """
        if h.size(1) != 1:
            raise ValueError(
                "incremental step mode requires h of shape [B, 1, D], got %s"
                % (tuple(h.shape),)
            )
        use_drop = self.training and self.dropout.p != 0.0
        sa, k_buf, v_buf, k_view, v_view = self._self_step_static(
            self.ln_self(h), k_buf, v_buf, int(cur_len), int(capacity),
            self_padding_mask,
        )
        h = h + (self.dropout(sa) if use_drop else sa)
        ca = self.cross_attn(self.ln_cross(h), k_glob, v_glob, glob_padding_mask)
        h = h + (self.dropout(ca) if use_drop else ca)
        ff = self._ffn(self.ln_ff(h))
        h = h + (self.dropout(ff) if use_drop else ff)
        return h, k_buf, v_buf, k_view, v_view


class CausalDecoder(nn.Module):
    """Stack of :class:`DecoderLayer` blocks plus a final LayerNorm."""

    def __init__(self, config: CEDConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_dec_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self,
        h: Tensor,
        k_glob: Tensor,
        v_glob: Tensor,
        self_padding_mask: Optional[Tensor] = None,
        glob_padding_mask: Optional[Tensor] = None,
        self_kv_cache: Optional[List[KVCache]] = None,
    ) -> Tuple[Tensor, Optional[List[KVCache]]]:
        """Run the decoder stack in full-sequence or incremental mode.

        Args:
            h: ``[B, T, D]`` hidden states (full mode) or ``[B, 1, D]`` (step mode).
            k_glob: Shared global keys ``[B, Tk, D]``.
            v_glob: Shared global values ``[B, Tk, D]``.
            self_padding_mask: Full mode ``[B, T]`` pad mask; step mode optional
                ``[B, T_past + 1]`` history key mask (``True`` = pad in both).
            glob_padding_mask: Optional bool ``[B, Tk]`` encoder pad mask.
            self_kv_cache: ``None`` for full-sequence mode, else one
                ``(K, V)`` cache tuple per layer for incremental decoding.

        Returns:
            ``(out, new_caches)`` with ``new_caches`` ``None`` in full mode or
            the updated per-layer cache list in step mode.
        """
        if self_kv_cache is None:
            for layer in self.layers:
                h, _ = layer(h, k_glob, v_glob, self_padding_mask, glob_padding_mask)
            return self.final_norm(h), None
        if len(self_kv_cache) != len(self.layers):
            raise ValueError(
                "expected %d per-layer caches, got %d"
                % (len(self.layers), len(self_kv_cache))
            )
        new_caches: List[KVCache] = []
        for layer, cache in zip(self.layers, self_kv_cache):
            h, new_kv = layer(
                h, k_glob, v_glob, self_padding_mask, glob_padding_mask, cache
            )
            new_caches.append(new_kv)
        return self.final_norm(h), new_caches

    def forward_static(
        self,
        h: Tensor,
        k_glob: Tensor,
        v_glob: Tensor,
        self_padding_mask: Optional[Tensor],
        glob_padding_mask: Optional[Tensor],
        k_bufs: List[Optional[Tensor]],
        v_bufs: List[Optional[Tensor]],
        cur_len: int,
        capacity: int,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """Stack step with STATIC per-layer buffers (no per-step cat).

        ``k_bufs``/``v_bufs`` are mutated in place (allocated when ``None``).
        Returns ``(out, new_k_views, new_v_views)`` where views are
        ``[B, H, cur_len+1, Dh]`` sharing buffer storage (bit-identical vals).
        """
        if len(k_bufs) != len(self.layers) or len(v_bufs) != len(self.layers):
            raise ValueError(
                "expected %d per-layer static buffers, got %d/%d"
                % (len(self.layers), len(k_bufs), len(v_bufs))
            )
        if h.size(1) != 1:
            raise ValueError(
                "incremental step mode requires h of shape [B, 1, D], got %s"
                % (tuple(h.shape),)
            )
        new_k_views: List[Tensor] = []
        new_v_views: List[Tensor] = []
        for i, layer in enumerate(self.layers):
            h, k_b, v_b, k_v, v_v = layer.forward_static(
                h, k_glob, v_glob, self_padding_mask, glob_padding_mask,
                k_bufs[i], v_bufs[i], int(cur_len), int(capacity),
            )
            k_bufs[i] = k_b
            v_bufs[i] = v_b
            new_k_views.append(k_v)
            new_v_views.append(v_v)
        return self.final_norm(h), new_k_views, new_v_views
