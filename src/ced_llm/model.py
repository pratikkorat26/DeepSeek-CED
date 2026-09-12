"""Dense Causal Encoder-Decoder (CED) language model.

Architecture (v1, dense only):

* The token sequence is embedded (token + learned position) and encoded once
  by a causal Transformer encoder.
* The final encoder states are projected exactly once to a shared global key
  ``K_g`` and value ``V_g`` (``kv_proj_k`` / ``kv_proj_v``).
* A causal decoder attends to its own prefix with causal self-attention and to
  the full encoder context with non-causal cross-attention over the shared
  ``K_g``/``V_g``.

CED KV-reuse: during autoregressive generation the encoder runs a single time
(``init_decode_cache`` -> ``encode_once``); every ``forward_step`` reuses the
identical ``k_glob``/``v_glob`` tensors (object identity preserved) together
with per-layer self-attention KV caches, so the encoder cost is paid once no
matter how many tokens are generated. ``encoder_forward_count`` tracks encoder
executions to make this assertable (it must stay at 1 across generation steps).

Loss alignment (caller shifts): ``forward`` returns logits aligned 1:1 with
the ``input_ids`` positions. When ``labels`` are provided, the loss is
``CrossEntropyLoss(logits, labels)`` with ``ignore_index=pad_token_id`` over
all positions. Callers (e.g. ``train.py``) perform the usual LM shift
themselves: ``model(input_ids=tok[:, :-1], labels=tok[:, 1:])``.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import CEDConfig, resolve_dtype
from .decoder import CausalDecoder, KVCache
from .encoder import CausalEncoder


class CEDForLM(nn.Module):
    """Causal Encoder-Decoder transformer for language modelling."""

    def __init__(self, config: CEDConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        # Dtype policy: default fp32 (bit-identical). Validates flag early.
        try:
            self._resolved_dtype = resolve_dtype("cpu", getattr(config, "dtype", "fp32"))
        except Exception:
            self._resolved_dtype = None
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.encoder = CausalEncoder(config)
        self.kv_proj_k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv_proj_v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.decoder = CausalDecoder(config)
        self.norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.encoder_forward_count: int = 0
        # Cache of position-row tensors keyed by (T, device-str) to avoid
        # re-allocating torch.arange on every _embed call.
        self._pos_cache: dict = {}
        # Cache of converted pad masks keyed by id(tensor) -> (ref, result).
        # Fixed toy loaders reuse the same mask objects, so hits avoid a
        # per-forward ``==0`` alloc + ``.any()`` device sync.
        self._pad_cache: dict = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def reset_encoder_counter(self) -> None:
        """Reset :attr:`encoder_forward_count` to zero."""
        self.encoder_forward_count = 0

    def _check_length(self, seq_len: int) -> None:
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                "sequence length %d exceeds max_seq_len %d" % (seq_len, self.config.max_seq_len)
            )

    def _positions(self, seq_len: int, device: torch.device) -> Tensor:
        # Cached arange row [1, T]; reuse avoids an alloc per _embed call.
        try:
            key = (int(seq_len), str(device))
            hit = self._pos_cache.get(key)
            if hit is not None:
                try:
                    if hit.device == device and hit.shape == (1, seq_len):
                        return hit
                except Exception:
                    pass
            fresh = torch.arange(seq_len, device=device).unsqueeze(0)
            try:
                if len(self._pos_cache) < 16:
                    self._pos_cache[key] = fresh
            except Exception:
                pass
            return fresh
        except Exception:
            return torch.arange(seq_len, device=device).unsqueeze(0)

    def _embed(self, input_ids: Tensor) -> Tensor:
        """Token + learned-position embeddings for ``input_ids`` ``[B, T]``."""
        batch, seq_len = input_ids.shape
        self._check_length(seq_len)
        pos = self._positions(seq_len, input_ids.device).expand(batch, seq_len)
        return self.tok_emb(input_ids) + self.pos_emb(pos)

    @staticmethod
    def _pad_mask(attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Convert a ``1=keep/0=pad`` mask to bool ``True=pad`` (or ``None``).

        Fast path: when the mask has no pads, return ``None`` (semantically
        identical to an all-False mask) so attention uses the fused
        ``is_causal=True`` / ``attn_mask=None`` kernels without allocating
        ``[B, 1, T, T]`` masks.
        """
        if attention_mask is None:
            return None
        try:
            m = attention_mask == 0
            try:
                if not bool(m.any()):
                    return None
            except Exception:
                pass
            return m
        except Exception:
            try:
                return attention_mask == 0
            except Exception:
                return None

    def _pad_mask_cached(self, attention_mask: Optional[Tensor]) -> Optional[Tensor]:
        """Cached ``_pad_mask`` for reused mask objects (no per-forward sync).

        Keys by ``id(tensor)`` holding a strong ref to prevent ABA reuse.
        Bit-identical: returns the same ``None``-vs-bool semantics as
        :meth:`_pad_mask`; converted bool tensors are read-only and shared.
        """
        if attention_mask is None:
            return None
        try:
            key = id(attention_mask)
            hit = self._pad_cache.get(key)
            if hit is not None:
                ref, res = hit
                if ref is attention_mask:
                    return res
            res = self._pad_mask(attention_mask)
            try:
                if len(self._pad_cache) >= 32:
                    # Evict oldest (dict preserves insertion order).
                    try:
                        self._pad_cache.pop(next(iter(self._pad_cache)))
                    except Exception:
                        self._pad_cache.clear()
                self._pad_cache[key] = (attention_mask, res)
            except Exception:
                pass
            return res
        except Exception:
            return self._pad_mask(attention_mask)

    def _encode_embedded(
        self, x_emb: Tensor, key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        """Run the encoder once, incrementing :attr:`encoder_forward_count`."""
        self.encoder_forward_count += 1
        return self.encoder(x_emb, key_padding_mask)

    def encode_once(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """Encode ``input_ids`` exactly once and project the shared global KV.

        Args:
            input_ids: Long tensor ``[B, T]``.
            attention_mask: Optional ``[B, T]`` with ``1=keep/0=pad``.

        Returns:
            Dict with ``h_enc`` (final encoder states ``[B, T, D]``),
            ``k_glob`` and ``v_glob`` (shared global KV ``[B, T, D]``).
            Increments :attr:`encoder_forward_count` by exactly 1.
        """
        h_enc = self._encode_embedded(
            self._embed(input_ids), self._pad_mask_cached(attention_mask)
        )
        return {
            "h_enc": h_enc,
            "k_glob": self.kv_proj_k(h_enc),
            "v_glob": self.kv_proj_v(h_enc),
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        """Full-sequence forward for training/evaluation.

        Logits are aligned 1:1 with ``input_ids`` positions (caller shifts:
        pass ``input_ids=tok[:, :-1]`` with ``labels=tok[:, 1:]`` for standard
        next-token LM training). If ``labels`` are given, the loss is plain
        cross-entropy over all positions with ``ignore_index=pad_token_id``.

        Args:
            input_ids: Long tensor ``[B, T]``.
            attention_mask: Optional ``[B, T]`` with ``1=keep/0=pad``.
            labels: Optional long tensor ``[B, T]`` aligned to ``input_ids``.

        Returns:
            ``{"logits": [B, T, V], "loss": scalar or None}``.
        """
        pad_mask = self._pad_mask_cached(attention_mask)
        # Single shared embedding: encoder and decoder read identical
        # token+pos vectors, so compute once instead of twice.
        x_emb = self._embed(input_ids)
        h_enc = self._encode_embedded(x_emb, pad_mask)
        k_glob = self.kv_proj_k(h_enc)
        v_glob = self.kv_proj_v(h_enc)
        dec_out, _ = self.decoder(x_emb, k_glob, v_glob, pad_mask, pad_mask)
        logits = self.lm_head(self.norm(dec_out))
        loss: Optional[Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
        return {"logits": logits, "loss": loss}

    def init_decode_cache(
        self, input_ids: Tensor, attention_mask: Optional[Tensor] = None
    ) -> Dict[str, Any]:
        """Encode the prompt once and build an empty incremental-decode cache.

        Calls :meth:`encode_once` internally, so :attr:`encoder_forward_count`
        increases by exactly 1. Per-layer self-attention caches start empty
        (``None``); positions for subsequent :meth:`forward_step` calls begin
        at 0, so replaying ``input_ids`` token-by-token reproduces the full
        forward pass exactly.

        Note: this starts a fresh generation episode, so the counter is reset
        to 0 before the single internal :meth:`encode_once` call; afterwards
        the count is exactly 1 no matter how many full forwards ran before.

        Args:
            input_ids: Long prompt tensor ``[B, T_enc]``.
            attention_mask: Optional ``[B, T_enc]`` with ``1=keep/0=pad``.

        Returns:
            Cache dict with ``k_glob``/``v_glob`` (shared global KV),
            ``self_k_list``/``self_v_list`` (``None`` per decoder layer),
            ``enc_ids``, ``enc_mask`` and ``T_enc``.
        """
        self.encoder_forward_count = 0
        enc = self.encode_once(input_ids, attention_mask)
        n_layers = len(self.decoder.layers)
        # Hoisted decode invariants: glob mask computed once (no per-step
        # ``_pad_mask`` alloc+sync), batch + reusable pos buffer, static KV
        # buffers (lazy-allocated on first step to match exact dtype/device).
        try:
            glob_mask = self._pad_mask_cached(attention_mask)
        except Exception:
            glob_mask = None
        try:
            _b = int(input_ids.size(0))
        except Exception:
            _b = 1
        try:
            _pos_buf = torch.empty(
                _b, 1, dtype=torch.long, device=input_ids.device
            ).fill_(0)
        except Exception:
            _pos_buf = None
        try:
            _use_static = bool(getattr(self.config, "use_static_cache", True))
        except Exception:
            _use_static = True
        return {
            "k_glob": enc["k_glob"],
            "v_glob": enc["v_glob"],
            "self_k_list": [None] * n_layers,
            "self_v_list": [None] * n_layers,
            "enc_ids": input_ids,
            "enc_mask": attention_mask,
            "T_enc": input_ids.size(1),
            # --- FORGE-MODEL static-decode state (backward-compat extras) ---
            "self_k_buf": [None] * n_layers,
            "self_v_buf": [None] * n_layers,
            "self_decode_len": 0,
            "self_static_capacity": int(self.config.max_seq_len),
            "glob_mask": glob_mask,
            "batch": _b,
            "_pos_buf": _pos_buf,
            "use_static": _use_static,
        }

    def forward_step(
        self,
        next_ids: Tensor,
        cache: Dict[str, Any],
        attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Single-token incremental decode step reusing the encoder KV.

        Never runs the encoder (leaves :attr:`encoder_forward_count`
        unchanged) and never recomputes ``k_glob``/``v_glob`` (their object
        identity in ``cache`` is preserved). Appends the new per-layer self
        K/V to ``cache["self_k_list"]`` / ``cache["self_v_list"]`` in place.

        Args:
            next_ids: Long tensor ``[B, 1]`` with the current token(s).
            cache: Dict created by :meth:`init_decode_cache` (mutated in place).
            attention_mask: Optional decode-history mask ``[B, t + 1]`` with
                ``1=keep/0=pad`` covering cached tokens plus the current one
                (used as the self-attention key mask, so padded replays match
                the full forward pass). A ``[B, 1]`` mask or ``None`` means
                "no masking" (decode-stream tokens are assumed non-pad).

        Returns:
            ``(logits [B, 1, V], cache)`` with ``cache`` the same (mutated) dict.
        """
        # --- Hoisted per-token validation (cheap int compares, no sync) ---
        if next_ids.dim() != 2 or next_ids.size(1) != 1:
            raise ValueError(
                "next_ids must have shape [B, 1], got %s" % (tuple(next_ids.shape),)
            )
        k_glob: Tensor = cache["k_glob"]
        v_glob: Tensor = cache["v_glob"]
        batch = next_ids.size(0)
        if k_glob.size(0) != batch or v_glob.size(0) != batch:
            raise ValueError("batch size of next_ids and cache must match")
        # Hoist config attr + decode length (cached int, no tensor size op).
        max_len = self.config.max_seq_len
        try:
            _dl = cache.get("self_decode_len", None)
        except Exception:
            _dl = None
        if _dl is None:
            # Backward compat: old caches without static length counter.
            try:
                self_k_list_old: List[KVCache] = cache["self_k_list"]
                first_k = self_k_list_old[0] if len(self_k_list_old) else None
            except Exception:
                first_k = None
            cur_len = 0 if first_k is None else int(first_k.size(2))
        else:
            try:
                cur_len = int(_dl)
            except Exception:
                cur_len = 0
        if cur_len + 1 > max_len:
            raise ValueError("decode position exceeds max_seq_len")
        # --- Reusable pos buffer (no per-step torch.tensor alloc) ---
        try:
            pos_buf = cache.get("_pos_buf", None)
        except Exception:
            pos_buf = None
        if (
            pos_buf is not None
            and getattr(pos_buf, "shape", None) is not None
            and tuple(pos_buf.shape) == (batch, 1)
            and getattr(pos_buf, "device", None) == next_ids.device
        ):
            try:
                pos_buf.fill_(int(cur_len))
                pos = pos_buf
            except Exception:
                pos = torch.tensor([[cur_len]], device=next_ids.device).expand(
                    batch, 1
                )
        else:
            try:
                fresh_pos = torch.empty(
                    batch, 1, dtype=torch.long, device=next_ids.device
                ).fill_(int(cur_len))
                try:
                    cache["_pos_buf"] = fresh_pos
                except Exception:
                    pass
                pos = fresh_pos
            except Exception:
                pos = torch.tensor([[cur_len]], device=next_ids.device).expand(
                    batch, 1
                )
        # Local embedding refs (avoid repeated attribute dispatch).
        tok_emb = self.tok_emb
        pos_emb = self.pos_emb
        h = tok_emb(next_ids) + pos_emb(pos)

        history_key_mask: Optional[Tensor] = None
        if attention_mask is not None and attention_mask.size(1) == cur_len + 1:
            history_key_mask = attention_mask == 0
        # Hoisted glob mask (computed once at init, no per-step sync).
        try:
            _gm_sentinel = object()
            glob_mask = cache.get("glob_mask", _gm_sentinel)
            if glob_mask is _gm_sentinel:
                glob_mask = self._pad_mask_cached(cache.get("enc_mask"))
                try:
                    cache["glob_mask"] = glob_mask
                except Exception:
                    pass
        except Exception:
            try:
                glob_mask = self._pad_mask_cached(cache.get("enc_mask"))
            except Exception:
                glob_mask = None

        # --- STATIC preallocated KV path (fill-in-place, default) ---
        try:
            use_static = bool(
                cache.get(
                    "use_static",
                    bool(getattr(self.config, "use_static_cache", True)),
                )
            )
        except Exception:
            use_static = True
        has_static_bufs = False
        try:
            has_static_bufs = (
                "self_k_buf" in cache
                and "self_v_buf" in cache
                and "self_decode_len" in cache
            )
        except Exception:
            has_static_bufs = False
        if use_static and has_static_bufs:
            try:
                k_bufs = cache["self_k_buf"]
                v_bufs = cache["self_v_buf"]
                try:
                    capacity = int(
                        cache.get("self_static_capacity", max_len)
                    )
                except Exception:
                    capacity = int(max_len)
                if cur_len + 1 > capacity:
                    raise ValueError("decode position exceeds max_seq_len")
                dec_out, new_k_views, new_v_views = self.decoder.forward_static(
                    h, k_glob, v_glob, history_key_mask, glob_mask,
                    k_bufs, v_bufs, int(cur_len), int(capacity),
                )
                # Compat views (share buffer storage, no copy) + length bump.
                try:
                    cache["self_k_list"] = new_k_views
                    cache["self_v_list"] = new_v_views
                    cache["self_decode_len"] = int(cur_len) + 1
                except Exception:
                    pass
                logits = self.lm_head(self.norm(dec_out))
                return logits, cache
            except ValueError:
                raise
            except Exception:
                # Fall through to cat path on unexpected static failure
                # (preserves correctness; static bugs never break parity).
                pass

        # --- Legacy cat path (old caches or use_static=False) ---
        self_k_list: List[KVCache] = cache["self_k_list"]
        self_v_list: List[KVCache] = cache["self_v_list"]
        layer_caches: List[KVCache] = []
        for k, v in zip(self_k_list, self_v_list):
            # NOTE: an empty cache is the (None, None) tuple (step mode),
            # never bare None (which means full-sequence mode).
            layer_caches.append((k, v))
        dec_out, new_caches = self.decoder(
            h, k_glob, v_glob, history_key_mask, glob_mask, layer_caches
        )
        assert new_caches is not None
        for i, new_kv in enumerate(new_caches):
            assert new_kv is not None
            self_k_list[i], self_v_list[i] = new_kv
        try:
            if "self_decode_len" in cache:
                cache["self_decode_len"] = int(cur_len) + 1
        except Exception:
            pass
        logits = self.lm_head(self.norm(dec_out))
        return logits, cache
