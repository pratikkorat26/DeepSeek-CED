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

    def _self_full(self, h_norm: Tensor, key_padding_mask: Optional[Tensor]) -> Tensor:
        """Full-sequence causal self-attention over ``h_norm`` -> ``[B, T, D]``."""
        q = split_heads(self.self_q_proj(h_norm), self.nhead)
        k = split_heads(self.self_k_proj(h_norm), self.nhead)
        v = split_heads(self.self_v_proj(h_norm), self.nhead)
        dropout_p = self.dropout_p if self.training else 0.0
        if _is_no_pad_mask(key_padding_mask):
            # Fast path: fused causal kernel, no [B,1,T,T] alloc.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            attn_mask = causal_attend_mask(key_padding_mask)
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
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
        q_new = split_heads(self.self_q_proj(h_norm), self.nhead)
        k_new = split_heads(self.self_k_proj(h_norm), self.nhead)
        v_new = split_heads(self.self_v_proj(h_norm), self.nhead)
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
        y = F.scaled_dot_product_attention(
            q_new, k_full, v_full, attn_mask=attn_mask, dropout_p=dropout_p
        )
        return self.self_out_proj(merge_heads(y)), (k_full, v_full)

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
