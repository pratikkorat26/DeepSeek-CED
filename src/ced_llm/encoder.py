"""Causal encoder: stack of pre-norm Transformer blocks with dense self-attention."""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import CausalSelfAttention
from .config import CEDConfig


class _EncoderBlock(nn.Module):
    """One pre-norm block: causal self-attn + GELU FFN, each with residual + dropout."""

    def __init__(self, config: CEDConfig) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.self_attn = CausalSelfAttention(
            config.d_model,
            config.nhead,
            config.dropout,
            fused_qkv=bool(getattr(config, "use_fused_qkv", True)),
        )
        self.ln_ff = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.d_model, config.dim_ff)
        self.fc2 = nn.Linear(config.dim_ff, config.d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Apply the block: ``x + drop(attn(ln(x)))`` then ``h + drop(ffn(ln(h)))``."""
        # Skip dropout dispatch when it is identity (p==0 or eval).
        if self.training and self.dropout.p != 0.0:
            x = x + self.dropout(self.self_attn(self.ln_attn(x), key_padding_mask))
            h = self.ln_ff(x)
            h = self.fc2(self.dropout(self.act(self.fc1(h))))
            return x + self.dropout(h)
        x = x + self.self_attn(self.ln_attn(x), key_padding_mask)
        h = self.ln_ff(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h


class CausalEncoder(nn.Module):
    """Stack of causal pre-norm Transformer blocks.

    The encoder runs exactly once per input ("encode once"); its final states
    are projected a single time to the global ``K_g``/``V_g`` reused by every
    decoder layer and every autoregressive step.
    """

    def __init__(self, config: CEDConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [_EncoderBlock(config) for _ in range(config.n_enc_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(
        self, x_emb: Tensor, key_padding_mask: Optional[Tensor] = None
    ) -> Tensor:
        """Encode embedded inputs.

        Args:
            x_emb: Token + position embeddings of shape ``[B, T, D]``.
            key_padding_mask: Optional bool ``[B, T]`` with ``True`` = pad.

        Returns:
            Encoded states of shape ``[B, T, D]``.
        """
        h = x_emb
        for layer in self.layers:
            h = layer(h, key_padding_mask)
        return self.final_norm(h)
