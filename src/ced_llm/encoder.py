"""Causal encoder: stack of pre-norm Transformer blocks with dense self-attention."""

from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import CausalSelfAttention
from .config import CEDConfig

try:
    from .moe import DeepSeekMoELayer
except Exception:
    DeepSeekMoELayer = None


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
        # DeepSeekMoE FFN (V4.1 recipe); None == dense path, bit-identical.
        self.moe_ffn = None
        if DeepSeekMoELayer is not None and bool(getattr(config, "moe_enabled", False)):
            try:
                self.moe_ffn = DeepSeekMoELayer(
                    config.d_model, config.dim_ff,
                    num_experts=int(getattr(config, "moe_num_experts", 8)),
                    top_k=int(getattr(config, "moe_top_k", 2)),
                    shared_experts=int(getattr(config, "moe_shared_experts", 1)),
                    expert_dim=int(getattr(config, "moe_expert_dim", 0)),
                    routed_scaling=float(getattr(config, "moe_routed_scaling", 1.5)),
                    norm_topk_prob=bool(getattr(config, "moe_norm_topk_prob", True)),
                    scoring=str(getattr(config, "moe_scoring", "sqrtsoftplus")),
                    balance_lr=float(getattr(config, "moe_balance_lr", 0.01)),
                )
            except Exception:
                self.moe_ffn = None

    def _ffn(self, h: Tensor) -> Tensor:
        """Dense GELU FFN, or DeepSeekMoE when enabled (dropout applied by caller)."""
        if self.moe_ffn is not None:
            return self.moe_ffn(h)
        return self.fc2(self.act(self.fc1(h)))

    def forward(self, x: Tensor, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Apply the block: ``x + drop(attn(ln(x)))`` then ``h + drop(ffn(ln(h)))``."""
        # Skip dropout dispatch when it is identity (p==0 or eval).
        if self.training and self.dropout.p != 0.0:
            x = x + self.dropout(self.self_attn(self.ln_attn(x), key_padding_mask))
            h = self.ln_ff(x)
            if self.moe_ffn is not None:
                h = self.dropout(self.moe_ffn(h))
            else:
                h = self.fc2(self.dropout(self.act(self.fc1(h))))
            return x + self.dropout(h)
        x = x + self.self_attn(self.ln_attn(x), key_padding_mask)
        h = self.ln_ff(x)
        h = self._ffn(h)
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
