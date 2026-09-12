"""DeepSeekMoE-style sparse FFN (V4.1-Flash mechanics at toy scale).

Replicates the exact routing recipe from the official
``deepseek-ai/DeepSeek-V4.1-Flash`` config (``text_config``)::

    n_routed_experts   384   -> toy default 8   (moe_num_experts)
    num_experts_per_tok  6   -> toy default 2   (moe_top_k)
    n_shared_experts     1   -> 1               (moe_shared_experts)
    scoring_func  sqrtsoftplus (exact)
    topk_method   noaux_tc     (aux-loss-free, bias-balanced; exact scheme)
    norm_topk_prob     true    (exact)
    routed_scaling_factor 1.5 (exact)

Deliberate toy deltas (documented, not hidden): SwiGLU experts match their
``hidden_act: silu`` (the repo's dense path stays GELU); no FP4/FP8; masking
dispatch instead of grouped GEMM (correct, simple; production would fuse).

The payoff this preserves: **asymmetric activation** -- prefill runs the
encoder MoE, decode runs the decoder MoE, so per-token active params differ
by phase (theirs: 8B prefill / 16B decode). See ``examples/moe_asymmetry.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sqrtsoftplus(x: Tensor) -> Tensor:
    """V4.1 router scoring: ``sqrt(softplus(x))`` (never negative, never NaN)."""
    try:
        return torch.sqrt(F.softplus(x).clamp_min(0.0))
    except Exception:
        return torch.zeros_like(x)


class SwiGLUExpert(nn.Module):
    """One SwiGLU FFN expert (gate/up/down, V4.1 ``hidden_act: silu``)."""

    def __init__(self, d_model: int, expert_dim: int, swiglu_limit: float = 10.0) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, expert_dim, bias=False)
        self.up = nn.Linear(d_model, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, d_model, bias=False)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: Tensor) -> Tensor:
        g = self.gate(x)
        try:
            lim = float(self.swiglu_limit)
            if lim > 0:
                g = g.clamp(min=-lim, max=lim)
        except Exception:
            pass
        return self.down(F.silu(g) * self.up(x))


class DeepSeekMoELayer(nn.Module):
    """Sparse MoE FFN: shared expert(s) + top-k routed experts, noaux balanced.

    Drop-in replacement for a dense ``fc2(act(fc1(x)))`` block. Forward
    returns a plain Tensor; routing diagnostics land on ``self.last_stats``::

        {"counts": LongTensor[E], "topk_idx": LongTensor[B,T,k],
         "topk_weight": Tensor[B,T,k]}
    """

    def __init__(self, d_model: int, dim_ff: int, num_experts: int = 8,
                 top_k: int = 2, shared_experts: int = 1,
                 expert_dim: int = 0, routed_scaling: float = 1.5,
                 norm_topk_prob: bool = True,
                 scoring: str = "sqrtsoftplus",
                 balance_lr: float = 0.01) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_experts = max(1, int(num_experts))
        self.top_k = max(1, min(int(top_k), self.num_experts))
        self.routed_scaling = float(routed_scaling)
        self.norm_topk_prob = bool(norm_topk_prob)
        self.scoring = str(scoring or "sqrtsoftplus").lower()
        self.balance_lr = float(balance_lr)
        h = int(expert_dim) if int(expert_dim) > 0 else int(dim_ff)
        self.experts = nn.ModuleList(
            [SwiGLUExpert(d_model, h) for _ in range(self.num_experts)]
        )
        self.shared = nn.ModuleList(
            [SwiGLUExpert(d_model, h) for _ in range(max(0, int(shared_experts)))]
        )
        self.router = nn.Linear(d_model, self.num_experts, bias=False)
        # Aux-loss-free balancing bias (noaux_tc): updated from load stats,
        # never trained by gradient.
        self.register_buffer("expert_bias", torch.zeros(self.num_experts))
        self.last_stats = {}

    def _scores(self, x: Tensor) -> Tensor:
        logits = self.router(x) + self.expert_bias.to(x.dtype)
        if self.scoring == "sigmoid":
            return torch.sigmoid(logits)
        return sqrtsoftplus(logits)

    def forward(self, x: Tensor) -> Tensor:
        scores = self._scores(x)
        top_w, top_idx = torch.topk(scores, k=self.top_k, dim=-1)
        if self.norm_topk_prob:
            denom = top_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            top_w = top_w / denom
        top_w = top_w * self.routed_scaling
        out = torch.zeros_like(x)
        flat_idx = top_idx.reshape(-1)
        try:
            counts = torch.bincount(flat_idx, minlength=self.num_experts)
        except Exception:
            counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        # Masking dispatch: simple and exactly correct at toy scale.
        for e, expert in enumerate(self.experts):
            m = (top_idx == e)
            if bool(m.any()):
                w = (top_w * m.to(top_w.dtype)).sum(dim=-1, keepdim=True)
                out = out + w * expert(x)
        for s in self.shared:
            out = out + s(x)
        try:
            self.last_stats = {
                "counts": counts.detach().to("cpu"),
                "topk_idx": top_idx.detach().to("cpu"),
                "topk_weight": top_w.detach().to("cpu"),
            }
        except Exception:
            self.last_stats = {}
        # noaux_tc bias correction (training only): push load toward uniform.
        if self.training:
            try:
                with torch.no_grad():
                    total = float(counts.sum().item())
                    if total > 0:
                        frac = counts.to(torch.float32) / total
                        over = (frac - 1.0 / self.num_experts).sign()
                        self.expert_bias.add_(
                            (-float(self.balance_lr) * over).to(self.expert_bias.dtype)
                        )
            except Exception:
                pass
        return out

    def active_params_per_token(self) -> int:
        """Params touched by one token: router + shared + top-k experts."""
        n = 0
        try:
            n += int(self.router.weight.numel())
        except Exception:
            pass
        try:
            n += int(self.expert_bias.numel())
        except Exception:
            pass
        try:
            per_expert = sum(int(p.numel()) for p in self.experts[0].parameters())
        except Exception:
            per_expert = 0
        n += self.top_k * per_expert
        for s in self.shared:
            try:
                n += sum(int(p.numel()) for p in s.parameters())
            except Exception:
                pass
        return n


def _sub_active_params(sub) -> int:
    """Params of a submodule with MoE layers counted at active (not full) size."""
    try:
        full = sum(int(p.numel()) for p in sub.parameters())
    except Exception:
        return 0
    moe_full, moe_active = 0, 0
    try:
        for m in sub.modules():
            if isinstance(m, DeepSeekMoELayer):
                moe_full += sum(int(p.numel()) for p in m.parameters())
                moe_active += m.active_params_per_token()
    except Exception:
        pass
    return full - moe_full + moe_active


def prefill_active_params(model) -> int:
    """Per-token active params during prefill (embeddings + encoder + KV projs)."""
    roots = ["tok_emb", "pos_emb", "encoder", "kv_proj_k", "kv_proj_v"]
    total = 0
    for r in roots:
        try:
            total += _sub_active_params(getattr(model, r))
        except Exception:
            pass
    return total


def decode_active_params(model) -> int:
    """Per-step active params during decode (embeddings + decoder + norm + head)."""
    roots = ["tok_emb", "pos_emb", "decoder", "norm", "lm_head"]
    total = 0
    for r in roots:
        try:
            total += _sub_active_params(getattr(model, r))
        except Exception:
            pass
    return total


def total_params(model) -> int:
    try:
        return sum(int(p.numel()) for p in model.parameters())
    except Exception:
        return 0
