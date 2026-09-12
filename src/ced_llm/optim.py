"""Optimizers for CED training: AdamW (default) + Muon (optional).

Muon (MomentUm Orthogonalized by Newton-Schulz) keeps a momentum buffer
like SGD, then orthogonalizes matrix gradients with a quintic
Newton-Schulz iteration before applying the update. Orthogonalized
updates have controlled scale, so Muon typically trains transformers in
fewer steps than AdamW -- at the cost of a few small matmuls per step.

Reference: Keller Jordan's ``muon`` (MODDED-NanoGPT era), reimplemented
here in torch-only code so the repo stays offline-first with zero new
dependencies.

Convention (matches common practice):
  * 2D parameters (matrices, incl. embeddings) -> Muon, lr ~0.02
  * everything else (biases, norms, 1D/0D)      -> AdamW, lr ~3e-4

Usage:
    python3 -m src.ced_llm.train --optimizer adamw              # default
    python3 -m src.ced_llm.train --optimizer muon               # muon 0.02 + adamw 3e-4
    python3 -m src.ced_llm.train --optimizer muon --muon-lr 0.01 --lr 1e-4
"""

import torch


def zeropower_via_newtonschulz5(G, steps=5):
    """Orthogonalize a 2D gradient via Newton-Schulz (quintic coefficients).

    Returns Q with Q @ Q.T ~= I, same shape/dtype/device as G. Runs in
    float32 for CPU safety. Never raises: degenerate input -> zeros.
    """
    assert isinstance(G, torch.Tensor) and G.ndim == 2, "Newton-Schulz needs a matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    try:
        X = G.detach().to(torch.float32)
        n = X.norm()
        if not torch.isfinite(n) or float(n) == 0.0:
            return torch.zeros_like(G)
        X = X / (n + 1e-7)
        for _ in range(max(1, int(steps))):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        if not torch.isfinite(X.norm()):
            return torch.zeros_like(G)
        return X.to(G.dtype)
    except Exception:
        return torch.zeros_like(G)


class Muon(torch.optim.Optimizer):
    """Muon for matrix parameters (momentum + Newton-Schulz orthogonalization).

    Args:
        params: parameters to optimize (ideally 2D; other shapes fall back
            to plain momentum updates, but prefer build_optimizer routing).
        lr: learning rate (Muon scale, e.g. 0.02 -- much larger than AdamW's).
        momentum: momentum coefficient for the buffer.
        nesterov: blend current grad with momentum (Nesterov-style).
        weight_decay: decoupled weight decay (0.0 default).
        ns_steps: Newton-Schulz iterations (5 default).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5):
        defaults = dict(lr=float(lr), momentum=float(momentum),
                        nesterov=bool(nesterov),
                        weight_decay=float(weight_decay),
                        ns_steps=int(ns_steps))
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns = group["ns_steps"]
            for p in group["params"]:
                g = p.grad
                if g is None or getattr(g, "is_sparse", False):
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.lerp_(g, 1.0 - mu)  # buf = mu*buf + (1-mu)*g
                update = g.lerp(buf, mu) if group["nesterov"] else buf
                if update.ndim == 2:
                    update = zeropower_via_newtonschulz5(update, steps=ns)
                if wd != 0.0:
                    p.mul_(1.0 - group["lr"] * wd)
                p.add_(update, alpha=-group["lr"])
        return loss


class ComboOptimizer:
    """Step several torch optimizers as one (unified zero_grad/step/param_groups).

    Lets train loops keep calling ``opt.zero_grad()`` / ``opt.step()`` and
    iterating ``opt.param_groups`` (e.g. for cosine decay) unchanged.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = [pg for o in self.optimizers for pg in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return [o.state_dict() for o in self.optimizers]

    def load_state_dict(self, states):
        for o, s in zip(self.optimizers, states):
            o.load_state_dict(s)


def build_optimizer(name, parameters, lr=3e-4, muon_lr=0.02, momentum=0.95,
                    weight_decay=0.0):
    """Factory: 'adamw' -> AdamW (all params); 'muon' -> Muon (2D) + AdamW (rest).

    The AdamW branch matches the historical default exactly
    (``torch.optim.AdamW(params, lr=lr)``). Unknown names raise ValueError.
    """
    key = str(name or "adamw").lower()
    params = [p for p in parameters if getattr(p, "requires_grad", True)]
    if key == "adamw":
        return torch.optim.AdamW(params, lr=float(lr))
    if key == "muon":
        mat = [p for p in params if getattr(p, "ndim", 0) == 2]
        rest = [p for p in params if getattr(p, "ndim", 0) != 2]
        opts = []
        if mat:
            opts.append(Muon(mat, lr=float(muon_lr), momentum=float(momentum),
                             weight_decay=float(weight_decay)))
        if rest:
            opts.append(torch.optim.AdamW(rest, lr=float(lr)))
        if len(opts) == 1:
            return opts[0]
        return ComboOptimizer(opts)
    raise ValueError("unknown optimizer %r (choices: adamw, muon)" % (name,))
