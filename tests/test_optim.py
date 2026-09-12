"""Tests for Muon optimizer + hybrid factory (torch-only, CPU, offline).

Covers Newton-Schulz orthogonality, Muon descent on a toy problem,
build_optimizer routing (2D -> Muon, rest -> AdamW), and an end-to-end
train --smoke with --optimizer muon.
"""

import torch

try:
    from src.ced_llm.optim import (
        ComboOptimizer,
        Muon,
        build_optimizer,
        zeropower_via_newtonschulz5,
    )
except ImportError:
    from ced_llm.optim import (
        ComboOptimizer,
        Muon,
        build_optimizer,
        zeropower_via_newtonschulz5,
    )


def test_zeropower_orthogonalizes():
    # Faithful to Keller Jordan's quintic NS5: by design it does NOT fully
    # converge to UV^T, but to US'V^T with S' ~= Uniform(0.5, 1.5) -- which
    # trains just as well. So assert the singular-value band, not exactness.
    torch.manual_seed(0)
    G = torch.randn(16, 16)
    Q = zeropower_via_newtonschulz5(G, steps=5)
    assert Q.shape == G.shape
    sv = torch.linalg.svdvals(Q)
    assert float(sv.min()) > 0.5 and float(sv.max()) < 1.5, (float(sv.min()), float(sv.max()))


def test_zeropower_degenerate_input_safe():
    Z = torch.zeros(8, 8)
    Q = zeropower_via_newtonschulz5(Z)
    assert Q.shape == Z.shape and torch.isfinite(Q).all()


def test_muon_descends_toy_problem():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4)
    opt = Muon(model.parameters(), lr=0.02)
    x = torch.randn(32, 8)
    y = torch.randn(32, 4)
    with torch.no_grad():
        l0 = float(torch.nn.functional.mse_loss(model(x), y))
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        l1 = float(torch.nn.functional.mse_loss(model(x), y))
    assert l1 < l0, (l0, l1)


def test_build_optimizer_adamw_default():
    torch.manual_seed(0)
    model = torch.nn.Linear(8, 4)
    opt = build_optimizer("adamw", model.parameters(), lr=1e-3)
    assert isinstance(opt, torch.optim.AdamW)
    assert abs(opt.param_groups[0]["lr"] - 1e-3) < 1e-12


def test_build_optimizer_muon_routing():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8))
    opt = build_optimizer("muon", model.parameters(), lr=3e-4, muon_lr=0.02)
    assert isinstance(opt, ComboOptimizer)
    kinds = sorted(type(o).__name__ for o in opt.optimizers)
    assert kinds == ["AdamW", "Muon"], kinds
    n_muon = sum(p.numel() for o in opt.optimizers if isinstance(o, Muon)
                 for pg in o.param_groups for p in pg["params"])
    n_all = sum(p.numel() for p in model.parameters())
    assert 0 < n_muon < n_all  # matrices to Muon, norms/biases to AdamW
    # Unified API works.
    opt.zero_grad()
    loss = model(torch.randn(4, 8)).square().mean()
    loss.backward()
    opt.step()
    assert len(opt.param_groups) >= 2  # cosine loop iterates this


def test_build_optimizer_unknown_raises():
    try:
        build_optimizer("sgd", [torch.nn.Parameter(torch.zeros(2))])
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown optimizer")


def test_train_smoke_muon(tmp_path):
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    rc = train_main(["--smoke", "--optimizer", "muon",
                     "--run-dir", str(tmp_path), "--run-name", "muon1"])
    assert rc == 0
    import json
    summary = json.load(open(str(tmp_path / "muon1" / "summary.json")))
    assert summary["summary"]["final_loss"] < summary["summary"]["init_loss"]
    config = json.load(open(str(tmp_path / "muon1" / "config.json")))
    assert config["optimizer"] == "muon"
