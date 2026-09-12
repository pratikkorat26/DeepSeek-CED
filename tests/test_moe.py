"""Tests for DeepSeekMoE-style layer (V4.1-Flash recipe at toy scale).

Covers shapes/determinism, top-k sparsity + renormalization, shared-expert
always-on semantics, gradient flow, noaux bias updates, and end-to-end
train --smoke with --moe. All offline, CPU, fast.
"""

import torch

try:
    from src.ced_llm.moe import (
        DeepSeekMoELayer,
        decode_active_params,
        prefill_active_params,
        sqrtsoftplus,
        total_params,
    )
except ImportError:
    from ced_llm.moe import (  # type: ignore
        DeepSeekMoELayer,
        decode_active_params,
        prefill_active_params,
        sqrtsoftplus,
        total_params,
    )


def _layer(**kw):
    args = dict(d_model=16, dim_ff=32, num_experts=4, top_k=2,
                shared_experts=1)
    args.update(kw)
    torch.manual_seed(0)
    return DeepSeekMoELayer(**args)


def test_sqrtsoftplus_nonnegative():
    x = torch.tensor([-10.0, -1.0, 0.0, 1.0, 10.0])
    s = sqrtsoftplus(x)
    assert bool((s >= 0).all()) and bool(torch.isfinite(s).all())
    assert float(s[0]) < float(s[-1])  # monotone


def test_moe_shapes_and_determinism():
    m = _layer().eval()
    x = torch.randn(2, 5, 16)
    a = m(x)
    assert a.shape == x.shape
    assert torch.equal(a, m(x))
    st = m.last_stats
    assert st["counts"].sum().item() == 2 * 5 * 2  # tokens x topk assignments


def test_topk_sparsity_and_renorm():
    m = _layer().eval()
    x = torch.randn(2, 5, 16)
    m(x)
    w = m.last_stats["topk_weight"]
    assert w.shape == (2, 5, 2)
    # norm_topk_prob + 1.5x routed scaling: rows sum to 1.5.
    assert torch.allclose(w.sum(-1), torch.full((2, 5), 1.5), atol=1e-5)


def test_shared_expert_always_on():
    m = _layer().eval()
    x = torch.randn(2, 4, 16)
    base = m(x)
    with torch.no_grad():
        for p in m.shared.parameters():
            p.add_(1.0)
    assert not torch.equal(base, m(x))
    # An expert the router never picks changes nothing.
    m2 = _layer().eval()
    base2 = m2(x)
    m2(x)
    used = set(m2.last_stats["topk_idx"].reshape(-1).tolist())
    unused = [e for e in range(4) if e not in used]
    if unused:
        with torch.no_grad():
            for p in m2.experts[unused[0]].parameters():
                p.add_(100.0)
        assert torch.equal(base2, m2(x))


def test_grad_flows_to_router_experts_shared():
    m = _layer()
    m.train()
    x = torch.randn(2, 4, 16)
    m(x).square().mean().backward()
    assert m.router.weight.grad is not None
    assert any(p.grad is not None for p in m.experts.parameters())
    assert any(p.grad is not None for p in m.shared.parameters())


def test_noaux_bias_updates_in_train_only():
    m = _layer()
    b0 = m.expert_bias.detach().clone()
    m.train()
    m(torch.randn(4, 8, 16))
    assert not torch.equal(b0, m.expert_bias)
    m.eval()
    b1 = m.expert_bias.detach().clone()
    m(torch.randn(4, 8, 16))
    assert torch.equal(b1, m.expert_bias)


def test_asymmetry_accounting_direction():
    try:
        from src.ced_llm.config import CEDConfig
        from src.ced_llm.model import CEDForLM
    except ImportError:
        from ced_llm.config import CEDConfig  # type: ignore
        from ced_llm.model import CEDForLM  # type: ignore
    torch.manual_seed(0)
    cfg = CEDConfig(vocab_size=64, d_model=16, n_enc_layers=1,
                    n_dec_layers=1, nhead=2, dim_ff=32, max_seq_len=16,
                    dropout=0.0, pad_token_id=0, moe_enabled=True,
                    moe_num_experts=4, moe_top_k=2, moe_shared_experts=1)
    cfg.validate()
    model = CEDForLM(cfg).eval()
    n_total = total_params(model)
    n_pre = prefill_active_params(model)
    n_dec = decode_active_params(model)
    assert 0 < n_pre < n_total
    assert 0 < n_dec < n_total
    assert n_dec != n_pre  # asymmetric activation, V4.1-Flash's signature shape
    # Sparse MoE actives far below a dense-equivalent would imply at scale.
    assert n_pre < n_total * 0.9 and n_dec < n_total * 0.9


def test_train_smoke_moe(tmp_path):
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    rc = train_main(["--smoke", "--moe", "--moe-experts", "4",
                     "--moe-topk", "2", "--run-dir", str(tmp_path),
                     "--run-name", "moe1"])
    assert rc == 0
    import json
    summary = json.load(open(str(tmp_path / "moe1" / "summary.json")))
    assert summary["summary"]["final_loss"] < summary["summary"]["init_loss"]
