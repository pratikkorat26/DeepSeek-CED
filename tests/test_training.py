"""Proves learning works end-to-end (catches broken loss/mask/grad flow).

a) Overfit: 8 toy sequences, 200 Adam steps lr 1e-3 => loss drops >=50% AND
   shifted-argmax accuracy >80%. Catches broken loss, wrong shift, bad mask,
   or no-grad bugs (none of those can memorize 8 short sequences).
b) Gradient flow: every requires_grad param gets a grad; encoder grad norm >0.
   Catches detached encoder (encode_once output not wired into decoder loss).
c) Padding invariance: padded-batch loss == unpadded loss (atol 1e-4) with a
   proper attention_mask + ignore_index. Catches ignored masks (pads leak
   into encoder/cross-attn or loss denominator).
"""
import torch

TINY = dict(vocab_size=64, d_model=32, n_enc_layers=1, n_dec_layers=1,
            nhead=2, dim_ff=64, max_seq_len=32, dropout=0.0, pad_token_id=0)


def _imports():
    try:
        from src.ced_llm.config import CEDConfig
    except ImportError:
        from ced_llm.config import CEDConfig
    try:
        from src.ced_llm.model import CEDForLM
    except ImportError:
        from ced_llm.model import CEDForLM
    return CEDConfig, CEDForLM


def _make_model(vocab_size=TINY["vocab_size"]):
    torch.manual_seed(0)
    CEDConfig, CEDForLM = _imports()
    cfg_d = dict(TINY)
    cfg_d["vocab_size"] = vocab_size
    cfg = CEDConfig(**cfg_d)
    if hasattr(cfg, "validate"):
        cfg.validate()
    return CEDForLM(cfg)


def _loss_from_out(out):
    if isinstance(out, dict):
        assert "loss" in out and out["loss"] is not None, (
            "forward(labels=...) returned no loss; model must compute causal LM loss when labels given"
        )
        return out["loss"]
    if isinstance(out, (tuple, list)):
        assert len(out) >= 2 and out[1] is not None, "forward returned no loss tuple-entry"
        return out[1]
    loss = getattr(out, "loss", None)
    assert loss is not None, "forward returned no loss attribute"
    return loss


def test_overfit_toy_batch():
    """8 sequences x T=16, 200 Adam steps: loss halves and acc >80%."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    B, T, V = 8, 16, TINY["vocab_size"]
    # avoid pad id 0 so memorization is about content, not pads
    full = torch.randint(1, V, (B, T))
    # caller-shifts convention per model doc: x_in=tok[:, :-1], y=tok[:, 1:];
    # logits align 1:1 with x_in, loss=CE(logits, y) with ignore_index=pad.
    x_in, y = full[:, :-1], full[:, 1:]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    with torch.no_grad():
        init_loss = _loss_from_out(model(x_in, labels=y)).item()
    assert init_loss > 0 and init_loss == init_loss, f"bad initial loss {init_loss}"
    for step in range(200):
        opt.zero_grad()
        loss = _loss_from_out(model(x_in, labels=y))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(x_in, labels=y)
        final_loss = _loss_from_out(out).item()
        logits = out["logits"] if isinstance(out, dict) else out[0]
    assert final_loss < 0.5 * init_loss, (
        f"no learning: init loss {init_loss:.4f} -> final {final_loss:.4f} "
        f"(need <50% of init after 200 Adam steps; check loss/shift/grad flow)"
    )
    # next-token accuracy aligned with the shifted loss: pred[t]==y[t]
    pred = logits.argmax(dim=-1)
    acc = (pred == y).float().mean().item()
    assert acc > 0.80, (
        f"overfit accuracy {acc * 100:.1f}% <= 80% (loss {init_loss:.4f}->{final_loss:.4f}); "
        f"model did not memorize 8 toy sequences"
    )


def test_gradient_flow_to_encoder():
    """All req-grad params get grads; encoder grad norm > 0."""
    torch.manual_seed(0)
    model = _make_model()
    model.train()
    model.zero_grad()
    B, T, V = 2, 8, TINY["vocab_size"]
    full = torch.randint(1, V, (B, T))
    x_in, y = full[:, :-1], full[:, 1:]
    loss = _loss_from_out(model(x_in, labels=y))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, (
        f"{len(missing)} params missing grads (e.g. {missing[:5]}); backward graph is broken/detached"
    )
    # locate encoder params: prefer explicit submodule, else name heuristic
    if hasattr(model, "encoder"):
        enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
        src = "model.encoder"
    else:
        enc_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and "enc" in n.lower()]
        src = "name~'enc'"
    assert len(enc_params) > 0, (
        f"no encoder params found via {src}; params are "
        f"{[n for n, _ in model.named_parameters()][:10]}..."
    )
    total = sum(p.grad.detach().float().norm().item() ** 2 for p in enc_params) ** 0.5
    assert total > 0.0, (
        "encoder grad norm == 0: encoder is detached from loss "
        "(encode_once output not used by decoder / cross-attn broken)"
    )


def test_padding_invariance():
    """Right-padded loss == unpadded loss given mask + ignore_index(pad_id).

    Uses 1:1-aligned labels (no caller shift) so both batches contain the same
    8 real tokens (padded adds only maskable pads). With correct masking the
    encoder/decoder never attend to pads and pad labels are ignored, so losses
    match. Catches ignored masks. NOTE: a shifted comparison would differ by
    design (padded encoder holds one extra real token), so unshifted is used.
    """
    torch.manual_seed(0)
    model = _make_model()
    model.eval()
    B, L, V = 2, 8, TINY["vocab_size"]
    pad_id = TINY["pad_token_id"]
    base_ids = torch.randint(1, V, (B, L))
    pad_len = L  # right half is all pads => total 2L, same 8 real tokens
    padded_ids = torch.cat([base_ids, torch.full((B, pad_len), pad_id, dtype=torch.long)], dim=1)
    mask_base = torch.ones(B, L, dtype=torch.long)
    mask_padded = torch.cat([torch.ones(B, L, dtype=torch.long),
                             torch.zeros(B, pad_len, dtype=torch.long)], dim=1)
    # model uses ignore_index=pad_token_id (NOT -100): padded labels keep pad_id.
    with torch.no_grad():
        loss_base = _loss_from_out(model(base_ids, attention_mask=mask_base, labels=base_ids)).item()
        loss_pad = _loss_from_out(
            model(padded_ids, attention_mask=mask_padded, labels=padded_ids)).item()
    diff = abs(loss_base - loss_pad)
    assert diff < 1e-4, (
        f"mask ignored: unpadded loss {loss_base:.6f} vs padded {loss_pad:.6f} "
        f"(diff {diff:.3e} >= 1e-4); pads leak into encoding/attention/loss"
    )
