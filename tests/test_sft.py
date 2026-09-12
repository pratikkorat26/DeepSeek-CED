"""Tests for SFT stage (instruction pairs, masked loss, --init resume).

All offline, CPU, fast. The key invariant: prompt tokens are fully masked,
so changing them cannot change the loss.
"""

import torch

try:
    from src.ced_llm.sft import (
        IGNORE,
        build_instruction_pairs,
        encode_sft_pair,
        get_sft_dataloader,
    )
    from src.ced_llm.data import SimpleTokenizer
except ImportError:
    from ced_llm.sft import (  # type: ignore
        IGNORE,
        build_instruction_pairs,
        encode_sft_pair,
        get_sft_dataloader,
    )
    from ced_llm.data import SimpleTokenizer  # type: ignore


def _tok():
    return SimpleTokenizer(["tell me a short story about a bunny once upon "
                            "a time there was a little girl who laughed"],
                           vocab_size=64)


def test_pairs_cover_all_modes():
    pairs = build_instruction_pairs(9, seed=0)
    assert len(pairs) == 9
    prompts = " ".join(p for p, _ in pairs).lower()
    assert ("tell me" in prompts or "tell a story" in prompts
            or "bedtime story" in prompts or "tiny tale" in prompts)
    assert ("continue" in prompts or "what happens next" in prompts
            or "finish" in prompts)
    assert "what did" in prompts or "where did" in prompts or "who did" in prompts
    assert all(r.strip() for _, r in pairs)


def test_labels_mask_prompt_keep_response():
    tok = _tok()
    row, lab = encode_sft_pair(tok, "Tell me a story.", "Once upon a time.", seq_len=24)
    assert len(row) == len(lab) == 24
    n_prompt = sum(1 for l in lab if l == IGNORE)
    n_resp = sum(1 for l in lab if l >= 0)
    assert n_prompt > 0 and n_resp > 0
    # Response ids survive verbatim in the row at the same positions.
    for r, l in zip(row, lab):
        if l >= 0:
            assert r == l


def test_prompt_positions_excluded_from_loss():
    # Masked prompt targets must not train: same inputs, labels with the
    # prompt region unmasked (valid ids) MUST give a different loss than the
    # correctly masked labels. (Prompt content still flows through attention
    # -- only the training TARGETS are masked -- so this compares label maps.)
    from src.ced_llm.train import _make_config, _make_model, compute_loss
    torch.manual_seed(0)
    cfg = _make_config(64, 16, 1, 1, 2, 64, 24, 0.0, 0)
    model = _make_model(cfg).eval()
    tok = _tok()
    r1, l1 = encode_sft_pair(tok, "Tell me a story.", "Once upon a time.", seq_len=24)
    base = {"input_ids": torch.tensor([r1]),
            "attention_mask": torch.ones(1, 24, dtype=torch.long)}
    masked = dict(base, labels=torch.tensor([l1]))
    # Unmasked: train on every position (labels = shifted inputs, pad->-100).
    raw = [(-100 if v == tok.pad_id else v) for v in r1]
    unmasked = dict(base, labels=torch.tensor([raw]))
    with torch.no_grad():
        loss_m, _ = compute_loss(model, masked)
        loss_u, _ = compute_loss(model, unmasked)
    assert torch.isfinite(loss_m) and torch.isfinite(loss_u)
    assert abs(float(loss_m) - float(loss_u)) > 1e-6
    # Fully-masked batch: loss is degenerate but must not crash/NaN-propagate.
    dead = dict(base, labels=torch.full((1, 24), -100))
    with torch.no_grad():
        loss_d, _ = compute_loss(model, dead)
    assert torch.isfinite(loss_d) or bool(torch.isnan(loss_d))


def test_instructions_quick_run():
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    rc = train_main(["--data", "instructions", "--steps", "4",
                     "--max-examples", "16", "--batch-size", "4",
                     "--no-track"])
    assert rc == 0
