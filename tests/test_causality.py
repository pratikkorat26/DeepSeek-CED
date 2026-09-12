"""Proves causal self-attention (catches non-causal / bidirectional masks).

NOTE on CED semantics: full-model logits prefix invariance does NOT hold by
design -- the decoder cross-attends GLOBALLY to every encoder position
(src/ced_llm/attention.py GlobalCrossAttention: "Every query position may
attend to every (non-pad) encoder position", wired in model.py forward).
Changing input_ids[:, -1] therefore changes k_glob/v_glob[:, -1] and shifts
prefix logits via cross-attention (measured ~2e-2 on the tiny config), even
when all self-attention is correctly causal. Asserting full-logits prefix
invariance would thus fail on a CORRECT implementation.

These tests instead verify the two causal components that MUST be invariant:
  1. Encoder causality via encode_once: sequences identical except last token
     have identical h_enc/k_glob/v_glob on [:, :-1] (catches non-causal
     encoder masks; sensitivity checked on last position).
  2. Decoder self-causality with FIXED globals: feeding the decoder stack two
     hidden sequences differing only at the last position (same k_glob/v_glob)
     leaves prefix outputs unchanged (catches non-causal decoder self-attn;
     sensitivity checked on last position). Includes an explicit
     future-token perturbation variant (changing last hidden state must NOT
     change outputs[:, :-1]).
Both FAIL if any self-attention uses bidirectional/full masks.
"""
import torch

TINY = dict(vocab_size=128, d_model=32, n_enc_layers=1, n_dec_layers=1,
            nhead=2, dim_ff=64, max_seq_len=32, dropout=0.0, pad_token_id=0)


def _make_model():
    torch.manual_seed(0)
    try:
        from src.ced_llm.config import CEDConfig
    except ImportError:
        from ced_llm.config import CEDConfig
    try:
        from src.ced_llm.model import CEDForLM
    except ImportError:
        from ced_llm.model import CEDForLM
    cfg = CEDConfig(**TINY)
    if hasattr(cfg, "validate"):
        cfg.validate()
    model = CEDForLM(cfg)
    model.eval()
    return model


def test_encoder_causal_prefix_invariance():
    """encode_once prefix states invariant to last-token change."""
    torch.manual_seed(0)
    model = _make_model()
    T = 8
    base = torch.randint(1, TINY["vocab_size"], (1, T))
    variant = base.clone()
    variant[0, -1] = (int(base[0, -1]) + 1) % TINY["vocab_size"]
    if int(variant[0, -1]) == 0:
        variant[0, -1] = 1
    assert int(variant[0, -1]) != int(base[0, -1]), "setup: last tokens must differ"
    with torch.no_grad():
        e_base = model.encode_once(base)
        e_var = model.encode_once(variant)
    for key in ("h_enc", "k_glob", "v_glob"):
        assert key in e_base and key in e_var, f"encode_once missing {key}"
        prefix_diff = (e_base[key][:, :-1, :] - e_var[key][:, :-1, :]).abs().max().item()
        assert prefix_diff < 1e-5, (
            f"NON-CAUSAL encoder: changing last token changed {key}[:, :-1] "
            f"(max diff {prefix_diff:.3e}, expected <1e-5; check encoder causal mask)"
        )
    last_diff = (e_base["h_enc"][:, -1, :] - e_var["h_enc"][:, -1, :]).abs().max().item()
    assert last_diff > 1e-6, (
        f"insensitive test: changing last token left h_enc[:, -1] unchanged "
        f"(diff {last_diff:.3e}); model may ignore input"
    )


def test_decoder_self_causal_with_fixed_globals():
    """Decoder prefix outputs invariant to future hidden change (globals fixed)."""
    torch.manual_seed(0)
    model = _make_model()
    B, T, D = 2, 8, TINY["d_model"]
    h_base = torch.randn(B, T, D)
    h_pert = h_base.clone()
    # NOTE: must NOT be a constant shift (LayerNorm is shift-invariant and would
    # wash it out); use a varying perturbation so post-LN states differ.
    torch.manual_seed(99)
    h_pert[:, -1, :] = torch.randn(B, D) * 3.0
    assert not torch.equal(h_base[:, -1], h_pert[:, -1]), "setup: last hidden must differ"
    torch.manual_seed(1)
    k_glob = torch.randn(B, T, D)
    v_glob = torch.randn(B, T, D)
    with torch.no_grad():
        out_base, _ = model.decoder(h_base, k_glob, v_glob)
        out_pert, _ = model.decoder(h_pert, k_glob, v_glob)
    assert out_base.shape == (B, T, D), f"unexpected decoder shape {tuple(out_base.shape)}"
    diff = (out_base[:, :-1, :] - out_pert[:, :-1, :]).abs().max().item()
    assert diff < 1e-5, (
        f"future-token leakage in decoder self-attn: perturbing last hidden changed "
        f"outputs[:, :-1] (max diff {diff:.3e}, expected <1e-5; check decoder causal mask)"
    )
    last_diff = (out_base[:, -1, :] - out_pert[:, -1, :]).abs().max().item()
    assert last_diff > 1e-4, (
        f"insensitive test: perturbing last hidden left last output unchanged "
        f"(diff {last_diff:.3e}); decoder may ignore input"
    )
