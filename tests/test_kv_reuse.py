"""Proves CED KV-reuse is real and correct (catches cache bugs / recompute-cheating).

a) Parity: full forward logits == incremental decode loop within 1e-4.
   Catches self-KV cache bugs (wrong positions, missing causal shift, etc.).
b) Single-encode: encoder_forward_count == 1 after K incremental steps and
   k_glob/v_glob tensor identity (data_ptr) unchanged. Catches
   recompute-cheating where the encoder is re-run per step.
c) Global-context dependence: zeroing k_glob/v_glob CHANGES decoder logits.
   Catches disconnected cross-attention where the decoder ignores globals
   (parity + single-encode would then pass vacuously).
"""
import copy

import torch

TINY = dict(vocab_size=128, d_model=32, n_enc_layers=1, n_dec_layers=1,
            nhead=2, dim_ff=64, max_seq_len=32, dropout=0.0, pad_token_id=0)
TOL = 1e-4


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


def _full_logits(model, input_ids, attention_mask=None):
    with torch.no_grad():
        kwargs = {} if attention_mask is None else {"attention_mask": attention_mask}
        out = model(input_ids, **kwargs)
        if isinstance(out, dict):
            return out["logits"]
        if isinstance(out, (tuple, list)):
            return out[0]
        return getattr(out, "logits")


def _incremental_logits(model, input_ids, attention_mask=None):
    """init_decode_cache on full input, then T forward_step calls feeding col t."""
    with torch.no_grad():
        if attention_mask is None:
            cache = model.init_decode_cache(input_ids)
        else:
            cache = model.init_decode_cache(input_ids, attention_mask)
        B, T = input_ids.shape
        steps = []
        for t in range(T):
            nxt = input_ids[:, t:t + 1]
            logits_step, cache = model.forward_step(nxt, cache)
            # forward_step returns [B,1,V]
            assert logits_step.shape == (B, 1, TINY["vocab_size"]), (
                f"forward_step logits shape {tuple(logits_step.shape)} != {(B, 1, TINY['vocab_size'])}"
            )
            steps.append(logits_step)
        return torch.cat(steps, dim=1), cache


def _counter(model):
    c = model.encoder_forward_count
    return c() if callable(c) else c


def _glob_keys(cache):
    assert isinstance(cache, dict), f"cache must be dict, got {type(cache)}"
    assert "k_glob" in cache and "v_glob" in cache, (
        f"cache missing k_glob/v_glob per contract; got keys {sorted(cache.keys())}"
    )
    return "k_glob", "v_glob"


def test_parity_full_vs_incremental():
    """Full forward must EXACTLY equal incremental loop (atol 1e-4)."""
    torch.manual_seed(0)
    model = _make_model()
    B, T = 2, 8
    input_ids = torch.randint(1, TINY["vocab_size"], (B, T))
    full = _full_logits(model, input_ids)
    inc, _ = _incremental_logits(model, input_ids)
    assert full.shape == inc.shape, f"shape mismatch full {tuple(full.shape)} vs inc {tuple(inc.shape)}"
    maxdiff = (full - inc).abs().max().item()
    assert torch.allclose(full, inc, atol=TOL, rtol=1e-5), (
        f"cache parity FAILED: max abs diff {maxdiff:.3e} exceeds {TOL}; "
        f"forward_step KV handling likely buggy (positions/mask/cache update)"
    )


def test_single_encode_and_ptr_stability():
    """Encoder runs once for K steps; global KV tensors keep identity."""
    torch.manual_seed(0)
    model = _make_model()
    if hasattr(model, "reset_encoder_counter"):
        model.reset_encoder_counter()
    B, T = 1, 8
    input_ids = torch.randint(1, TINY["vocab_size"], (B, T))
    with torch.no_grad():
        cache = model.init_decode_cache(input_ids)
        kk, vv = _glob_keys(cache)
        ptr_k0 = cache[kk].data_ptr()
        ptr_v0 = cache[vv].data_ptr()
        K = 4
        # generate K new tokens incrementally (feed last col repeatedly / argmax)
        nxt = input_ids[:, -1:]
        for _ in range(K):
            logits_step, cache = model.forward_step(nxt, cache)
            kk2, vv2 = _glob_keys(cache)
            assert cache[kk2].data_ptr() == ptr_k0, (
                "k_glob tensor reallocated across steps (data_ptr changed); "
                "encoder globals must be reused, not recomputed/reallocated"
            )
            assert cache[vv2].data_ptr() == ptr_v0, (
                "v_glob tensor reallocated across steps (data_ptr changed); "
                "encoder globals must be reused, not recomputed/reallocated"
            )
            nxt = logits_step[:, -1:, :].argmax(dim=-1)
    n = _counter(model)
    assert n == 1, (
        f"recompute-cheating: encoder_forward_count=={n} after {K} incremental steps, "
        f"expected exactly 1 (encoder must run once in init_decode_cache only)"
    )


def test_global_context_dependence():
    """Zeroing k_glob/v_glob must CHANGE decoder logits (decoder uses globals)."""
    torch.manual_seed(0)
    model = _make_model()
    B, T = 1, 8
    input_ids = torch.randint(1, TINY["vocab_size"], (B, T))
    nxt = torch.randint(1, TINY["vocab_size"], (B, 1))
    with torch.no_grad():
        cache_clean = model.init_decode_cache(input_ids)
        # fresh second cache so self-KV state is identical (both empty) before the step
        cache_zero = model.init_decode_cache(input_ids)
        kk, vv = _glob_keys(cache_clean)
        logits_clean, _ = model.forward_step(nxt, cache_clean)
        cache_zero[kk].zero_()
        cache_zero[vv].zero_()
        logits_zero, _ = model.forward_step(nxt, cache_zero)
    diff = (logits_clean - logits_zero).abs().max().item()
    assert diff > 1e-4, (
        f"disconnected cross-attn: zeroing k_glob/v_glob left logits unchanged "
        f"(max diff {diff:.3e} <= 1e-4); decoder must actually consume global KV"
    )
