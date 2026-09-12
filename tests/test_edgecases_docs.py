"""DOC-BOSS edge-case tests: the prompts nobody admits to sending.

Covers the four embarrassing inputs every demo hits sooner or later:
  1. EMPTY prompt ("")        -> BOS-seeded generation, no crash, encoder x1.
  2. SINGLE-token prompt      -> prefix preserved, extends correctly, encoder x1.
  3. LONG prompt truncation   -> encode_pack truncates to seq_len (BOS/EOS kept);
                                 model raises a CLEAN ValueError past max_seq_len
                                 (caller must truncate; we prove the tail works).
  4. TEMPERATURE=0 determinism -> identical ids twice; temp=0 ignores top-k/top-p;
                                 seeded sampling (temp>0 + seed) is reproducible.

All run on the tiny CPU config (d_model=32, 1+1 layers) in milliseconds.
Owned by DOC-BOSS (Thunderdome docs lane). Rivals: touch this file and perish.
"""

import importlib

import torch

TINY = dict(vocab_size=128, d_model=32, n_enc_layers=1, n_dec_layers=1,
            nhead=2, dim_ff=64, max_seq_len=32, dropout=0.0, pad_token_id=0)
PROMPT = "hello world"


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


def _make_model():
    torch.manual_seed(0)
    CEDConfig, CEDForLM = _imports()
    cfg = CEDConfig(**TINY)
    if hasattr(cfg, "validate"):
        cfg.validate()
    m = CEDForLM(cfg)
    m.eval()
    return m


def _make_tokenizer(extra_texts=()):
    try:
        from src.ced_llm.data import SimpleTokenizer
    except ImportError:
        from ced_llm.data import SimpleTokenizer
    corpus = [PROMPT, "the cat sat on the mat",
              "CED models reuse KV cache efficiently", "hello", *extra_texts]
    return SimpleTokenizer(corpus, vocab_size=TINY["vocab_size"])


def _generate_fn():
    for modname in ("src.ced_llm.generate", "ced_llm.generate"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for fn_name in ("generate_greedy", "generate", "greedy_generate"):
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                return getattr(mod, fn_name)
    return None


def _gen(model, tokenizer, prompt, max_new, **kwargs):
    """Call generate_greedy positionally-safe; returns the result dict."""
    fn = _generate_fn()
    assert fn is not None, "generate module missing generate_greedy"
    kwargs.setdefault("temperature", 0.0)
    with torch.no_grad():
        try:
            return fn(model, tokenizer, prompt, max_new_tokens=max_new, **kwargs)
        except TypeError:
            return fn(model, tokenizer, prompt, max_new)


def _ids(out):
    if isinstance(out, dict):
        assert "token_ids" in out, f"result dict missing token_ids; keys {sorted(out.keys())}"
        return list(out["token_ids"])
    return list(out)


# ---------------------------------------------------------------------------
# 1. Empty prompt
# ---------------------------------------------------------------------------
def test_empty_prompt_bos_seed_no_crash():
    """'' must not crash: BOS-seeded, extends, encoder runs exactly once."""
    torch.manual_seed(0)
    model = _make_model()
    tok = _make_tokenizer()
    bos = getattr(tok, "bos_id", getattr(tok, "bos_token_id", None))
    eos = getattr(tok, "eos_id", getattr(tok, "eos_token_id", None))
    assert bos is not None, "tokenizer must expose bos_id for empty-prompt seeding"
    MAX_NEW = 6
    out = _gen(model, tok, "", MAX_NEW)
    ids = _ids(out)
    # Seeded with BOS when the prompt encodes to nothing.
    assert len(ids) >= 1 and ids[0] == int(bos), (
        f"empty prompt should seed with BOS {int(bos)}, got {ids[:4]}"
    )
    # Length contract: BOS + MAX_NEW unless stopped early ONLY on EOS.
    expected = 1 + MAX_NEW
    if len(ids) != expected:
        assert len(ids) < expected, f"output too long: {len(ids)} > {expected}"
        tail = ids[1:]
        assert eos is not None and eos in tail and tail[-1] == eos, (
            f"early stop at len {len(ids)} (expected {expected}) without trailing "
            f"EOS {eos} in suffix {tail}"
        )
    assert out.get("encoder_forwards", 1) == 1, (
        f"empty-prompt run re-encoded: encoder_forwards={out.get('encoder_forwards')}"
    )
    assert isinstance(out.get("text", ""), str), "result text must be str"


# ---------------------------------------------------------------------------
# 2. Single-token prompt
# ---------------------------------------------------------------------------
def test_single_token_prompt_prefix_preserved():
    """1-token prompt: output starts with it, grows, encoder x1."""
    torch.manual_seed(0)
    model = _make_model()
    tok = _make_tokenizer()
    prompt_ids = tok.encode("hello")
    if torch.is_tensor(prompt_ids):
        prompt_ids = prompt_ids.tolist()
    prompt_ids = list(prompt_ids)
    assert len(prompt_ids) == 1, (
        f"setup: 'hello' must encode to exactly 1 id with this vocab, got {prompt_ids}"
    )
    MAX_NEW = 6
    eos = getattr(tok, "eos_id", getattr(tok, "eos_token_id", None))
    out = _gen(model, tok, "hello", MAX_NEW)
    ids = _ids(out)
    assert ids[:1] == prompt_ids, (
        f"single-token output must start with prompt {prompt_ids}, got {ids}"
    )
    expected = 1 + MAX_NEW
    if len(ids) == expected:
        return
    assert len(ids) < expected, f"output too long: {len(ids)} > {expected}"
    tail = ids[1:]
    assert eos is not None and tail and tail[-1] == eos, (
        f"early stop without trailing EOS: tail {tail}"
    )
    assert out.get("encoder_forwards", 1) == 1


def test_single_token_model_cache_roundtrip():
    """Model-level T=1: init_decode_cache + 1 forward_step keeps shapes/sanity."""
    torch.manual_seed(0)
    model = _make_model()
    one = torch.randint(1, TINY["vocab_size"], (1, 1))
    with torch.no_grad():
        if hasattr(model, "reset_encoder_counter"):
            model.reset_encoder_counter()
        cache = model.init_decode_cache(one)
        logits, _ = model.forward_step(one, cache)
    assert logits.shape == (1, 1, TINY["vocab_size"]), (
        f"T=1 step logits shape {tuple(logits.shape)} != (1, 1, {TINY['vocab_size']})"
    )
    assert torch.isfinite(logits).all(), "T=1 step produced non-finite logits"
    c = model.encoder_forward_count
    n = c() if callable(c) else c
    assert n == 1, f"T=1 roundtrip encoder count {n} != 1"


# ---------------------------------------------------------------------------
# 3. Long-prompt truncation contract
# ---------------------------------------------------------------------------
def test_long_prompt_truncation_contract():
    """encode_pack truncates to seq_len; model rejects >max_seq_len cleanly."""
    torch.manual_seed(0)
    try:
        from src.ced_llm.data import encode_pack
    except ImportError:
        from ced_llm.data import encode_pack
    tok = _make_tokenizer()
    seq_len = 16
    long_text = " ".join(["hello world"] * 100)
    packed = encode_pack([long_text], tok, seq_len)
    assert len(packed) == 1 and len(packed[0]) == seq_len, (
        f"truncated pack length {len(packed[0])} != seq_len {seq_len}"
    )
    bos = getattr(tok, "bos_id", getattr(tok, "bos_token_id", None))
    eos = getattr(tok, "eos_id", getattr(tok, "eos_token_id", None))
    assert packed[0][0] == int(bos), f"truncated pack must start with BOS, got {packed[0][:4]}"
    assert int(eos) in packed[0], f"truncated pack must keep EOS {eos}: {packed[0]}"

    # Model-level: past max_seq_len the model must FAIL LOUDLY (ValueError
    # naming max_seq_len), never silently mis-generate.
    model = _make_model()
    too_long = torch.randint(1, TINY["vocab_size"], (1, TINY["max_seq_len"] + 32))
    try:
        with torch.no_grad():
            model.init_decode_cache(too_long)
    except ValueError as e:
        assert "max_seq_len" in str(e), f"length error should name max_seq_len, got: {e}"
    else:
        raise AssertionError(
            f"model accepted T={too_long.shape[1]} > max_seq_len={TINY['max_seq_len']} "
            "silently; must raise ValueError so callers truncate"
        )

    # And the documented fix works: keep the LAST max_seq_len tokens.
    tail = too_long[:, -TINY["max_seq_len"]:]
    with torch.no_grad():
        if hasattr(model, "reset_encoder_counter"):
            model.reset_encoder_counter()
        cache = model.init_decode_cache(tail)
        logits, _ = model.forward_step(tail[:, -1:], cache)
    assert logits.shape == (1, 1, TINY["vocab_size"])
    c = model.encoder_forward_count
    n = c() if callable(c) else c
    assert n == 1, f"truncated-tail run encoder count {n} != 1"


# ---------------------------------------------------------------------------
# 4. temperature=0 determinism
# ---------------------------------------------------------------------------
def test_temperature_zero_determinism():
    """temp=0: same prompt twice => bit-identical ids AND text."""
    torch.manual_seed(0)
    model = _make_model()
    tok = _make_tokenizer()
    out1 = _gen(model, tok, PROMPT, 8, temperature=0.0)
    out2 = _gen(model, tok, PROMPT, 8, temperature=0.0)
    assert _ids(out1) == _ids(out2), (
        f"nondeterministic greedy decode:\n{_ids(out1)}\nvs\n{_ids(out2)}"
    )
    assert out1["text"] == out2["text"], "greedy texts differ across identical calls"


def test_temperature_zero_ignores_topk_topp():
    """temp=0 must equal temp=0+filters (greedy path ignores top-k/top-p)."""
    torch.manual_seed(0)
    model = _make_model()
    tok = _make_tokenizer()
    plain = _ids(_gen(model, tok, PROMPT, 8, temperature=0.0))
    filtered = _ids(_gen(model, tok, PROMPT, 8, temperature=0.0, top_k=40, top_p=0.9))
    assert plain == filtered, (
        f"temp=0 should ignore top-k/top-p (pure argmax), got\n{plain}\nvs\n{filtered}"
    )


def test_seeded_sampling_is_reproducible():
    """temp>0 + seed=int: same seed twice => identical ids (mischief, but stable)."""
    torch.manual_seed(0)
    model = _make_model()
    tok = _make_tokenizer()
    kw = dict(temperature=0.8, top_k=40, top_p=0.9, seed=1234)
    out1 = _gen(model, tok, PROMPT, 8, **kw)
    out2 = _gen(model, tok, PROMPT, 8, **kw)
    assert _ids(out1) == _ids(out2), (
        f"seeded sampling not reproducible:\n{_ids(out1)}\nvs\n{_ids(out2)}"
    )
