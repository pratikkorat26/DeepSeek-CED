"""Greedy generation tests (catches nondeterminism / KV-reuse / length bugs).

- Determinism: same prompt+seed => identical ids. Catches RNG leakage
  (dropout left on, unseeded sampling) in the generate path.
- KV-reuse: encoder_forward_count == 1 after generation. Catches generate
  loops that re-encode the prompt every step instead of reusing globals.
- Length: output == prompt_len + max_new unless stopped early ONLY on EOS.
  Catches truncation/overrun and spurious early stopping.
Uses src.ced_llm.generate.generate_greedy when present (string-prompt API:
generate_greedy(model, tokenizer, prompt, max_new_tokens, device)), else an
equivalent manual cache loop (init_decode_cache + forward_step argmax) so the
CED contract is still exercised when the helper module is absent.
"""
import importlib

import torch

TINY = dict(vocab_size=128, d_model=32, n_enc_layers=1, n_dec_layers=1,
            nhead=2, dim_ff=64, max_seq_len=32, dropout=0.0, pad_token_id=0)
PROMPT = "hello world"
PROMPT2 = "the cat sat on the mat"


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


def _make_tokenizer():
    torch.manual_seed(0)
    try:
        from src.ced_llm.data import SimpleTokenizer
    except ImportError:
        from ced_llm.data import SimpleTokenizer
    # Build from the prompts we use so they are in-vocab; cap matches model vocab.
    return SimpleTokenizer([PROMPT, PROMPT2, "CED models reuse KV cache efficiently"],
                           vocab_size=TINY["vocab_size"])


def _try_generate_fn():
    for modname in ("src.ced_llm.generate", "ced_llm.generate"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        for fn_name in ("generate_greedy", "generate", "greedy_generate"):
            if hasattr(mod, fn_name) and callable(getattr(mod, fn_name)):
                return getattr(mod, fn_name)
    return None


def _manual_greedy_ids(model, prompt_ids, max_new, eos_id=None):
    """Manual CED loop on id tensors: init once, argmax forward_step. Returns list."""
    model.eval()
    if hasattr(model, "reset_encoder_counter"):
        model.reset_encoder_counter()
    with torch.no_grad():
        prompt = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        cache = model.init_decode_cache(prompt)
        # Mirror src generate.py: seed first step with last prompt token.
        cur = prompt[:, -1:]
        out = prompt[0].tolist()
        for _ in range(max_new):
            logits_step, cache = model.forward_step(cur, cache)
            nxt = int(logits_step[:, -1, :].argmax(dim=-1).item())
            out.append(nxt)
            if eos_id is not None and nxt == int(eos_id):
                break
            cur = torch.tensor([[nxt]], dtype=torch.long)
        return out


def _helper_greedy_ids(model, tokenizer, prompt_str, max_new):
    """Call real generate_greedy(model, tokenizer, prompt_str, ...) -> list ids."""
    fn = _try_generate_fn()
    assert fn is not None
    import inspect
    sig = inspect.signature(fn)
    names = set(sig.parameters)
    kwargs = {}
    # Actual src signature: (model, tokenizer, prompt, max_new_tokens, device, ...)
    if "tokenizer" in names or "tok" in names:
        key = "tokenizer" if "tokenizer" in names else "tok"
        kwargs[key] = tokenizer
    if "prompt" in names:
        kwargs["prompt"] = prompt_str
    for k in ("max_new_tokens", "max_new", "max_tokens", "max_length", "n_new"):
        if k in names:
            kwargs[k] = max_new
            break
    if "device" in names:
        kwargs["device"] = "cpu"
    if "temperature" in names:
        kwargs["temperature"] = 0.0
    # Build positional fallback: fn(model, tokenizer, prompt, max_new)
    with torch.no_grad():
        try:
            if kwargs:
                # need model + prompt positionally if not in kwargs
                args = []
                # inspect order: first params are model, tokenizer, prompt
                params = list(sig.parameters)
                for p in params:
                    if p in kwargs:
                        continue
                    if p.lower().startswith("model"):
                        args.append(model)
                    elif "prompt" in p.lower() or "text" in p.lower():
                        args.append(prompt_str)
                # Simpler: try keyword-only call with model included
                call_kwargs = dict(kwargs)
                # ensure model present
                mkey = next((p for p in params if "model" in p.lower()), None)
                if mkey and mkey not in call_kwargs:
                    call_kwargs[mkey] = model
                # ensure prompt present
                pkey = next((p for p in params if "prompt" in p.lower()), None)
                if pkey and pkey not in call_kwargs:
                    call_kwargs[pkey] = prompt_str
                # ensure tokenizer present
                tkey = next((p for p in params if "token" in p.lower()), None)
                if tkey and tkey not in call_kwargs:
                    call_kwargs[tkey] = tokenizer
                out = fn(**call_kwargs)
            else:
                out = fn(model, tokenizer, prompt_str, max_new)
        except TypeError:
            out = fn(model, tokenizer, prompt_str, max_new_tokens=max_new)
    if isinstance(out, dict):
        assert "token_ids" in out, f"generate_greedy dict missing token_ids; keys {sorted(out.keys())}"
        return list(out["token_ids"]), out
    if isinstance(out, (list, tuple)) and out and isinstance(out[0], int):
        return list(out), {"token_ids": list(out)}
    raise AssertionError(f"generate_greedy returned unexpected type {type(out)}: {out!r}")


def _counter(model):
    c = model.encoder_forward_count
    return c() if callable(c) else c


def test_greedy_determinism():
    """Same prompt + seed => identical generated ids (helper or manual loop)."""
    torch.manual_seed(0)
    fn = _try_generate_fn()
    if fn is not None:
        model = _make_model()
        tok = _make_tokenizer()
        torch.manual_seed(0)
        model.eval()
        ids1, _ = _helper_greedy_ids(model, tok, PROMPT, 8)
        torch.manual_seed(0)
        ids2, _ = _helper_greedy_ids(model, tok, PROMPT, 8)
        assert ids1 == ids2, (
            f"nondeterministic greedy decode via {fn.__name__}: same prompt+seed gave\n{ids1}\nvs\n{ids2}"
        )
        assert len(ids1) > len(tok.encode(PROMPT)), "generate must extend the prompt"
    else:
        # Fallback contract loop on raw ids when helper module is absent.
        model = _make_model()
        prompt = torch.randint(1, TINY["vocab_size"], (1, 6))
        torch.manual_seed(0)
        out1 = _manual_greedy_ids(model, prompt[0], 8)
        torch.manual_seed(0)
        model2 = _make_model()
        out2 = _manual_greedy_ids(model2, prompt[0], 8)
        assert out1 == out2, f"nondeterministic manual decode:\n{out1}\nvs\n{out2}"


def test_kv_reuse_counter_is_one():
    """Encoder runs exactly once during generation (helper or manual loop)."""
    torch.manual_seed(0)
    fn = _try_generate_fn()
    if fn is not None:
        model = _make_model()
        tok = _make_tokenizer()
        if hasattr(model, "reset_encoder_counter"):
            model.reset_encoder_counter()
        else:
            raise AssertionError("model missing reset_encoder_counter per contract")
        _, out = _helper_greedy_ids(model, tok, PROMPT, 6)
        n = _counter(model)
        assert n == 1, (
            f"generate re-encoded prompt: encoder_forward_count=={n} after helper generate, "
            f"expected 1 (helper returned encoder_forwards={out.get('encoder_forwards')})"
        )
        assert out.get("encoder_forwards", 1) == 1, (
            f"helper self-reports encoder_forwards={out.get('encoder_forwards')} != 1"
        )
    else:
        model = _make_model()
        prompt = torch.randint(1, TINY["vocab_size"], (1, 6))
        if hasattr(model, "reset_encoder_counter"):
            model.reset_encoder_counter()
        else:
            raise AssertionError("model missing reset_encoder_counter per contract")
        _manual_greedy_ids(model, prompt[0], 6)
        n = _counter(model)
        assert n == 1, (
            f"manual loop re-encoded: encoder_forward_count=={n}, expected 1"
        )


def test_output_length_or_eos_stop():
    """Output len == prompt + max_new unless stopped early ONLY on EOS."""
    torch.manual_seed(0)
    fn = _try_generate_fn()
    MAX_NEW = 8
    if fn is not None:
        model = _make_model()
        tok = _make_tokenizer()
        eos_id = getattr(tok, "eos_id", getattr(tok, "eos_token_id", None))
        prompt_ids = tok.encode(PROMPT)
        assert len(prompt_ids) > 0, "prompt encodes to empty ids"
        ids, _ = _helper_greedy_ids(model, tok, PROMPT, MAX_NEW)
        assert ids[:len(prompt_ids)] == list(prompt_ids), (
            f"generate output must start with prompt ids {prompt_ids}, got {ids}"
        )
        expected = len(prompt_ids) + MAX_NEW
        if len(ids) == expected:
            return
        assert len(ids) < expected, f"output too long: {len(ids)} > {expected}"
        tail = ids[len(prompt_ids):]
        assert eos_id is not None and eos_id in tail, (
            f"early stop at len {len(ids)} (expected {expected}) without EOS {eos_id} in suffix {tail}"
        )
        assert tail[-1] == eos_id, f"early-stopped output should end with EOS, got tail {tail}"
    else:
        model = _make_model()
        P = 6
        prompt = torch.randint(1, TINY["vocab_size"], (1, P))
        eos_id = TINY["vocab_size"] - 1
        ids = _manual_greedy_ids(model, prompt[0], MAX_NEW, eos_id=eos_id)
        assert ids[:P] == prompt[0].tolist(), "manual output must start with prompt"
        expected = P + MAX_NEW
        if len(ids) == expected:
            return
        assert len(ids) < expected, f"output too long: {len(ids)} > {expected}"
        tail = ids[P:]
        assert eos_id in tail and tail[-1] == eos_id, f"early stop without trailing EOS: {tail}"
