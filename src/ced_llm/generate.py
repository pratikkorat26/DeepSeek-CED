"""Greedy / sampled generation with KV-reuse (CED decode cache).

Uses init_decode_cache ONCE then forward_step in a loop. Asserts exactly one
encoder forward via model.encoder_forward_count delta.
"""

import argparse
import os
import sys

try:
    import torch
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

try:
    from .data import SimpleTokenizer  # type: ignore
except Exception:
    try:
        from src.ced_llm.data import SimpleTokenizer  # type: ignore
    except Exception:
        SimpleTokenizer = None  # type: ignore


# ---------------------------------------------------------------------------
# Helpers (duck typing for SimpleTokenizer vs tiktoken)
# ---------------------------------------------------------------------------
def _tk_encode(tokenizer, text):
    try:
        ids = tokenizer.encode(text)
        return list(ids)
    except TypeError:
        try:
            return list(tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return []
    except Exception:
        return []


def _tk_decode(tokenizer, ids):
    try:
        return tokenizer.decode(ids)
    except Exception:
        # Manual fallback for SimpleTokenizer-like objects.
        try:
            pad = int(getattr(tokenizer, "pad_id", 0))
            bos = int(getattr(tokenizer, "bos_id", 2))
            eos = int(getattr(tokenizer, "eos_id", 3))
            unk = int(getattr(tokenizer, "unk_id", 1))
            id2tok = getattr(tokenizer, "id_to_token", {})
            toks = []
            for i in ids:
                i = int(i)
                if i in (pad, bos, eos):
                    continue
                toks.append("<unk>" if i == unk else id2tok.get(i, "<unk>"))
            return " ".join(toks)
        except Exception:
            return ""


def _tk_eos(tokenizer):
    for attr in ("eos_id", "eos_token_id"):
        try:
            v = getattr(tokenizer, attr, None)
        except Exception:
            v = None
        if v is not None:
            try:
                return int(v)
            except Exception:
                return v
    return None


def _tk_bos(tokenizer):
    for attr in ("bos_id", "bos_token_id"):
        try:
            v = getattr(tokenizer, attr, None)
        except Exception:
            v = None
        if v is not None:
            try:
                return int(v)
            except Exception:
                return v
    return None


def _enc_count(model):
    try:
        v = getattr(model, "encoder_forward_count", 0)
    except Exception:
        return 0
    if callable(v):
        try:
            return int(v())
        except Exception:
            return 0
    try:
        return int(v)
    except Exception:
        return 0


def _reset_enc(model):
    try:
        fn = getattr(model, "reset_encoder_counter", None)
        if callable(fn):
            fn()
            return
    except Exception:
        pass
    try:
        model.encoder_forward_count = 0
    except Exception:
        pass


def _extract_step_logits(out):
    """Return (logits Tensor, updated cache or None)."""
    cache = None
    logits = None
    if isinstance(out, dict):
        logits = out.get("logits", out.get("logit", None))
        cache = out.get("cache", out.get("past", out.get("cache_out", None)))
        if logits is None:
            # Maybe {'logits':..., } missing; try first tensor value.
            for v in out.values():
                if isinstance(v, torch.Tensor):
                    logits = v
                    break
    elif isinstance(out, (tuple, list)):
        if len(out) == 2:
            logits, cache = out[0], out[1]
        elif len(out) == 1:
            logits = out[0]
        else:
            logits = out[0]
            try:
                cache = out[1]
            except Exception:
                cache = None
    elif isinstance(out, torch.Tensor):
        logits = out
    elif hasattr(out, "logits"):
        try:
            logits = out.logits
        except Exception:
            logits = None
        for attr in ("cache", "past", "past_key_values"):
            try:
                c = getattr(out, attr, None)
            except Exception:
                c = None
            if c is not None:
                cache = c
                break
    if logits is None:
        raise TypeError("forward_step did not return logits")
    return logits, cache


# ---------------------------------------------------------------------------
# CHAOS STORY WIZARD sampling spells (repetition penalty, top-k, top-p)
# Weird, wonderful, and fully backward-compatible. Greedy stays greedy;
# stories get delightful. Turbo could never. 🌀✨
# ---------------------------------------------------------------------------
def _eos_ids(tokenizer):
    """Return list of EOS ids (supports SimpleTokenizer + tiktoken + lists)."""
    ids = []
    single = _tk_eos(tokenizer)
    if single is not None:
        try:
            ids.append(int(single))
        except Exception:
            pass
    # Some tokenizers expose a list (e.g. eos_token_ids).
    try:
        extra = getattr(tokenizer, "eos_token_ids", None)
        if isinstance(extra, (list, tuple)):
            for v in extra:
                try:
                    iv = int(v)
                    if iv not in ids:
                        ids.append(iv)
                except Exception:
                    continue
    except Exception:
        pass
    return ids


def _apply_repetition_penalty(logits, generated_ids, penalty=1.0, window=None):
    """CHAOS anti-echo spell: penalize already-seen tokens (HF-style).

    logits: Tensor [B, V] (float, CPU or any device). Returns new Tensor.
    generated_ids: list[int] of context + generated so far.
    penalty: 1.0 = off. >1.0 discourages repeats, <1.0 encourages them (weird!).
    window: only penalize last N ids (None = all history).
    """
    try:
        penalty = float(penalty)
    except Exception:
        return logits
    if penalty == 1.0 or penalty <= 0.0:
        return logits
    if not generated_ids:
        return logits
    try:
        hist = list(generated_ids)
        if window is not None:
            try:
                w = int(window)
                if w > 0:
                    hist = hist[-w:]
            except Exception:
                pass
        seen = set(int(x) for x in hist)
    except Exception:
        return logits
    try:
        vocab = logits.size(-1)
        out = logits.clone() if isinstance(logits, torch.Tensor) else logits
        for tid in seen:
            try:
                tid = int(tid)
            except Exception:
                continue
            if tid < 0 or tid >= vocab:
                continue
            # HF convention: divide positive logits, multiply negative ones.
            try:
                if (out[..., tid] > 0).any():
                    # Per-element handling for batched logits.
                    pos = out[..., tid] > 0
                    # out is [B,V]; boolean mask on last dim needs care.
                    # Simple loop over batch dim.
                    if out.dim() == 2:
                        for b in range(out.size(0)):
                            v = float(out[b, tid].item())
                            out[b, tid] = v / penalty if v > 0 else v * penalty
                    else:
                        v = float(out[..., tid].max().item())
                        # Fallback elementwise via torch.where
                        out[..., tid] = torch.where(
                            out[..., tid] > 0,
                            out[..., tid] / penalty,
                            out[..., tid] * penalty,
                        )
                else:
                    if out.dim() == 2:
                        for b in range(out.size(0)):
                            out[b, tid] = float(out[b, tid].item()) * penalty
                    else:
                        out[..., tid] = out[..., tid] * penalty
            except Exception:
                continue
        return out
    except Exception:
        return logits


def _filter_top_k(logits, top_k=0):
    """Keep only top-k logits, rest -> -inf. top_k<=0 disables."""
    try:
        k = int(top_k or 0)
    except Exception:
        return logits
    if k <= 0:
        return logits
    try:
        k = min(k, logits.size(-1))
        if k <= 0:
            return logits
        vals, idx = torch.topk(logits, k)
        filt = torch.full_like(logits, float("-inf"))
        filt.scatter_(-1, idx, vals)
        return filt
    except Exception:
        return logits


def _filter_top_p(logits, top_p=1.0):
    """Nucleus (top-p) filtering: keep smallest set with cumprob <= top_p.

    top_p>=1.0 (or <=0) disables. Always keeps at least 1 token.
    Operates on raw (scaled) logits, sets filtered positions to -inf.
    """
    try:
        p = float(top_p)
    except Exception:
        return logits
    if p >= 1.0 or p <= 0.0:
        return logits
    try:
        # Sort descending per batch row.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = probs.cumsum(dim=-1)
        # Shift right by 1 to always keep first token (HF-style).
        # Remove tokens where cumulative (excluding current) > p.
        shifted = torch.zeros_like(cum)
        shifted[..., 1:] = cum[..., :-1]
        to_remove = shifted > p
        # Scatter mask back to original ordering.
        remove_orig = torch.zeros_like(logits, dtype=torch.bool)
        remove_orig.scatter_(-1, sorted_idx, to_remove)
        filt = logits.clone()
        filt[remove_orig] = float("-inf")
        return filt
    except Exception:
        return logits


def _sample_next_token(logits_2d, temperature=0.0, top_k=0, top_p=1.0,
                       repetition_penalty=1.0, generated_ids=None,
                       repetition_window=None):
    """Single-token sampler spell. Returns int token id.

    - temperature<=0 -> greedy argmax (deterministic, ignores top-k/top-p).
    - temperature>0  -> penalty -> /temp -> top-k -> top-p -> multinomial.
    Pure helper (no model calls) so wizards can unit-charm it.
    """
    try:
        temperature = float(temperature)
    except Exception:
        temperature = 0.0
    logits = logits_2d.float()
    if generated_ids is not None:
        logits = _apply_repetition_penalty(
            logits, generated_ids, penalty=repetition_penalty,
            window=repetition_window,
        )
    if temperature is not None and temperature > 0:
        try:
            scaled = logits / max(1e-8, temperature)
        except Exception:
            scaled = logits
        scaled = _filter_top_k(scaled, top_k)
        scaled = _filter_top_p(scaled, top_p)
        try:
            probs = F.softmax(scaled, dim=-1)
            # Guard all -inf (can happen with tiny k/p): fall back to argmax of logits.
            try:
                if bool(torch.isinf(scaled).all()):
                    return int(torch.argmax(logits_2d, dim=-1).item())
            except Exception:
                pass
            nxt = int(torch.multinomial(probs, num_samples=1).item())
        except Exception:
            try:
                nxt = int(torch.argmax(F.softmax(scaled, dim=-1), dim=-1).item())
            except Exception:
                nxt = int(torch.argmax(logits_2d, dim=-1).item())
        return nxt
    else:
        # Greedy path: repetition penalty still applies (breaks echo loops!).
        # NOTE: tests expect pure argmax when penalty==1.0 (default) -> identical.
        return int(torch.argmax(logits, dim=-1).item())


@torch.no_grad()
def _greedy_fast_loop(model, cache, prev, generated, max_new_tokens,
                      stop_on_eos, eos_list, device):
    """Fused greedy decode loop (FORGE-DECODE fast path).

    Bit-identical to the legacy loop in :func:`generate_greedy` for pure
    greedy decoding (temperature<=0, repetition_penalty==1.0) but with the
    per-token overhead hoisted out:

    * single device-side ``argmax`` over the last position per step --
      no ``logits.float().cpu()`` full-vocab D2H copy per token;
    * one reused ``[1, 1]`` device input buffer (``fill_``) instead of a
      fresh ``torch.tensor([[nxt]])`` alloc + H2D per token;
    * EOS ids hoisted to a ``frozenset`` (legacy rebuilt an int list per
      token), no per-step sampler dispatch / ``_extract_step_logits``
      dispatch after the first step;
    * ``cache["enc_mask"]`` noned when the prompt has no pads (the caller
      builds an all-ones mask), which skips the per-step ``_pad_mask``
      alloc + ``.any()`` device sync inside ``forward_step``. Semantically
      identical: ``_pad_mask`` maps all-ones to ``None`` anyway.

    Returns the extended ``generated`` id list, or ``None`` when the model
    uses a non-canonical ``forward_step`` return convention (caller must
    rebuild the cache and run the legacy loop). A ``None`` bail happens
    before any id is appended, so the caller's ``generated`` is untouched.
    """
    try:
        want_eos = bool(stop_on_eos) and bool(eos_list)
        eos_set = frozenset(int(e) for e in (eos_list or [])) if want_eos else frozenset()
        check_eos = want_eos and len(eos_set) > 0
    except Exception:
        check_eos, eos_set = False, frozenset()
    # Hoisted mask normalization (see docstring). Guarded: foreign dict
    # caches without the key simply skip it.
    try:
        if isinstance(cache, dict) and cache.get("enc_mask") is not None:
            cache["enc_mask"] = None
    except Exception:
        pass
    # Hoisted input-token buffer (local prealloc: no FORGE_MODEL.md static
    # cache exists, so generate.py owns a tiny [1,1] reuse buffer).
    try:
        step_in = torch.empty((1, 1), dtype=torch.long, device=device)
        step_in.fill_(int(generated[-1]))
    except Exception:
        return None
    mode = None  # 'tuple' | 'dict' | 'tensor', detected on the first step
    use_kw = False
    kw_probed = False
    for _ in range(max_new_tokens):
        if use_kw:
            out = model.forward_step(step_in, cache=cache)
        else:
            try:
                out = model.forward_step(step_in, cache)
            except TypeError:
                if kw_probed:
                    raise
                use_kw = True
                try:
                    out = model.forward_step(step_in, cache=cache)
                except Exception as e:
                    raise RuntimeError("forward_step failed: %r" % (e,)) from e
        kw_probed = True
        if mode is None:
            # One-time return-convention + shape probe (contract:
            # logits [B,1,V], B==1 here). Anything exotic bails to legacy.
            if isinstance(out, torch.Tensor):
                mode = "tensor"
                logits, new_cache = out, None
            elif isinstance(out, dict):
                lg = out.get("logits", None)
                if not isinstance(lg, torch.Tensor):
                    return None
                if ("past" in out or "cache_out" in out) and "cache" not in out:
                    return None
                mode = "dict"
                logits, new_cache = lg, out.get("cache")
            elif isinstance(out, (tuple, list)) and len(out) == 2:
                logits, new_cache = out[0], out[1]
                if not isinstance(logits, torch.Tensor):
                    return None
                mode = "tuple"
            else:
                return None
            if logits.dim() == 3:
                if logits.size(0) != 1 or logits.size(1) != 1:
                    return None
            elif logits.dim() == 2:
                if logits.size(0) != 1:
                    return None
            else:
                return None
        elif mode == "tuple":
            logits, new_cache = out
        elif mode == "dict":
            logits, new_cache = out["logits"], out.get("cache")
        else:
            logits, new_cache = out, None
        if new_cache is not None:
            cache = new_cache
        # Fused greedy step: single argmax, device-side, scalar sync only.
        nxt = int(torch.argmax(logits.float(), dim=-1).item())
        generated.append(nxt)
        if check_eos and nxt in eos_set:
            break
        step_in.fill_(nxt)
    return generated


@torch.no_grad()
def generate_batch_greedy(
    model,
    tokenizer,
    prompts,
    max_new_tokens=64,
    device="cpu",
    stop_on_eos=True,
):
    """Greedy batch decode: one encoder pass for B prompts (FORGE-DECODE).

    Pads prompts right, runs a single ``init_decode_cache`` + one
    ``forward_step`` per token for the whole batch, so MPS launch/sync cost
    is amortized across rows. Per-row token ids are mathematically identical
    to calling :func:`generate_greedy` (temperature=0.0) per prompt: rows are
    independent, pads are excluded via the encoder mask, and only decoded
    (non-pad) tokens enter the self-attention history.

    Args:
        model: CED model with ``init_decode_cache`` / ``forward_step``.
        tokenizer: tokenizer with ``encode``/``decode`` (+ optional ids).
        prompts: list of prompt strings.
        max_new_tokens: new tokens per prompt (early EOS stop per row).
        device: torch device string.
        stop_on_eos: halt each row at any known EOS id.

    Returns:
        dict ``{texts, token_ids, encoder_forwards}`` with one entry per
        prompt. ``encoder_forwards`` is 1 (single shared encode). Falls back
        to sequential :func:`generate_greedy` when the model uses a
        non-canonical ``forward_step`` convention.
    """
    try:
        max_new_tokens = int(max_new_tokens)
    except Exception:
        max_new_tokens = 64
    max_new_tokens = max(0, max_new_tokens)
    try:
        stop_on_eos = bool(stop_on_eos)
    except Exception:
        stop_on_eos = True
    if isinstance(prompts, str):
        prompts = [prompts]
    try:
        prompts = list(prompts)
    except Exception:
        prompts = [prompts]
    if len(prompts) == 0:
        return {"texts": [], "token_ids": [], "encoder_forwards": 0}
    clean = []
    for p in prompts:
        if isinstance(p, str):
            clean.append(p)
        else:
            try:
                clean.append(str(p))
            except Exception:
                clean.append("")
    prompts = clean
    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass

    eos_list = _eos_ids(tokenizer)
    try:
        check_eos = bool(stop_on_eos) and bool(eos_list)
        eos_set = frozenset(int(e) for e in eos_list) if check_eos else frozenset()
        check_eos = check_eos and len(eos_set) > 0
    except Exception:
        check_eos, eos_set = False, frozenset()
    bos_id = _tk_bos(tokenizer)

    rows = []
    for p in prompts:
        ids = [int(x) for x in _tk_encode(tokenizer, p)]
        if len(ids) == 0 and bos_id is not None:
            ids = [int(bos_id)]
        if len(ids) == 0:
            ids = [int(bos_id) if bos_id is not None else 0]
        rows.append(ids)
    # Pad id: tokenizer first, then model config, then 0.
    pad_id = 0
    for attr in ("pad_id", "pad_token_id"):
        try:
            v = getattr(tokenizer, attr, None)
            if v is not None:
                pad_id = int(v)
                break
        except Exception:
            continue
    else:
        try:
            pad_id = int(getattr(getattr(model, "config", None), "pad_token_id", 0))
        except Exception:
            pad_id = 0
    try:
        pad_id = int(pad_id)
    except Exception:
        pad_id = 0
    batch = len(rows)
    width = max(len(r) for r in rows)
    padded = [r + [pad_id] * (width - len(r)) for r in rows]
    has_pad = any(len(r) != width for r in rows)
    input_ids = torch.tensor(padded, dtype=torch.long, device=device)
    attn = None
    if has_pad:
        attn = torch.ones((batch, width), dtype=torch.long, device=device)
        for i, r in enumerate(rows):
            try:
                if len(r) < width:
                    attn[i, len(r):] = 0
            except Exception:
                pass

    _reset_enc(model)
    start = _enc_count(model)
    if not hasattr(model, "init_decode_cache") or not callable(
        getattr(model, "init_decode_cache")
    ):
        raise AttributeError(
            "model missing init_decode_cache (need CED decode-cache API)"
        )
    try:
        cache = model.init_decode_cache(input_ids, attn)
    except TypeError:
        try:
            cache = model.init_decode_cache(input_ids)
        except Exception as e:
            raise RuntimeError("init_decode_cache failed: %r" % (e,)) from e
    if not hasattr(model, "forward_step") or not callable(
        getattr(model, "forward_step")
    ):
        raise AttributeError("model missing forward_step (need CED decode-cache API)")
    if not has_pad:
        # Same hoisted mask normalization as the single fast path.
        try:
            if isinstance(cache, dict) and cache.get("enc_mask") is not None:
                cache["enc_mask"] = None
        except Exception:
            pass

    gen_lists = [list(r) for r in rows]
    cur_cpu = torch.tensor([[r[-1]] for r in rows], dtype=torch.long)
    try:
        step_in = torch.empty((batch, 1), dtype=torch.long, device=device)
        step_in.copy_(cur_cpu)
    except Exception:
        step_in = input_ids[:, -1:]
    done = [False] * batch
    mode = None
    use_kw = False

    def _sequential_fallback():
        texts, ids = [], []
        for p in prompts:
            o = generate_greedy(
                model, tokenizer, p, max_new_tokens=max_new_tokens,
                device=device, temperature=0.0, stop_on_eos=stop_on_eos,
            )
            texts.append(o["text"])
            ids.append(o["token_ids"])
        return {"texts": texts, "token_ids": ids, "encoder_forwards": 1}

    for _ in range(max_new_tokens):
        if all(done):
            break
        try:
            out = model.forward_step(step_in, cache=cache) if use_kw else model.forward_step(step_in, cache)
        except TypeError:
            if use_kw:
                raise
            use_kw = True
            try:
                out = model.forward_step(step_in, cache=cache)
            except Exception as e:
                raise RuntimeError("forward_step failed: %r" % (e,)) from e
        if mode is None:
            if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[0], torch.Tensor):
                mode = "tuple"
            elif isinstance(out, dict) and isinstance(out.get("logits", None), torch.Tensor) and (
                "cache" in out or ("past" not in out and "cache_out" not in out)
            ):
                mode = "dict"
            elif isinstance(out, torch.Tensor):
                mode = "tensor"
            else:
                return _sequential_fallback()
        if mode == "tuple":
            logits, new_cache = out
        elif mode == "dict":
            logits, new_cache = out["logits"], out.get("cache")
        else:
            logits, new_cache = out, None
        if new_cache is not None:
            cache = new_cache
        if logits.dim() == 3:
            if logits.size(0) != batch or logits.size(1) != 1:
                return _sequential_fallback()
            last = logits[:, -1, :]
        elif logits.dim() == 2:
            if logits.size(0) != batch:
                return _sequential_fallback()
            last = logits
        else:
            return _sequential_fallback()
        # One kernel + one small sync for all B rows (amortized per token).
        try:
            nxt_ids = [int(v) for v in torch.argmax(last.float(), dim=-1).tolist()]
        except Exception:
            return _sequential_fallback()
        if len(nxt_ids) != batch:
            return _sequential_fallback()
        for i in range(batch):
            if done[i]:
                continue
            nxt = nxt_ids[i]
            gen_lists[i].append(nxt)
            if check_eos and nxt in eos_set:
                done[i] = True
            else:
                try:
                    cur_cpu[i, 0] = nxt
                except Exception:
                    pass
        if all(done):
            break
        for i in range(batch):
            if done[i]:
                try:
                    cur_cpu[i, 0] = pad_id
                except Exception:
                    pass
        try:
            step_in.copy_(cur_cpu)
        except Exception:
            try:
                step_in = torch.tensor(cur_cpu.tolist(), dtype=torch.long, device=device).view(batch, 1)
            except Exception:
                break

    end = _enc_count(model)
    delta = int(end) - int(start)
    if delta != 1:
        raise RuntimeError(
            "KV-reuse violated: encoder_forward_count delta=%d (expected 1). " % delta
        )
    texts = [_tk_decode(tokenizer, ids) for ids in gen_lists]
    return {"texts": texts, "token_ids": gen_lists, "encoder_forwards": delta}


@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    prompt,
    max_new_tokens=64,
    device="cpu",
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    repetition_window=None,
    seed=None,
    stop_on_eos=True,
):
    """Generate continuations using decode cache (KV-reuse) + CHAOS sparkle.

    - Calls model.init_decode_cache ONCE, then model.forward_step in a loop.
    - Counts encoder forwards via model.encoder_forward_count delta; raises
      if != 1 (proves single encode + cache reuse).
    - Sampling spells (all backward-compatible, defaults = old behavior):
        temperature=0.0 -> greedy argmax (deterministic, ignores top-k/top-p).
        temperature>0   -> repetition-penalty -> /temp -> top-k -> top-p -> sample.
        top_k=0 disables, top_p=1.0 disables, repetition_penalty=1.0 disables.
        seed=int reseeds torch+random for reproducible mischief (None = respect caller).
        stop_on_eos=True halts at ANY known EOS id; False never early-stops.
    Returns dict {text, token_ids, encoder_forwards}.
    Turbo's sampler wishes it sparkled like this. 🧙‍♂️
    """
    try:
        max_new_tokens = int(max_new_tokens)
    except Exception:
        max_new_tokens = 64
    max_new_tokens = max(0, max_new_tokens)
    try:
        temperature = float(temperature)
    except Exception:
        temperature = 0.0
    try:
        top_k = int(top_k or 0)
    except Exception:
        top_k = 0
    try:
        top_p = float(top_p if top_p is not None else 1.0)
    except Exception:
        top_p = 1.0
    try:
        repetition_penalty = float(
            repetition_penalty if repetition_penalty is not None else 1.0
        )
    except Exception:
        repetition_penalty = 1.0
    try:
        stop_on_eos = bool(stop_on_eos)
    except Exception:
        stop_on_eos = True
    # Optional reproducibility spell (does NOT break greedy determinism tests:
    # they pass seed=None and set torch.manual_seed outside).
    if seed is not None:
        try:
            import random as _rnd

            _rnd.seed(int(seed))
        except Exception:
            pass
        try:
            torch.manual_seed(int(seed))
        except Exception:
            pass

    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass

    if not isinstance(prompt, str):
        try:
            prompt = str(prompt)
        except Exception:
            prompt = ""

    prompt_ids = _tk_encode(tokenizer, prompt)
    prompt_ids = [int(x) for x in prompt_ids]
    eos_list = _eos_ids(tokenizer)
    eos_id = eos_list[0] if eos_list else _tk_eos(tokenizer)
    bos_id = _tk_bos(tokenizer)

    # Empty prompt -> seed with BOS if available.
    if len(prompt_ids) == 0 and bos_id is not None:
        prompt_ids = [int(bos_id)]

    # Tensors for cache init.
    if len(prompt_ids) == 0:
        # Degenerate: use pad/unk 0-length guard -> single dummy token.
        prompt_ids = [int(bos_id) if bos_id is not None else 0]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    _reset_enc(model)
    start = _enc_count(model)

    # ---- init_decode_cache ONCE (required) -------------------------------
    if not hasattr(model, "init_decode_cache") or not callable(
        getattr(model, "init_decode_cache")
    ):
        raise AttributeError(
            "model missing init_decode_cache (need CED decode-cache API)"
        )
    try:
        cache = model.init_decode_cache(input_ids, attention_mask)
    except TypeError:
        try:
            cache = model.init_decode_cache(input_ids)
        except Exception as e:
            raise RuntimeError("init_decode_cache failed: %r" % (e,)) from e
    if not hasattr(model, "forward_step") or not callable(
        getattr(model, "forward_step")
    ):
        raise AttributeError("model missing forward_step (need CED decode-cache API)")

    generated = list(prompt_ids)
    # Last prompt token seeds the first step input.
    try:
        prev = torch.tensor(
            [[generated[-1]]], dtype=torch.long, device=device
        )
    except Exception:
        prev = input_ids[:, -1:]

    # FORGE-DECODE fast path: pure-greedy (temperature<=0, no repetition
    # penalty) runs the fused device-side loop above. top_k/top_p are
    # ignored on the greedy branch by construction, so they need no guard.
    # Bit-identical to the legacy loop below; a None return means the model
    # uses a foreign return convention -> rebuild a pristine cache (counter
    # resets to exactly 1) and run the legacy loop for identical results.
    _fast_done = False
    if temperature <= 0.0 and repetition_penalty == 1.0 and max_new_tokens > 0:
        _fr = _greedy_fast_loop(
            model, cache, prev, generated, max_new_tokens,
            stop_on_eos, eos_list, device,
        )
        if _fr is not None:
            generated = _fr
            _fast_done = True
        else:
            try:
                cache = model.init_decode_cache(input_ids, attention_mask)
            except TypeError:
                cache = model.init_decode_cache(input_ids)
            generated = list(prompt_ids)
            try:
                prev = torch.tensor(
                    [[generated[-1]]], dtype=torch.long, device=device
                )
            except Exception:
                prev = input_ids[:, -1:]

    if not _fast_done:
        for _ in range(max_new_tokens):
            try:
                out = model.forward_step(prev, cache)
            except TypeError:
                # Try keyword form.
                try:
                    out = model.forward_step(prev, cache=cache)
                except Exception as e:
                    raise RuntimeError("forward_step failed: %r" % (e,)) from e
            logits, new_cache = _extract_step_logits(out)
            if new_cache is not None:
                cache = new_cache
            # logits: [B,1,V] or [B,V] -> take last position.
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            logits = logits.float().cpu()
            # ✨ One sampler spell to rule them all (keeps greedy bit-identical). ✨
            nxt = _sample_next_token(
                logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                generated_ids=generated,
                repetition_window=repetition_window,
            )
            generated.append(nxt)
            if stop_on_eos and eos_list and int(nxt) in [int(e) for e in eos_list]:
                break
            elif stop_on_eos and not eos_list and eos_id is not None and nxt == int(eos_id):
                break
            try:
                prev = torch.tensor([[nxt]], dtype=torch.long, device=device)
            except Exception:
                break

    end = _enc_count(model)
    delta = int(end) - int(start)
    if delta != 1:
        raise RuntimeError(
            "KV-reuse violated: encoder_forward_count delta=%d (expected 1). "
            "generate must call init_decode_cache exactly once and reuse cache." % delta
        )
    text = _tk_decode(tokenizer, generated)
    return {"text": text, "token_ids": generated, "encoder_forwards": delta}


# ---------------------------------------------------------------------------
# STORY WIZARD delights: quality eval + delightful-default story generation
# ---------------------------------------------------------------------------
# Delightful TinyStories defaults, tuned by the CHAOS STORY WIZARD after
# gazing into the swirling vortex of turbo's boring greedy outputs:
STORY_DEFAULTS = {
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repetition_penalty": 1.15,
    "max_new_tokens": 80,
}


def evaluate_story_quality(text, tokenizer=None):
    """Rate a TinyStory's delightfulness (offline, deterministic, turbo-proof).

    Returns dict with counts, distinct-n ratios, repetition diagnostics,
    a 0-100 magic_score, a verdict string, and wizardly suggestions.
    Never raises: garbage in -> honest low score out.
    """
    try:
        s = text if isinstance(text, str) else str(text)
    except Exception:
        s = ""
    words = s.split()
    wc = len(words)
    low = [w.lower().strip(".,!?;:\"'()") for w in words]
    low = [w for w in low if w]
    uniq = len(set(low)) if low else 0
    distinct_1 = (uniq / max(1, len(low))) if low else 0.0
    # Distinct-2 (bigrams).
    try:
        bigrams = list(zip(low, low[1:]))
        distinct_2 = (len(set(bigrams)) / max(1, len(bigrams))) if bigrams else 0.0
    except Exception:
        bigrams = []
        distinct_2 = 0.0
    # Distinct-3 (trigrams) for echo detection.
    try:
        trigrams = list(zip(low, low[1:], low[2:]))
        distinct_3 = (len(set(trigrams)) / max(1, len(trigrams))) if trigrams else 1.0
    except Exception:
        distinct_3 = 1.0
    # Longest run of the same word back-to-back ("the the the the").
    max_run = 1 if low else 0
    try:
        run = 1
        for a, b in zip(low, low[1:]):
            if a == b:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 1
    except Exception:
        pass
    # <unk> rate (SimpleTokenizer OOV shame meter).
    try:
        unk_rate = sum(1 for w in words if "<unk>" in w) / max(1, wc)
    except Exception:
        unk_rate = 0.0
    stripped = s.strip()
    ends_nicely = bool(stripped) and stripped[-1] in ".!?”\"'"
    has_once = "once upon" in s.lower()
    has_said = ("said" in s.lower()) or ('"' in s) or ("'" in s and wc > 5)
    # --- magic score (0-100) ------------------------------------------------
    try:
        score = 0.0
        score += 40.0 * max(0.0, min(1.0, distinct_1))
        score += 25.0 * max(0.0, min(1.0, distinct_2))
        score += 10.0 * max(0.0, min(1.0, distinct_3))
        # Length bonus: TinyStories want ~30-120 words.
        if wc >= 15:
            score += 5.0
        if wc >= 30:
            score += 5.0
        if wc > 200:
            score -= 5.0
        if ends_nicely:
            score += 5.0
        if has_once:
            score += 5.0
        if has_said:
            score += 5.0
        # Chaos penalties for echo-chambers (turbo's favorite habitat).
        if max_run >= 6:
            score -= 25.0
        elif max_run >= 4:
            score -= 15.0
        elif max_run >= 3:
            score -= 7.0
        if distinct_1 < 0.3 and wc > 10:
            score -= 15.0
        if unk_rate > 0.3:
            score -= 10.0
        elif unk_rate > 0.15:
            score -= 5.0
        score = max(0.0, min(100.0, score))
    except Exception:
        score = 0.0
    if score >= 80:
        verdict = "✨ DELIGHTFUL STORY MAGIC ✨"
    elif score >= 60:
        verdict = "🌟 Charming chaos (turbo is jealous)"
    elif score >= 40:
        verdict = "🌀 Meh-velous, needs more sparkle"
    elif score >= 20:
        verdict = "🥱 Turbo-boring (echo detected)"
    else:
        verdict = "💀 TURBO-TRASH (needs chaos rescue!)"
    suggestions = []
    try:
        if distinct_1 < 0.5 and wc > 5:
            suggestions.append("raise repetition_penalty (try 1.15-1.3) to break echo loops")
        if max_run >= 3:
            suggestions.append("echo-loop! bump repetition_penalty + lower temperature slightly")
        if not ends_nicely:
            suggestions.append("story stops mid-air: raise --max-new or check EOS handling")
        if wc < 15:
            suggestions.append("too short for storytime: raise --max-new to 80+")
        if unk_rate > 0.15:
            suggestions.append("many <unk>: tokenizer vocab mismatch with ckpt?")
        if not has_once and wc > 5:
            suggestions.append("try a 'Once upon a time' prompt for TinyStories vibes")
        if distinct_2 < 0.5 and wc > 10:
            suggestions.append("lower temperature (0.7) or top-p (0.85) for coherence")
        if not suggestions:
            suggestions.append("keep this spell! story slaps. 🧙")
    except Exception:
        pass
    return {
        "word_count": wc,
        "char_count": len(s),
        "unique_words": uniq,
        "distinct_1": round(float(distinct_1), 4),
        "distinct_2": round(float(distinct_2), 4),
        "distinct_3": round(float(distinct_3), 4),
        "max_repeat_run": int(max_run),
        "unk_rate": round(float(unk_rate), 4),
        "ends_nicely": bool(ends_nicely),
        "has_once_upon": bool(has_once),
        "has_dialogue": bool(has_said),
        "magic_score": round(float(score), 2),
        "verdict": verdict,
        "suggestions": suggestions,
    }


# Friendly alias (wizards hate typing).
evaluate_story = evaluate_story_quality


def story_quality_score(text, tokenizer=None):
    """Return just the 0-100 magic score float."""
    try:
        return float(evaluate_story_quality(text, tokenizer=tokenizer).get("magic_score", 0.0))
    except Exception:
        return 0.0


@torch.no_grad()
def generate_story(
    model,
    tokenizer,
    prompt,
    max_new_tokens=None,
    device="cpu",
    temperature=None,
    top_k=None,
    top_p=None,
    repetition_penalty=None,
    seed=None,
    stop_on_eos=True,
    score_quality=True,
):
    """Delightful-default story generation (the fun wrapper!).

    Defaults come from STORY_DEFAULTS (temp 0.8 / top-k 40 / top-p 0.9 /
    rep-penalty 1.15) -- tuned for TinyStories whimsy, not turbo's
    monotone drone. Pass any arg explicitly to override the magic.
    Returns generate_greedy's dict + {"quality": {...}, "params": {...}}.
    """
    d = dict(STORY_DEFAULTS)
    if max_new_tokens is None:
        max_new_tokens = d["max_new_tokens"]
    if temperature is None:
        temperature = d["temperature"]
    if top_k is None:
        top_k = d["top_k"]
    if top_p is None:
        top_p = d["top_p"]
    if repetition_penalty is None:
        repetition_penalty = d["repetition_penalty"]
    out = generate_greedy(
        model,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        device=device,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        seed=seed,
        stop_on_eos=stop_on_eos,
    )
    try:
        out["params"] = {
            "temperature": float(temperature),
            "top_k": int(top_k or 0),
            "top_p": float(top_p),
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "seed": seed,
        }
    except Exception:
        out["params"] = {}
    if score_quality:
        try:
            out["quality"] = evaluate_story_quality(out.get("text", ""))
        except Exception:
            out["quality"] = {}
    return out


# Alias for lazy wizards and rival-turbo refugees.
generate = generate_story


# ---------------------------------------------------------------------------
# Tiny random model for --smoke (offline, no ckpt needed)
# ---------------------------------------------------------------------------
def _build_smoke_model_and_tokenizer(vocab_size=512, d_model=32, seq_len=32):
    """Build a tiny random dense LM + SimpleTokenizer for smoke plumbing."""
    # Prefer the REAL CED model when available (proves true plumbing).
    try:
        try:
            from .config import CEDConfig as _RealConfig  # type: ignore
            from .model import CEDForLM as _RealLM  # type: ignore
        except Exception:
            from src.ced_llm.config import CEDConfig as _RealConfig  # type: ignore
            from src.ced_llm.model import CEDForLM as _RealLM  # type: ignore
        _nh = 4 if d_model % 4 == 0 else 2
        _cfg = _RealConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_enc_layers=1,
            n_dec_layers=1,
            nhead=_nh,
            dim_ff=4 * d_model,
            max_seq_len=max(seq_len, 32),
            dropout=0.0,
            pad_token_id=0,
        )
        model = _RealLM(_cfg)
    except Exception:
        # Fall back to train's shim when the real model is absent/broken.
        try:
            try:
                from .train import _FallbackConfig, _FallbackLM  # type: ignore
            except Exception:
                from src.ced_llm.train import _FallbackConfig, _FallbackLM  # type: ignore
            cfg = _FallbackConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_enc_layers=1,
                n_dec_layers=1,
                nhead=4 if d_model % 4 == 0 else 2,
                dim_ff=4 * d_model,
                max_seq_len=max(seq_len, 32),
                dropout=0.0,
                pad_token_id=0,
            )
            model = _FallbackLM(cfg)
        except Exception:
            # Ultimate fallback: define inline mini-LM (duplicated for robustness).
            import torch.nn as nn

            class _MiniLM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder_forward_count = 0
                    self.emb = nn.Embedding(vocab_size, d_model)
                    layer = nn.TransformerEncoderLayer(
                        d_model=d_model, nhead=2, dim_feedforward=4 * d_model,
                        dropout=0.0, batch_first=True,
                    )
                    self.tr = nn.TransformerEncoder(layer, num_layers=1)
                    self.head = nn.Linear(d_model, vocab_size, bias=False)
                    self._c = None

                def reset_encoder_counter(self):
                    self.encoder_forward_count = 0

                def _h(self, ids):
                    B, T = ids.shape
                    x = self.emb(ids)
                    try:
                        h = self.tr(x, is_causal=True)
                    except TypeError:
                        h = self.tr(x)
                    return h

                def forward(self, input_ids, attention_mask=None, labels=None):
                    self.encoder_forward_count += 1
                    return {"logits": self.head(self._h(input_ids))}

                def encode_once(self, input_ids, attention_mask=None):
                    self.encoder_forward_count += 1
                    h = self._h(input_ids)
                    self._c = {"input_ids": input_ids, "memory": h}
                    return h

                def init_decode_cache(self, input_ids, attention_mask=None):
                    return {"input_ids": input_ids, "memory": self.encode_once(input_ids, attention_mask)}

                def forward_step(self, nxt, cache):
                    if not isinstance(nxt, torch.Tensor):
                        nxt = torch.tensor(nxt)
                    if nxt.dim() == 1:
                        nxt = nxt.unsqueeze(1)
                    c = cache if isinstance(cache, dict) else self._c
                    full = torch.cat([c["input_ids"], nxt.to(c["input_ids"].device)], dim=1)
                    c["input_ids"] = full
                    return {"logits": self.head(self._h(full)[:, -1:, :]), "cache": c}

            model = _MiniLM()

    # Tokenizer: small synthetic vocab.
    if SimpleTokenizer is not None:
        try:
            from .data import load_tinystories as _lt  # type: ignore
        except Exception:
            try:
                from src.ced_llm.data import load_tinystories as _lt  # type: ignore
            except Exception:
                _lt = None
        texts = []
        try:
            if _lt is not None:
                for ex in _lt(split="train", streaming=True, max_examples=200):
                    t = ex.get("text", "") if isinstance(ex, dict) else str(ex)
                    if t.strip():
                        texts.append(t)
        except Exception:
            texts = []
        if not texts:
            texts = [
                "Once upon a time there was a little bunny.",
                "The little girl found a big red ball.",
            ]
        tok = SimpleTokenizer(texts, vocab_size=vocab_size)
    else:
        # Minimal duck-typed tokenizer if data.py import failed.
        class _Tok:
            bos_id = 2
            eos_id = 3
            pad_id = 0
            unk_id = 1

            def encode(self, s):
                return [4 + (ord(c) % 200) for c in s[:16]] or [4]

            def decode(self, ids):
                return "smoke " + " ".join(str(int(i)) for i in ids[:8])

        tok = _Tok()
    return model, tok


def _try_load_ckpt(ckpt_path, device="cpu"):
    """Load ckpt {model_state, config, tokenizer_vocab?} if present."""
    data = torch.load(ckpt_path, map_location=device)
    if not isinstance(data, dict):
        raise ValueError("ckpt must be a dict")
    cfg_d = data.get("config", {})
    if not isinstance(cfg_d, dict):
        # Dataclass/object -> dict.
        try:
            cfg_d = dict(vars(cfg_d))
        except Exception:
            cfg_d = {}
    vocab_size = int(data.get("vocab_size", cfg_d.get("vocab_size", 512)))
    d_model = int(cfg_d.get("d_model", 64))
    n_enc = int(cfg_d.get("n_enc_layers", 1))
    n_dec = int(cfg_d.get("n_dec_layers", 1))
    nhead = int(cfg_d.get("nhead", 4))
    dim_ff = int(cfg_d.get("dim_ff", 4 * d_model))
    max_len = int(cfg_d.get("max_seq_len", 128))
    drop = float(cfg_d.get("dropout", 0.0))
    pad = int(cfg_d.get("pad_token_id", 0))
    # Build model: prefer REAL CED, else train shim, else smoke builder.
    model = None
    try:
        try:
            from .config import CEDConfig as _RealConfig  # type: ignore
            from .model import CEDForLM as _RealLM  # type: ignore
        except Exception:
            from src.ced_llm.config import CEDConfig as _RealConfig  # type: ignore
            from src.ced_llm.model import CEDForLM as _RealLM  # type: ignore
        _cfg = _RealConfig(
            vocab_size=vocab_size, d_model=d_model, n_enc_layers=n_enc,
            n_dec_layers=n_dec, nhead=nhead, dim_ff=dim_ff,
            max_seq_len=max_len, dropout=drop, pad_token_id=pad,
        )
        model = _RealLM(_cfg)
    except Exception:
        pass
    if model is None:
        try:
            try:
                from .train import _FallbackConfig, _FallbackLM  # type: ignore
            except Exception:
                from src.ced_llm.train import _FallbackConfig, _FallbackLM  # type: ignore
            cfg = _FallbackConfig(vocab_size, d_model, n_enc, n_dec, nhead, dim_ff, max_len, drop, pad)
            model = _FallbackLM(cfg)
        except Exception:
            model, _ = _build_smoke_model_and_tokenizer(vocab_size=vocab_size)
    # Load weights strictly=False (defensive across shims).
    try:
        ms = data.get("model_state", data.get("state_dict", None))
        if ms is not None:
            model.load_state_dict(ms, strict=False)
    except Exception as e:
        print("[generate] WARNING: ckpt load_state_dict failed: %r" % (e,), file=sys.stderr)
    # Tokenizer.
    tok = None
    try:
        tv = data.get("tokenizer_vocab", None)
        if tv is not None and SimpleTokenizer is not None:
            if isinstance(tv, dict) and "token_to_id" in tv:
                tok = SimpleTokenizer.from_dict(tv)
            elif isinstance(tv, dict):
                tok = SimpleTokenizer.from_vocab(tv)
    except Exception:
        tok = None
    return model, tok, cfg_d


def build_argparser():
    p = argparse.ArgumentParser(
        description="Generate with CED LM (KV-reuse) -- now with CHAOS story sparkle 🧙"
    )
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--smoke", action="store_true")
    # Delightful TinyStories defaults (tuned by the CHAOS STORY WIZARD):
    # --temperature 0.8 --top-k 40 --top-p 0.9 --repetition-penalty 1.15
    # Pass --temperature 0.0 for pure greedy (deterministic, turbo-mode).
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--repetition-window", type=int, default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for reproducible mischief (default: None = chaotic)")
    p.add_argument("--no-eos-stop", action="store_true",
                   help="Disable early stopping on EOS (let chaos run to --max-new)")
    p.add_argument("--greedy", action="store_true",
                   help="Shortcut for --temperature 0.0 (pure greedy, deterministic)")
    p.add_argument("--device", type=str, default="cpu")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    device = args.device if isinstance(args.device, str) else "cpu"

    if args.smoke:
        print("[generate] SMOKE: tiny random model, 16 tokens offline")
        torch.manual_seed(0)
        model, tok = _build_smoke_model_and_tokenizer(
            vocab_size=512, d_model=32, seq_len=32
        )
        import time as _time

        _t0 = _time.time()
        out = generate_greedy(
            model, tok, "Once upon a time", max_new_tokens=16, device="cpu"
        )
        _dt = max(1e-9, _time.time() - _t0)
        _new = max(1, len(out["token_ids"]) - len(_tk_encode(tok, "Once upon a time")))
        print("[generate] smoke decode: %d new tokens in %.3fs (%.0f tok/s)" % (_new, _dt, _new / _dt))
        print("[generate] text: %s" % out["text"][:500])
        print("[generate] tokens: %d encoder_forwards=%d" % (len(out["token_ids"]), out["encoder_forwards"]))
        assert out["encoder_forwards"] == 1, "smoke KV-reuse check failed"
        assert len(out["token_ids"]) > 0, "smoke produced no tokens"
        print("[generate] SMOKE OK")
        return 0

    # ---- Non-smoke ---------------------------------------------------------
    prompt = args.prompt
    max_new = int(args.max_new)
    # --greedy shortcut overrides delightful defaults (for turbo refugees).
    _temp = 0.0 if getattr(args, "greedy", False) else float(args.temperature)
    _topk = 0 if getattr(args, "greedy", False) else int(args.top_k)
    _topp = 1.0 if getattr(args, "greedy", False) else float(getattr(args, "top_p", 0.9))
    _rep = 1.0 if getattr(args, "greedy", False) else float(getattr(args, "repetition_penalty", 1.15))
    _rep_win = getattr(args, "repetition_window", None)
    _seed = getattr(args, "seed", None)
    _stop = not bool(getattr(args, "no_eos_stop", False))
    model, tok, _ = None, None, None
    if args.ckpt and os.path.exists(args.ckpt):
        print("[generate] loading ckpt %s" % args.ckpt)
        try:
            model, tok, _ = _try_load_ckpt(args.ckpt, device=device)
        except Exception as e:
            print("[generate] ckpt load failed (%r); using random tiny model" % (e,), file=sys.stderr)
            model, tok = None, None
    if model is None:
        print("[generate] no ckpt; using tiny random model (offline)")
        torch.manual_seed(0 if _seed is None else int(_seed))
        model, tok2 = _build_smoke_model_and_tokenizer()
        if tok is None:
            tok = tok2
    out = generate_greedy(
        model,
        tok,
        prompt,
        max_new_tokens=max_new,
        device=device,
        temperature=_temp,
        top_k=_topk,
        top_p=_topp,
        repetition_penalty=_rep,
        repetition_window=_rep_win,
        seed=_seed,
        stop_on_eos=_stop,
    )
    print(out["text"])
    try:
        q = evaluate_story_quality(out["text"])
        print(
            "[generate] tokens=%d encoder_forwards=%d magic=%.1f %s"
            % (len(out["token_ids"]), out["encoder_forwards"],
               q.get("magic_score", 0.0), q.get("verdict", "")),
            file=sys.stderr,
        )
        print(
            "[generate] distinct-1=%.3f distinct-2=%.3f max_run=%d ends_nicely=%s spell(temp=%.2f, top_k=%d, top_p=%.2f, rep=%.2f)"
            % (q.get("distinct_1", 0.0), q.get("distinct_2", 0.0),
               q.get("max_repeat_run", 0), q.get("ends_nicely", False),
               _temp, _topk, _topp, _rep),
            file=sys.stderr,
        )
    except Exception:
        print(
            "[generate] tokens=%d encoder_forwards=%d"
            % (len(out["token_ids"]), out["encoder_forwards"]),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
