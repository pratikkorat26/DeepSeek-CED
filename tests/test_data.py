"""Tokenizer / batch / offline-loader tests (catches data-pipeline slop).

- Round-trip: decode(encode(s)) == normalized s on 3 fixed sentences.
  Catches lossy vocab/UNK handling or broken detokenization.
- Pack: packed length == seq_len with BOS/EOS present. Catches missing
  special tokens or wrong pad/truncate logic.
- Shapes/dtypes: get_toy_batch returns [B,T] long tensors. Catches
  transposed/flat/wrong-dtype batches that silently break training.
- Offline fallback: with `datasets` (and `tiktoken`) imports forced to fail,
  the TinyStories loader must still return synthetic texts. Catches hard
  network/dataset dependencies. Never imports tiktoken/datasets directly.
"""
import importlib
import inspect
import sys

import torch

SENTENCES = [
    "hello world",
    "the cat sat on the mat",
    "CED models reuse KV cache efficiently!",
]


def _data_mod():
    try:
        return importlib.import_module("src.ced_llm.data")
    except ImportError:
        return importlib.import_module("ced_llm.data")


def _norm(s):
    return " ".join(s.strip().lower().split())


def _make_tokenizer(texts):
    mod = _data_mod()
    assert hasattr(mod, "SimpleTokenizer"), (
        f"data module missing SimpleTokenizer; has {sorted(n for n in dir(mod) if not n.startswith('_'))}"
    )
    Tok = mod.SimpleTokenizer
    # Try common constructor signatures without hard-coding one.
    for kwargs in (dict(texts=texts), dict(corpus=texts), dict(data=texts), dict()):
        try:
            tok = Tok(**kwargs) if kwargs else Tok()
            # if vocab built from texts, prefer the one that saw our sentences
            return tok
        except TypeError:
            continue
    # last resort positional
    try:
        return Tok(texts)
    except Exception as e:
        raise AssertionError(f"could not construct SimpleTokenizer: {e}")


def _bos_eos_ids(tok):
    bos = getattr(tok, "bos_token_id", getattr(tok, "bos_id", getattr(tok, "BOS", None)))
    eos = getattr(tok, "eos_token_id", getattr(tok, "eos_id", getattr(tok, "EOS", None)))
    # common defaults if attributes are None but methods exist
    return bos, eos


def test_tokenizer_roundtrip():
    """decode(encode(s)) == normalized s for 3 fixed sentences."""
    torch.manual_seed(0)
    tok = _make_tokenizer(SENTENCES)
    assert hasattr(tok, "encode") and hasattr(tok, "decode"), "SimpleTokenizer needs encode/decode"
    for s in SENTENCES:
        ids = tok.encode(s)
        assert isinstance(ids, (list, tuple)) or torch.is_tensor(ids), f"encode must return ids, got {type(ids)}"
        if torch.is_tensor(ids):
            ids = ids.tolist()
        assert len(ids) > 0, f"encode({s!r}) returned empty ids"
        back = tok.decode(ids)
        assert isinstance(back, str), f"decode must return str, got {type(back)}"
        # accept exact or whitespace/case-normalized equality (tokenizer may lowercase)
        assert back == s or _norm(back) == _norm(s), (
            f"round-trip FAILED: {s!r} -> {ids} -> {back!r} "
            f"(normalized {_norm(back)!r} != {_norm(s)!r})"
        )


def test_pack_length_with_bos_eos():
    """Packed ids length == seq_len and contain BOS/EOS (via encode_pack)."""
    torch.manual_seed(0)
    mod = _data_mod()
    tok = _make_tokenizer(SENTENCES)
    seq_len = 16
    assert hasattr(mod, "encode_pack"), (
        f"data module missing encode_pack for pack test; has {sorted(n for n in dir(mod) if not n.startswith('_'))}"
    )
    packed = mod.encode_pack(SENTENCES, tok, seq_len)
    assert isinstance(packed, list) and len(packed) == len(SENTENCES), (
        f"encode_pack must return one row per input (got {type(packed)} len "
        f"{len(packed) if isinstance(packed, list) else '?'})"
    )
    bos, eos = _bos_eos_ids(tok)
    assert bos is not None and eos is not None, "tokenizer must expose bos/eos ids"
    for row in packed:
        assert isinstance(row, list), f"packed row must be list, got {type(row)}"
        assert len(row) == seq_len, f"pack length {len(row)} != seq_len {seq_len} (row {row})"
        assert bos in row, f"BOS id {bos} missing from packed {row}"
        assert eos in row, f"EOS id {eos} missing from packed {row}"
        assert row[0] == bos, f"packed row must start with BOS {bos}, got {row}"
    # truncation: very long input still packs to exactly seq_len
    long_text = " ".join(SENTENCES * 20)
    packed_long = mod.encode_pack([long_text], tok, seq_len)
    assert len(packed_long[0]) == seq_len, (
        f"truncated pack length {len(packed_long[0])} != {seq_len}"
    )


def _call_toy_batch(mod, batch_size=4, seq_len=16, vocab_size=128):
    fn = mod.get_toy_batch
    sig = inspect.signature(fn)
    kwargs = {}
    for name, val in (("batch_size", batch_size), ("batch", batch_size), ("B", batch_size),
                      ("seq_len", seq_len), ("seq_length", seq_len), ("T", seq_len),
                      ("vocab_size", vocab_size), ("vocab", vocab_size)):
        if name in sig.parameters:
            kwargs[name] = val
    # fill any other required args with None/default if possible
    try:
        return fn(**kwargs)
    except TypeError:
        # try positional (batch_size, seq_len)
        try:
            return fn(batch_size, seq_len)
        except Exception:
            return fn()


def test_get_toy_batch_shapes_dtypes():
    """get_toy_batch returns [B,T] long input_ids (+ mask/labels if provided)."""
    torch.manual_seed(0)
    mod = _data_mod()
    assert hasattr(mod, "get_toy_batch"), "data module missing get_toy_batch"
    B, T = 4, 16
    batch = _call_toy_batch(mod, batch_size=B, seq_len=T)
    if isinstance(batch, dict):
        assert "input_ids" in batch, f"toy batch dict missing input_ids; keys {sorted(batch.keys())}"
        ids = batch["input_ids"]
        assert isinstance(ids, torch.Tensor), f"input_ids must be Tensor, got {type(ids)}"
        assert ids.shape == (B, T), f"input_ids shape {tuple(ids.shape)} != {(B, T)}"
        assert ids.dtype == torch.long, f"input_ids dtype {ids.dtype} != torch.long"
        for key in ("attention_mask", "labels"):
            if key in batch and batch[key] is not None:
                t = batch[key]
                assert isinstance(t, torch.Tensor), f"{key} must be Tensor"
                assert t.shape == (B, T), f"{key} shape {tuple(t.shape)} != {(B, T)}"
    elif isinstance(batch, (tuple, list)):
        ids = batch[0]
        assert isinstance(ids, torch.Tensor), f"batch[0] must be Tensor, got {type(ids)}"
        assert ids.shape == (B, T), f"batch[0] shape {tuple(ids.shape)} != {(B, T)}"
        assert ids.dtype == torch.long, f"batch[0] dtype {ids.dtype} != torch.long"
    else:
        raise AssertionError(f"get_toy_batch returned unexpected type {type(batch)}")


def test_tinystories_loader_fallback_offline(monkeypatch):
    """With datasets/tiktoken imports broken, loader still returns synthetic texts."""
    torch.manual_seed(0)
    mod = _data_mod()
    assert hasattr(mod, "load_tinystories"), (
        f"data module missing load_tinystories; has {sorted(n for n in dir(mod) if not n.startswith('_'))}"
    )

    class _Failer:
        def __getattr__(self, _):
            raise ImportError("forced offline: datasets unavailable")
        def __call__(self, *a, **k):
            raise ImportError("forced offline: datasets unavailable")

    monkeypatch.setitem(sys.modules, "datasets", _Failer())
    monkeypatch.setitem(sys.modules, "tiktoken", _Failer())
    import builtins
    real_import = builtins.__import__

    def _guarded(name, *a, **k):
        if name == "datasets" or name.startswith("datasets."):
            raise ImportError("forced offline (guarded __import__)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _guarded)

    # 1) load_tinystories is a generator: consume it offline, expect synthetics.
    import itertools
    gen = mod.load_tinystories(split="train", max_examples=8)
    items = list(itertools.islice(gen, 8))
    assert len(items) > 0, "load_tinystories offline fallback yielded zero examples"
    texts = _coerce_texts(items)
    assert texts is not None and len(texts) > 0, (
        f"load_tinystories items not coercible to texts: {items[:2]}"
    )
    assert all(isinstance(s, str) and len(s.strip()) > 0 for s in texts), (
        f"offline fallback returned empty/non-string texts: {texts[:2]}"
    )

    # 2) get_dataloader must also work offline (uses synthetic fallback internally).
    assert hasattr(mod, "get_dataloader"), "data module missing get_dataloader"
    tok = _make_tokenizer(SENTENCES)
    loader = mod.get_dataloader(tok, split="train", seq_len=16, batch_size=2, max_examples=8)
    batch = next(iter(loader))
    assert isinstance(batch, dict) and "input_ids" in batch and "attention_mask" in batch, (
        f"get_dataloader batch keys {sorted(batch.keys()) if isinstance(batch, dict) else type(batch)}"
    )
    assert batch["input_ids"].shape == (2, 16), (
        f"dataloader input_ids shape {tuple(batch['input_ids'].shape)} != (2, 16)"
    )
    assert batch["input_ids"].dtype == torch.long, "dataloader input_ids must be long"


def _coerce_texts(out):
    import collections.abc
    if out is None:
        return None
    if isinstance(out, str):
        return [out]
    if isinstance(out, torch.utils.data.DataLoader):
        # pull one batch and decode if possible, else treat as opaque failure
        try:
            batch = next(iter(out))
            if isinstance(batch, dict) and "text" in batch:
                return list(batch["text"])
            return None
        except Exception:
            return None
    if isinstance(out, dict):
        for key in ("texts", "text", "data", "sentences"):
            if key in out:
                return _coerce_texts(out[key])
        # maybe {"train": [...]} style
        for v in out.values():
            r = _coerce_texts(v)
            if r is not None:
                return r
        return None
    if isinstance(out, (list, tuple)):
        if len(out) == 0:
            return None
        if all(isinstance(x, str) for x in out):
            return list(out)
        # list of dicts with text?
        if all(isinstance(x, dict) and "text" in x for x in out):
            return [x["text"] for x in out]
        return None
    return None
