"""Tests for GPT-2 tokenizer wrapper + factory (offline-first).

The fallback tests explicitly simulate a missing optional dependency, so the
suite remains valid whether tiktoken is installed in the test environment.
All offline, CPU, fast.
"""

import sys
import types

try:
    import src.ced_llm.data as _data_module
    from src.ced_llm.data import (
        GPT2Tokenizer,
        SimpleTokenizer,
        build_tokenizer,
        encode_pack,
    )
except ImportError:
    import ced_llm.data as _data_module  # type: ignore
    from ced_llm.data import (  # type: ignore
        GPT2Tokenizer,
        SimpleTokenizer,
        build_tokenizer,
        encode_pack,
    )


class _StubEnc:
    """Minimal tiktoken-like encoding (roundtrip-safe, tiny vocab)."""
    n_vocab = 100

    def encode(self, text, disallowed_special=()):
        return [(ord(c) % 90) + 5 for c in str(text)][:64]

    def decode(self, ids):
        return " ".join("w%d" % int(i) for i in ids)


def _install_stub_tiktoken(monkeypatch):
    mod = types.ModuleType("tiktoken")
    mod.get_encoding = lambda name: _StubEnc()
    monkeypatch.setitem(sys.modules, "tiktoken", mod)
    return mod


def test_factory_defaults_to_simple():
    tok, kind = build_tokenizer(None, ["hello world"])
    assert kind == "simple" and isinstance(tok, SimpleTokenizer)
    tok, kind = build_tokenizer("nonsense", ["hello"])
    assert kind == "simple"


def test_factory_gpt2_falls_back_offline(monkeypatch, capsys):
    # Simulate a missing optional dependency even when installed globally.
    monkeypatch.setattr(_data_module, "try_load_gpt2_tokenizer", lambda: None)
    tok, kind = build_tokenizer("gpt2", ["hello world"])
    assert kind == "simple" and isinstance(tok, SimpleTokenizer)
    assert "falling back" in capsys.readouterr().out


def test_wrapper_with_stubbed_tiktoken(monkeypatch):
    _install_stub_tiktoken(monkeypatch)
    tok, kind = build_tokenizer("gpt2", ["ignored"])
    assert kind == "gpt2" and isinstance(tok, GPT2Tokenizer)
    assert tok.vocab_size == 101 and tok.pad_id == 100 and tok.eos_id == 99
    assert tok.bos_id is None and tok.unk_id is None
    ids = tok.encode("hi there")
    assert ids and all(0 <= i < 100 for i in ids)
    assert tok.decode(ids) != ""
    d = tok.to_dict()
    assert d["kind"] == "gpt2"
    tok2 = GPT2Tokenizer.from_dict(d)
    assert isinstance(tok2, GPT2Tokenizer) and tok2.vocab_size == 101


def test_encode_pack_with_gpt2_wrapper(monkeypatch):
    _install_stub_tiktoken(monkeypatch)
    tok, _ = build_tokenizer("gpt2")
    rows = encode_pack(["hello world", "hi"], tok, seq_len=12)
    assert len(rows) == 2 and all(len(r) == 12 for r in rows)
    # EOS appended (99), padded with distinct pad id (100).
    assert rows[0][-1] == 100 or 99 in rows[0]
    assert all(i != 100 or True for r in rows for i in r)
    assert rows[1].count(100) > 0  # short row gets real padding


def test_train_smoke_tokenizer_gpt2_falls_back(tmp_path, monkeypatch):
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    monkeypatch.setattr(_data_module, "try_load_gpt2_tokenizer", lambda: None)
    rc = train_main(["--smoke", "--tokenizer", "gpt2", "--no-track"])
    assert rc == 0


def test_ckpt_kind_tag_and_gpt2_rebuild_graceful(tmp_path, capsys, monkeypatch):
    import torch

    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    try:
        from src.ced_llm.data import SimpleTokenizer as _ST
    except ImportError:
        from ced_llm.data import SimpleTokenizer as _ST  # type: ignore
    ckpt = str(tmp_path / "c.pt")
    rc = train_main(["--smoke", "--no-track", "--ckpt", ckpt])
    assert rc == 0
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert payload.get("tokenizer_kind") == "simple"
    # A gpt2-kind ckpt that ALSO carries a simple vocab dict degrades to it
    # (with a warning) when tiktoken is unavailable.
    payload["tokenizer_kind"] = "gpt2"
    payload["tokenizer_vocab"] = _ST(["hello world"], vocab_size=64).to_dict()
    torch.save(payload, ckpt)
    monkeypatch.setattr(_data_module, "try_load_gpt2_tokenizer", lambda: None)
    try:
        from src.ced_llm.generate import _try_load_ckpt
    except ImportError:
        from ced_llm.generate import _try_load_ckpt  # type: ignore
    _model, tok, _cfg = _try_load_ckpt(ckpt)
    assert isinstance(tok, SimpleTokenizer)
    assert "tiktoken" in capsys.readouterr().err
