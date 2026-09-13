"""Data utilities for minimal dense CED LLM (TinyStories, offline-first).

Offline-first: every helper works with NO network / NO optional packages.
- SimpleTokenizer: word-level fallback, no downloads.
- load_tinystories: tries `datasets`, falls back to synthetic generator.
- try_load_gpt2_tokenizer: tries `tiktoken`, else None.
"""

import random
import re
from collections import Counter

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except Exception as e:  # torch is required; fail loudly if truly missing
    raise ImportError("torch>=2.0 is required for src.ced_llm.data") from e


# ---------------------------------------------------------------------------
# SimpleTokenizer
# ---------------------------------------------------------------------------
class SimpleTokenizer:
    """Word-level fallback tokenizer that works OFFLINE with no downloads.

    Normalization: lowercase + whitespace split (``" ".join(s.lower().split())``).
    ``decode(encode(s))`` recovers the normalized ``s`` (for in-vocab text).
    """

    def __init__(self, texts=None, vocab_size=8000):
        self.pad_id = 0
        self.unk_id = 1
        self.bos_id = 2
        self.eos_id = 3
        self._specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        # vocab_size param is a CAP; attribute is the ACTUAL size.
        try:
            cap = int(vocab_size)
        except Exception:
            cap = 8000
        self._vocab_cap = max(4, cap)

        self.token_to_id = {tok: i for i, tok in enumerate(self._specials)}
        self.id_to_token = {i: tok for tok, i in self.token_to_id.items()}

        if texts:
            counter = Counter()
            for t in texts:
                if isinstance(t, dict):
                    t = t.get("text", "")
                if not isinstance(t, str):
                    try:
                        t = str(t)
                    except Exception:
                        continue
                toks = self.normalize(t).split()
                counter.update(toks)
            budget = self._vocab_cap - len(self._specials)
            for tok, _ in counter.most_common(max(0, budget)):
                if tok not in self.token_to_id:
                    idx = len(self.token_to_id)
                    if idx >= self._vocab_cap:
                        break
                    self.token_to_id[tok] = idx
                    self.id_to_token[idx] = tok
        # Public attribute: actual vocab size.
        self.vocab_size = len(self.token_to_id)

    @staticmethod
    def normalize(text):
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return ""
        return " ".join(text.lower().split())

    def encode(self, text):
        """Encode text -> list[int] (no BOS/EOS added; see encode_pack)."""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception:
                return []
        norm = self.normalize(text)
        if not norm:
            return []
        ids = []
        for tok in norm.split():
            ids.append(self.token_to_id.get(tok, self.unk_id))
        return ids

    def decode(self, ids):
        """Decode ids -> str. Skips pad/bos/eos; unk -> <unk>."""
        # Accept torch tensors as well (duck typing).
        try:
            import torch as _torch

            if isinstance(ids, _torch.Tensor):
                ids = ids.tolist()
        except Exception:
            pass
        toks = []
        for i in ids:
            try:
                i = int(i)
            except Exception:
                continue
            if i in (self.pad_id, self.bos_id, self.eos_id):
                continue
            if i == self.unk_id:
                toks.append("<unk>")
            else:
                toks.append(self.id_to_token.get(i, "<unk>"))
        return " ".join(toks)

    def __len__(self):
        return self.vocab_size

    # -- helpers for ckpt save/load --------------------------------------
    def get_vocab(self):
        return dict(self.token_to_id)

    def to_dict(self):
        return {
            "token_to_id": dict(self.token_to_id),
            "vocab_size_cap": self._vocab_cap,
        }

    @classmethod
    def from_vocab(cls, token_to_id, vocab_cap=8000):
        obj = cls(texts=None, vocab_size=vocab_cap)
        obj.token_to_id = dict(token_to_id)
        obj.id_to_token = {i: t for t, i in obj.token_to_id.items()}
        # Ensure specials exist with canonical ids.
        for tok, idx in [("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3)]:
            obj.token_to_id.setdefault(tok, idx)
            obj.id_to_token.setdefault(idx, tok)
        obj.vocab_size = len(obj.token_to_id)
        return obj

    @classmethod
    def from_dict(cls, d):
        return cls.from_vocab(
            d.get("token_to_id", {}), vocab_cap=d.get("vocab_size_cap", 8000)
        )


# ---------------------------------------------------------------------------
# Optional GPT-2 (tiktoken) tokenizer
# ---------------------------------------------------------------------------
def try_load_gpt2_tokenizer():
    """Attempt ``tiktoken.get_encoding('gpt2')``; return None on ANY failure."""
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.get_encoding("gpt2")
            return enc
        except Exception:
            return None
    except Exception:
        return None


class GPT2Tokenizer:
    """GPT-2 BPE via tiktoken, wrapped in the SimpleTokenizer interface.

    Requires ``pip install tiktoken`` AND the cached BPE file (one-time
    download); construction returns None-safe via :func:`build_tokenizer`
    which falls back to :class:`SimpleTokenizer` offline.

    Conventions (documented deltas vs SimpleTokenizer):
      * ``eos_id = n_vocab - 1`` (50256 ``<|endoftext|>`` for real GPT-2);
        ``bos_id = None`` (GPT-2 has no BOS, so ``encode_pack`` adds none);
        ``unk_id = None`` (BPE has no OOV -- every byte sequence encodes).
      * ``pad_id = n_vocab``: one row PAST the BPE vocab, so padding never
        collides with real tokens (masks compare ids to pad). The model
        embedding is therefore sized ``vocab_size = n_vocab + 1`` (50258 for
        GPT-2; ~25.7 MB fp32 at d_model=128 -- laptop-safe).
      * ``decode`` delegates to tiktoken (full fidelity, no skipping).
    """

    kind = "gpt2"
    name = "gpt2"

    def __init__(self, enc):
        self._enc = enc
        try:
            base = int(enc.n_vocab)
        except Exception:
            base = 50257
        self.base_vocab = base
        self.eos_id = base - 1
        self.eos_token_id = base - 1
        self.pad_id = base
        self.pad_token_id = base
        self.bos_id = None
        self.unk_id = None
        self.vocab_size = base + 1

    def encode(self, text):
        try:
            return list(self._enc.encode(str(text), disallowed_special=()))
        except TypeError:
            try:
                return list(self._enc.encode(str(text)))
            except Exception:
                return []
        except Exception:
            return []

    def decode(self, ids):
        try:
            import torch as _torch

            if isinstance(ids, _torch.Tensor):
                ids = ids.tolist()
        except Exception:
            pass
        try:
            return self._enc.decode([int(i) for i in ids])
        except Exception:
            return ""

    def __len__(self):
        return self.vocab_size

    def to_dict(self):
        return {"kind": "gpt2", "name": "gpt2", "vocab_size": self.vocab_size}

    @classmethod
    def from_dict(cls, d):
        enc = try_load_gpt2_tokenizer()
        if enc is None:
            return None
        return cls(enc)


def build_tokenizer(kind="simple", texts=None, vocab_size=8000):
    """Build (tokenizer, actual_kind). ``gpt2`` falls back to simple offline.

    Never raises for missing deps: when tiktoken (or its BPE download) is
    unavailable, prints a warning and returns a SimpleTokenizer.
    """
    try:
        key = str(kind or "simple").strip().lower()
    except Exception:
        key = "simple"
    if key in ("gpt2", "gpt-2", "gpt2-bpe"):
        enc = try_load_gpt2_tokenizer()
        if enc is not None:
            try:
                return GPT2Tokenizer(enc), "gpt2"
            except Exception:
                pass
        print("[tok] WARNING: --tokenizer gpt2 requested but tiktoken (or its "
              "BPE download) is unavailable; falling back to SimpleTokenizer "
              "(pip install tiktoken + network once to enable).")
    try:
        cap = int(vocab_size)
    except Exception:
        cap = 8000
    return SimpleTokenizer(texts, vocab_size=cap), "simple"


# ---------------------------------------------------------------------------
# TinyStories loading (with offline synthetic fallback)
# ---------------------------------------------------------------------------
_SYNTH_SUBJECTS = [
    "little girl",
    "little boy",
    "bunny",
    "puppy",
    "kitten",
    "bird",
    "frog",
    "bear",
    "fox",
    "turtle",
]
_SYNTH_OBJECTS = [
    "big red ball",
    "tiny blue box",
    "shiny star",
    "green hill",
    "tall tree",
    "little house",
    "round cake",
    "happy song",
]
_SYNTH_VERBS = [
    "found",
    "saw",
    "liked",
    "shared",
    "hugged",
    "helped",
    "played with",
    "ran to",
]
_SYNTH_TAILS = [
    "and they were very happy.",
    "and they played all day.",
    "and then they went home.",
    "and they laughed together.",
    "and it was a sunny day.",
    "and everyone smiled.",
]


def _synthetic_generator(max_examples=None):
    """Deterministic synthetic TinyStories-like generator (never needs net)."""
    n = max_examples if max_examples is not None else 1000
    try:
        n = int(n)
    except Exception:
        n = 1000
    n = max(0, n)
    rng = random.Random(1234)
    for i in range(n):
        s = _SYNTH_SUBJECTS[i % len(_SYNTH_SUBJECTS)]
        v = _SYNTH_VERBS[(i * 3 + 1) % len(_SYNTH_VERBS)]
        o = _SYNTH_OBJECTS[(i * 7 + 2) % len(_SYNTH_OBJECTS)]
        t = _SYNTH_TAILS[(i * 5) % len(_SYNTH_TAILS)]
        extra = ""
        # Add variety without randomness blowup (deterministic via rng).
        if rng.random() < 0.5:
            extra = " Once upon a time there was a %s who %s a %s %s" % (s, v, o, t)
            text = extra.strip() + " They loved to play in the garden."
        else:
            text = "Once upon a time there was a %s. The %s %s a %s %s" % (
                s,
                s,
                v,
                o,
                t,
            )
        yield {"text": text}


def load_tinystories(split="train", streaming=True, max_examples=None):
    """Yield dicts ``{text: ...}`` from TinyStories, or synthetic fallback.

    Tries ``datasets.load_dataset('roneneldan/TinyStories', split=...)``.
    On ANY exception (no package / no network / bad split) falls back to a
    synthetic generator. Never raises for offline use.
    """
    # Normalize max_examples.
    if max_examples is not None:
        try:
            max_examples = int(max_examples)
        except Exception:
            max_examples = None

    # --- Try real HF dataset -------------------------------------------
    try:
        from datasets import load_dataset  # type: ignore

        try:
            ds = load_dataset("roneneldan/TinyStories", split=split, streaming=streaming)
        except Exception:
            # Retry non-streaming once (some environments dislike streaming).
            try:
                ds = load_dataset("roneneldan/TinyStories", split=split)
            except Exception:
                raise
        count = 0
        for ex in ds:
            try:
                if isinstance(ex, dict):
                    txt = ex.get("text", "")
                else:
                    txt = str(ex)
            except Exception:
                continue
            if not isinstance(txt, str):
                try:
                    txt = str(txt)
                except Exception:
                    continue
            if not txt or not txt.strip():
                continue
            yield {"text": txt}
            count += 1
            if max_examples is not None and count >= max_examples:
                return
        return
    except Exception:
        pass

    # --- Offline synthetic fallback (never crashes) ----------------------
    try:
        for ex in _synthetic_generator(max_examples):
            yield ex
    except Exception:
        # Absolute last resort: yield a single trivial example.
        yield {"text": "Once upon a time there was a little bunny."}


def build_tokenizer_from_corpus(texts, vocab_size=8000):
    """Build a SimpleTokenizer from an iterable of str/dict."""
    strs = []
    try:
        for t in texts:
            if isinstance(t, dict):
                strs.append(t.get("text", ""))
            elif isinstance(t, str):
                strs.append(t)
            else:
                try:
                    strs.append(str(t))
                except Exception:
                    continue
    except Exception:
        strs = []
    return SimpleTokenizer(strs, vocab_size=vocab_size)


# ---------------------------------------------------------------------------
# Packing / batching
# ---------------------------------------------------------------------------
def _tokenizer_specials(tokenizer, name, default_none=True):
    """Fetch bos/eos/pad ids via duck typing; None if absent (e.g. tiktoken)."""
    candidates = {
        "bos": ["bos_id", "bos_token_id"],
        "eos": ["eos_id", "eos_token_id"],
        "pad": ["pad_id", "pad_token_id"],
    }
    for attr in candidates.get(name, []):
        try:
            v = getattr(tokenizer, attr, None)
        except Exception:
            v = None
        if v is not None:
            try:
                return int(v)
            except Exception:
                return v
    return None if default_none else 0


def _tk_encode(tokenizer, text):
    """Duck-typed encode: works for SimpleTokenizer and tiktoken."""
    try:
        ids = tokenizer.encode(text)
        return list(ids)
    except TypeError:
        # tiktoken with strict special-token handling.
        try:
            return list(tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return []
    except Exception:
        return []


def encode_pack(texts, tokenizer, seq_len):
    """Pack texts -> list[list[int]] with BOS/EOS, truncation, pad to seq_len."""
    try:
        seq_len = int(seq_len)
    except Exception:
        seq_len = 128
    seq_len = max(1, seq_len)

    bos = _tokenizer_specials(tokenizer, "bos", default_none=True)
    eos = _tokenizer_specials(tokenizer, "eos", default_none=True)
    pad = _tokenizer_specials(tokenizer, "pad", default_none=True)
    if pad is None:
        pad = 0

    # Normalize input to list of strings.
    str_list = []
    try:
        for t in texts:
            if isinstance(t, dict):
                str_list.append(t.get("text", ""))
            elif isinstance(t, str):
                str_list.append(t)
            else:
                try:
                    str_list.append(str(t))
                except Exception:
                    str_list.append("")
    except Exception:
        str_list = []

    n_special = (1 if bos is not None else 0) + (1 if eos is not None else 0)
    max_content = max(0, seq_len - n_special)

    packed = []
    for s in str_list:
        ids = _tk_encode(tokenizer, s)
        # Truncate content, then add specials.
        ids = ids[:max_content]
        row = []
        if bos is not None:
            row.append(int(bos))
        row.extend([int(x) for x in ids])
        if eos is not None:
            row.append(int(eos))
        # Truncate defensively (in case encode already had specials).
        row = row[:seq_len]
        # Pad.
        if len(row) < seq_len:
            row = row + [int(pad)] * (seq_len - len(row))
        packed.append(row)
    return packed


class _DictDataset(Dataset):
    """Map-style dataset yielding dicts {input_ids, attention_mask}."""

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def get_dataloader(
    tokenizer, split="train", seq_len=128, batch_size=8, max_examples=500, shuffle=True,
    num_workers=0, persistent_workers=False, prefetch_factor=None,
    pin_memory=False,
):
    """Return a DataLoader yielding dicts {input_ids:[B,T], attention_mask:[B,T]}.

    Uses TensorDataset-style storage (via _DictDataset). T == seq_len.
    Works offline via synthetic fallback.

    Loader-throughput knobs (all optional, defaults preserve the historical
    single-process behavior exactly):
      num_workers: DataLoader workers (0 = main-process only, as before).
      persistent_workers: keep workers alive across epochs (only if >0).
      prefetch_factor: batches prefetched per worker (None = torch default;
        only forwarded when num_workers > 0, else torch raises).
      pin_memory: page-locked host buffers for faster H2D copies.
    """
    try:
        seq_len = int(seq_len)
    except Exception:
        seq_len = 128
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 8
    try:
        max_examples = int(max_examples)
    except Exception:
        max_examples = 500
    seq_len = max(1, seq_len)
    batch_size = max(1, batch_size)
    max_examples = max(1, max_examples)

    texts = []
    try:
        for ex in load_tinystories(
            split=split, streaming=True, max_examples=max_examples
        ):
            if isinstance(ex, dict):
                t = ex.get("text", "")
            else:
                t = str(ex)
            if t and isinstance(t, str) and t.strip():
                texts.append(t)
            if len(texts) >= max_examples:
                break
    except Exception:
        texts = []
    if not texts:
        # Guarantee non-empty even if loader misbehaves.
        texts = [ex["text"] for ex in _synthetic_generator(max_examples)]

    packed = encode_pack(texts, tokenizer, seq_len)
    if not packed:
        packed = [[0] * seq_len]

    input_ids = torch.tensor(packed, dtype=torch.long)
    pad = _tokenizer_specials(tokenizer, "pad", default_none=True)
    if pad is None:
        pad = 0
    attention_mask = (input_ids != int(pad)).long()

    ds = _DictDataset(input_ids, attention_mask)
    try:
        nw = int(num_workers)
    except Exception:
        nw = 0
    nw = max(0, nw)
    try:
        pin = bool(pin_memory)
    except Exception:
        pin = False
    try:
        persist = bool(persistent_workers) and nw > 0
    except Exception:
        persist = False
    kw = dict(batch_size=batch_size, shuffle=bool(shuffle),
              num_workers=nw, persistent_workers=persist, pin_memory=pin)
    # torch raises if prefetch_factor is set with num_workers==0.
    if nw > 0 and prefetch_factor is not None:
        try:
            kw["prefetch_factor"] = int(prefetch_factor)
        except Exception:
            pass
    try:
        loader = DataLoader(ds, **kw)
    except Exception:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=bool(shuffle))
    return loader


def get_toy_batch(batch_size=4, seq_len=32, vocab_size=512):
    """Random ids for offline smoke/tests (no downloads)."""
    try:
        batch_size = int(batch_size)
    except Exception:
        batch_size = 4
    try:
        seq_len = int(seq_len)
    except Exception:
        seq_len = 32
    try:
        vocab_size = int(vocab_size)
    except Exception:
        vocab_size = 512
    batch_size = max(1, batch_size)
    seq_len = max(1, seq_len)
    vocab_size = max(5, vocab_size)
    # Avoid specials 0..3 so masks stay all-ones and loss ignores nothing.
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}
