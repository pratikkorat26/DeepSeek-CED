"""Instruction-tuning data for CED (SFT stage: learn to FOLLOW, not just continue).

Builds synthetic (prompt, response) pairs offline from the same TinyStories
atom pool as ``data.py``: story requests, continuations, and follow-up
questions. Labels mask the prompt (``-100``) so ``compute_loss`` trains on
response tokens only -- the core SFT mechanic (stage 1 of every alignment
pipeline, DeepSeek's SFT-before-GRPO included).

Usage:
    python3 -m src.ced_llm.train --data instructions --init checkpoints/tinysmall.pt \\
        --steps 300 --lr 1e-4 --ckpt checkpoints/tinysmall-sft.pt
"""

import random

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

try:
    from .data import (
        _DictDataset,
        _synthetic_generator,
        _tokenizer_specials,
        _tk_encode,
    )
except Exception:
    from src.ced_llm.data import (  # type: ignore
        _DictDataset,
        _synthetic_generator,
        _tokenizer_specials,
        _tk_encode,
    )

IGNORE = -100

_STORY_ASK = [
    "Tell me a short story about {s}.",
    "Write a tiny tale with {s} and {o}.",
    "Make up a bedtime story about {s}.",
    "Tell a story where {s} finds {o}.",
]

_CONTINUE_ASK = [
    "Continue this story: {p}",
    "What happens next? {p}",
    "Finish the tale: {p}",
]

_QUESTION_ASK = [
    "What did {s} do next?",
    "Where did {s} go?",
    "Who did {s} meet?",
]


def _story_texts(n, seed=0):
    """Deterministic synthetic stories (offline, no downloads)."""
    out = []
    try:
        for ex in _synthetic_generator(max_examples=max(1, int(n) * 2)):
            t = ex.get("text", "") if isinstance(ex, dict) else str(ex)
            if t and t.strip():
                out.append(t.strip())
            if len(out) >= max(1, int(n)):
                break
    except Exception:
        pass
    if not out:
        out = ["Once upon a time there was a little bunny."]
    return out


def build_instruction_pairs(n=200, seed=0):
    """Return list[(prompt, response)] cycling request/continue/question."""
    rng = random.Random(int(seed))
    stories = _story_texts(n, seed)
    try:
        from .data import _SYNTH_SUBJECTS, _SYNTH_OBJECTS
    except Exception:
        try:
            from src.ced_llm.data import _SYNTH_SUBJECTS, _SYNTH_OBJECTS  # type: ignore
        except Exception:
            _SYNTH_SUBJECTS, _SYNTH_OBJECTS = ["bunny"], ["ball"]
    pairs = []
    i = 0
    while len(pairs) < max(1, int(n)):
        story = stories[i % len(stories)]
        s = _SYNTH_SUBJECTS[i % len(_SYNTH_SUBJECTS)]
        o = _SYNTH_OBJECTS[(i * 3) % len(_SYNTH_OBJECTS)]
        mode = i % 3
        if mode == 0:
            prompt = rng.choice(_STORY_ASK).format(s=s, o=o)
            response = story
        elif mode == 1:
            words = story.split()
            cut = max(4, len(words) // 2)
            prompt = rng.choice(_CONTINUE_ASK).format(p=" ".join(words[:cut]))
            response = " ".join(words[cut:]) or story
        else:
            prompt = rng.choice(_QUESTION_ASK).format(s=s)
            response = story
        pairs.append((prompt, response))
        i += 1
    return pairs


def encode_sft_pair(tokenizer, prompt, response, seq_len=128):
    """Encode one pair -> (row_ids, label_ids); prompt masked with -100."""
    try:
        seq_len = max(2, int(seq_len))
    except Exception:
        seq_len = 128
    eos = _tokenizer_specials(tokenizer, "eos", default_none=True)
    pad = _tokenizer_specials(tokenizer, "pad", default_none=True)
    if pad is None:
        pad = 0
    p_ids = _tk_encode(tokenizer, prompt)
    r_ids = _tk_encode(tokenizer, response)
    # Budget: response first (it carries the learning signal), prompt keeps tail.
    max_resp = max(1, seq_len - 1 - min(len(p_ids), seq_len // 2))
    r_ids = r_ids[:max_resp]
    room = max(0, seq_len - len(r_ids) - (1 if eos is not None else 0))
    p_ids = p_ids[-room:] if room < len(p_ids) else p_ids
    row, lab = list(p_ids), [IGNORE] * len(p_ids)
    row.extend(r_ids)
    lab.extend(list(r_ids))
    if eos is not None:
        row.append(int(eos))
        lab.append(int(eos))
    row = row[:seq_len]
    lab = lab[:seq_len]
    while len(row) < seq_len:
        row.append(int(pad))
        lab.append(IGNORE)
    return row, lab


def get_sft_dataloader(tokenizer, seq_len=128, batch_size=8, max_examples=200,
                       seed=0, shuffle=True):
    """Loader yielding {input_ids, attention_mask, labels} (labels: -100 on prompt)."""
    try:
        seq_len = max(2, int(seq_len))
    except Exception:
        seq_len = 128
    try:
        batch_size = max(1, int(batch_size))
    except Exception:
        batch_size = 8
    try:
        max_examples = max(1, int(max_examples))
    except Exception:
        max_examples = 200
    pairs = build_instruction_pairs(max_examples, seed)
    rows, labs = [], []
    for prompt, response in pairs:
        r, l = encode_sft_pair(tokenizer, prompt, response, seq_len)
        rows.append(r)
        labs.append(l)
    if not rows:
        rows = [[0] * seq_len]
        labs = [[IGNORE] * seq_len]
    input_ids = torch.tensor(rows, dtype=torch.long)
    labels = torch.tensor(labs, dtype=torch.long)
    pad = _tokenizer_specials(tokenizer, "pad", default_none=True)
    if pad is None:
        pad = 0
    attention_mask = (input_ids != int(pad)).long()
    ds = _DictDataset(input_ids, attention_mask, labels=labels)
    try:
        loader = DataLoader(ds, batch_size=batch_size, shuffle=bool(shuffle))
    except Exception:
        loader = DataLoader(ds, batch_size=batch_size)
    return loader
