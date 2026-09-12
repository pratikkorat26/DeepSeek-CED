"""Eval for minimal dense CED LLM: perplexity + generation samples.

Offline-first, CPU-friendly. Reports mean loss + perplexity on synthetic
fallback data (or an optional TinyStories sample when online), then prints
3 generation samples (proving encoder-once KV-reuse along the way).

Usage:
    python3 -m src.ced_llm.eval --smoke            # tiny random model, <60s CPU
    python3 -m src.ced_llm.eval --ckpt ckpt.pt     # eval a trained checkpoint
    python3 -m src.ced_llm.eval --data tinystories --max-examples 200

No network required: every path falls back to synthetic TinyStories-style
texts + SimpleTokenizer when `datasets` is missing or offline.
"""

import argparse
import os
import sys

try:
    import torch
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

# -- data imports (support `python -m src.ced_llm.eval` and pytest layouts) --
try:
    from .data import SimpleTokenizer, encode_pack, get_dataloader
except Exception:
    try:
        from src.ced_llm.data import SimpleTokenizer, encode_pack, get_dataloader  # type: ignore
    except Exception:
        import importlib.util as _ilu

        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = _ilu.spec_from_file_location(
            "_ced_data_fallback", os.path.join(_here, "data.py")
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # type: ignore
        SimpleTokenizer = _mod.SimpleTokenizer
        encode_pack = _mod.encode_pack
        get_dataloader = _mod.get_dataloader

# -- train helpers (evaluate / seeding) --
try:
    from .train import evaluate, set_seed
except Exception:
    try:
        from src.ced_llm.train import evaluate, set_seed  # type: ignore
    except Exception:
        import importlib.util as _ilu2

        _here2 = os.path.dirname(os.path.abspath(__file__))
        _spec2 = _ilu2.spec_from_file_location(
            "_ced_train_fallback", os.path.join(_here2, "train.py")
        )
        _mod2 = _ilu2.module_from_spec(_spec2)
        _spec2.loader.exec_module(_mod2)  # type: ignore
        evaluate = _mod2.evaluate
        set_seed = _mod2.set_seed

# -- generate helpers (smoke model, ckpt loading, generation) --
try:
    from .generate import (
        _build_smoke_model_and_tokenizer,
        _try_load_ckpt,
        generate_greedy,
        generate_story,
    )
except Exception:
    try:
        from src.ced_llm.generate import (  # type: ignore
            _build_smoke_model_and_tokenizer,
            _try_load_ckpt,
            generate_greedy,
            generate_story,
        )
    except Exception:
        import importlib.util as _ilu3

        _here3 = os.path.dirname(os.path.abspath(__file__))
        _spec3 = _ilu3.spec_from_file_location(
            "_ced_gen_fallback", os.path.join(_here3, "generate.py")
        )
        _mod3 = _ilu3.module_from_spec(_spec3)
        _spec3.loader.exec_module(_mod3)  # type: ignore
        _build_smoke_model_and_tokenizer = _mod3._build_smoke_model_and_tokenizer
        _try_load_ckpt = _mod3._try_load_ckpt
        generate_greedy = _mod3.generate_greedy
        generate_story = _mod3.generate_story

# Three fixed eval prompts (mirrors the demo carnival, minus the glitter).
EVAL_PROMPTS = [
    ("eval-1", "Once upon a time there was a brave little bunny", 7),
    ("eval-2", "The tiny dragon lost his shiny star and", 42),
    ("eval-3", "Lily found a talking frog in the garden who said", 1234),
]


def _fit_max_new(model, prompt_len, want):
    """Cap max_new so prompt + generation fits model.config.max_seq_len."""
    try:
        cap = int(getattr(getattr(model, "config", None), "max_seq_len", 0) or 0)
    except Exception:
        cap = 0
    if cap and cap > 0:
        try:
            room = cap - int(prompt_len) - 1
            if room < 1:
                return 1
            return max(1, min(int(want), room))
        except Exception:
            return want
    return want


def _prompt_len(tokenizer, prompt):
    try:
        return len(tokenizer.encode(prompt)) or 8
    except Exception:
        return 8


def build_argparser():
    p = argparse.ArgumentParser(description="Eval CED LM: perplexity + 3 samples")
    p.add_argument("--smoke", action="store_true",
                   help="fast offline eval: tiny random model, synthetic data (<60s CPU)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="checkpoint path (ckpts from train --ckpt); else tiny random model")
    p.add_argument("--data", type=str, default="synthetic",
                   choices=["synthetic", "tinystories"],
                   help="synthetic = offline fallback texts; tinystories = HF sample if online, else synthetic")
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-examples", type=int, default=200)
    p.add_argument("--max-new", type=int, default=32,
                   help="tokens per generation sample")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p


def _resolve_model_and_tokenizer(args, device):
    """Return (model, tok, source). Prefers ckpt, else tiny random model."""
    if args.ckpt and os.path.exists(args.ckpt):
        print("[eval] loading ckpt %s" % args.ckpt)
        try:
            model, tok, _ = _try_load_ckpt(args.ckpt, device=device)
            if tok is None:
                print("[eval] ckpt had no tokenizer; using smoke tokenizer",
                      file=sys.stderr)
                _, tok = _build_smoke_model_and_tokenizer()
            return model, tok, "ckpt:%s" % args.ckpt
        except Exception as e:
            print("[eval] ckpt load failed (%r); using tiny random model" % (e,),
                  file=sys.stderr)
    seq_need = max(32, min(256, max(int(args.seq_len), int(args.max_new) + 32)))
    torch.manual_seed(int(args.seed))
    model, tok = _build_smoke_model_and_tokenizer(
        vocab_size=512, d_model=32, seq_len=seq_need)
    return model, tok, "smoke:tiny-random"


def _resolve_loader(args, tok, model=None):
    """Return eval DataLoader: synthetic fallback, or TinyStories sample."""
    seq_len = max(2, int(args.seq_len))
    if model is not None:
        # Never ask for rows longer than the model can score; otherwise
        # evaluate() would skip every batch and report a bogus loss of 0.
        try:
            cap = int(getattr(getattr(model, "config", None), "max_seq_len", 0) or 0)
        except Exception:
            cap = 0
        if cap and cap > 0 and seq_len > cap:
            print("[eval] capping --seq-len %d -> %d (model max_seq_len)" % (
                seq_len, cap), file=sys.stderr)
            seq_len = cap
    batch_size = max(1, int(args.batch_size))
    max_examples = max(1, int(args.max_examples))
    split = "train" if args.data == "tinystories" else "train"
    # get_dataloader tries HF TinyStories first for tinystories-flavoured
    # corpora and ALWAYS falls back to synthetic offline texts, so both
    # --data choices are offline-safe. For --data synthetic we keep the
    # sample small and deterministic (seeded shuffle off).
    loader = get_dataloader(
        tok, split=split, seq_len=seq_len, batch_size=batch_size,
        max_examples=max_examples, shuffle=False,
    )
    return loader


def run_eval(model, tok, loader, device, max_new, delightful=False):
    """Evaluate perplexity + 3 generation samples. Returns results dict."""
    res = evaluate(model, loader, device=device)
    loss, ppl = float(res["loss"]), float(res["ppl"])
    print("[eval] loss=%.4f ppl=%.2f" % (loss, ppl))
    samples = []
    for name, prompt, seed in EVAL_PROMPTS:
        fit_new = _fit_max_new(model, _prompt_len(tok, prompt), int(max_new))
        if delightful:
            out = generate_story(model, tok, prompt, max_new_tokens=fit_new,
                                 device=device, seed=seed)
            q = out.get("quality", {})
            print("-" * 60)
            print("[eval] %s prompt=%r seed=%d" % (name, prompt, seed))
            print(out.get("text", "")[:500])
            print("[eval] %s magic=%s encoder_forwards=%s" % (
                name, q.get("magic_score"), out.get("encoder_forwards")))
        else:
            out = generate_greedy(model, tok, prompt, max_new_tokens=fit_new,
                                  device=device)
            print("-" * 60)
            print("[eval] %s prompt=%r" % (name, prompt))
            print(out.get("text", "")[:500])
            print("[eval] %s tokens=%d encoder_forwards=%d" % (
                name, len(out.get("token_ids", [])),
                out.get("encoder_forwards")))
        samples.append(out)
    return {"loss": loss, "ppl": ppl, "samples": samples}


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    device = args.device if isinstance(args.device, str) else "cpu"

    if args.smoke:
        # Tiny + fast: 32-token context, 32 synthetic examples, greedy 16-token
        # samples. Runs in seconds on CPU (budget: <60s).
        print("[eval] SMOKE mode: tiny random model, synthetic data")
        torch.manual_seed(int(args.seed))
        model, tok = _build_smoke_model_and_tokenizer(
            vocab_size=512, d_model=32, seq_len=32)
        loader = get_dataloader(tok, split="train", seq_len=32,
                                batch_size=4, max_examples=32, shuffle=False)
        res = run_eval(model, tok, loader, device="cpu", max_new=16,
                       delightful=False)
        for s in res["samples"]:
            assert s.get("encoder_forwards") == 1, "smoke KV-reuse check failed"
        assert len(res["samples"]) == 3, "smoke must produce 3 samples"
        print("[eval] SMOKE OK (loss=%.4f ppl=%.2f samples=3)" % (
            res["loss"], res["ppl"]))
        return 0

    # ---- Non-smoke ------------------------------------------------------
    model, tok, source = _resolve_model_and_tokenizer(args, device)
    print("[eval] model source: %s | data: %s" % (source, args.data),
          file=sys.stderr)
    loader = _resolve_loader(args, tok, model)
    res = run_eval(model, tok, loader, device, int(args.max_new),
                   delightful=True)
    print("[eval] DONE loss=%.4f ppl=%.2f samples=%d (source=%s data=%s)" % (
        res["loss"], res["ppl"], len(res["samples"]), source, args.data))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
