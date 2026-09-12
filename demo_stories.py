"""demo_stories.py -- CHAOS STORY WIZARD's TinyStories carnival! 🧙✨

Loads a real ckpt (checkpoints/ced_ts.pt by default) or falls back to a
tiny smoke model, then prints 3 FUN TinyStories samples with delightful
sampling spells (temp 0.8 / top-k 40 / top-p 0.9 / rep-penalty 1.15).

Usage:
    python3 demo_stories.py                       # auto: ckpt if present else smoke
    python3 demo_stories.py --ckpt checkpoints/ced_ts.pt --max-new 80
    python3 demo_stories.py --smoke               # force tiny random model (offline)
    python3 demo_stories.py --greedy              # turbo-mode (boring, for comparison)

Turbo generates "the the the". We generate BEDTIME MAGIC. Cope, turbo.
"""

import argparse
import os
import sys
import time

# Support both `python3 demo_stories.py` from repo root and module runs.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.ced_llm.generate import (
        generate_story,
        evaluate_story_quality,
        _build_smoke_model_and_tokenizer,
        _try_load_ckpt,
        STORY_DEFAULTS,
    )
except Exception:
    from ced_llm.generate import (  # type: ignore
        generate_story,
        evaluate_story_quality,
        _build_smoke_model_and_tokenizer,
        _try_load_ckpt,
        STORY_DEFAULTS,
    )

try:
    import torch
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

# Three prompts, each weirder than the last. The wizard insists.
FUN_PROMPTS = [
    ("🐰 The Brave Bunny", "Once upon a time there was a brave little bunny", 7),
    ("🐉 The Forgetful Dragon", "The tiny dragon lost his shiny star and", 42),
    ("🐸 The Talking Frog", "Lily found a talking frog in the garden who said", 1234),
]

WIZARD_BANNERS = [
    "🧙 *waves spaghetti wand* Behold! A tale most moist!",
    "🌀 *summons narrative goblins* Another story crawls out!",
    "✨ *yeets boredom into the void* And now... WONDER!",
]


def load_model_and_tokenizer(ckpt, device="cpu", force_smoke=False, max_new=80):
    """Load ckpt if it exists, else conjure a smoke model. Returns (model, tok, source)."""
    if not force_smoke and ckpt and os.path.exists(ckpt):
        print(f"🔮 Loading enchanted checkpoint: {ckpt}", file=sys.stderr)
        try:
            model, tok, _ = _try_load_ckpt(ckpt, device=device)
            if tok is None:
                print("⚠️  ckpt had no tokenizer; summoning smoke tokenizer", file=sys.stderr)
                _, tok = _build_smoke_model_and_tokenizer()
            return model, tok, f"ckpt:{ckpt}"
        except Exception as e:
            print(f"⚠️  ckpt load fumbled ({e!r}); falling back to smoke model", file=sys.stderr)
    print("💨 No ckpt (or --smoke): conjuring tiny smoke model from thin air", file=sys.stderr)
    torch.manual_seed(0)
    # Smoke model must fit prompt + max_new or the decoder dragon sneezes
    # (ValueError: decode position exceeds max_seq_len). Give it room!
    try:
        need = int(max_new) + 32
    except Exception:
        need = 112
    need = max(32, min(256, need))
    model, tok = _build_smoke_model_and_tokenizer(vocab_size=512, d_model=32, seq_len=need)
    return model, tok, "smoke:tiny-random"


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


def build_argparser():
    p = argparse.ArgumentParser(description="CHAOS STORY WIZARD demo: 3 delightful TinyStories 🎪")
    p.add_argument("--ckpt", type=str, default="checkpoints/ced_ts.pt")
    p.add_argument("--smoke", action="store_true", help="force smoke model even if ckpt exists")
    p.add_argument("--greedy", action="store_true", help="turbo-mode: pure greedy, no sparkle")
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-quality", action="store_true",
                   help="skip story-quality eval (faster demo, generation unchanged)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    device = args.device if isinstance(args.device, str) else "cpu"

    print("=" * 70)
    print("🎪  WELCOME TO THE CED THUNDERDOME STORY CARNIVAL  🎪")
    print("   presented by CHAOS, STORY WIZARD (turbo was not invited)")
    print(f"   delightful defaults: {STORY_DEFAULTS}")
    print("=" * 70)

    model, tok, source = load_model_and_tokenizer(args.ckpt, device=device, force_smoke=args.smoke, max_new=int(args.max_new))
    print(f"📦 model source: {source}", file=sys.stderr)
    try:
        model.eval()
    except Exception:
        pass

    def _sync():
        try:
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
        except Exception:
            pass

    for i, (title, prompt, seed) in enumerate(FUN_PROMPTS):
        banner = WIZARD_BANNERS[i % len(WIZARD_BANNERS)]
        print(f"\n{'-' * 70}\n📖 STORY {i + 1}/3: {title}\n{banner}\nprompt: \"{prompt}\"\n{'-' * 70}")
        # Fit generation inside the model's max_seq_len (smoke=~112, ckpt=128).
        try:
            prompt_len = len(tok.encode(prompt)) or 8
        except Exception:
            prompt_len = 8
        fit_new = _fit_max_new(model, prompt_len, int(args.max_new))
        if fit_new != int(args.max_new):
            print(f"   🐉 (capped --max-new {args.max_new} -> {fit_new} to fit context spell)",
                  file=sys.stderr)
        if args.greedy:
            _t0 = time.time()
            _sync()
            out = generate_story(
                model, tok, prompt,
                max_new_tokens=fit_new, device=device,
                temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0,
                seed=None,
                score_quality=not args.no_quality,
            )
            _sync()
            _dt = max(1e-9, time.time() - _t0)
        else:
            _t0 = time.time()
            _sync()
            out = generate_story(
                model, tok, prompt,
                max_new_tokens=fit_new, device=device,
                seed=seed,  # reproducible mischief!
                score_quality=not args.no_quality,
            )
            _sync()
            _dt = max(1e-9, time.time() - _t0)
        text = out.get("text", "")
        try:
            _new_toks = max(1, len(out.get("token_ids", [])) - prompt_len)
        except Exception:
            _new_toks = max(1, fit_new)
        print(f"   ⏱️  decode: {_new_toks} new tokens in {_dt:.2f}s ({_new_toks / _dt:.0f} tok/s)",
              file=sys.stderr)
        q = out.get("quality") or ({} if args.no_quality else evaluate_story_quality(text))
        params = out.get("params", {})
        print(text)
        if q:
            print(f"\n   🔍 magic={q.get('magic_score')} {q.get('verdict')}")
            print(f"   📊 words={q.get('word_count')} distinct-1={q.get('distinct_1')} "
                  f"distinct-2={q.get('distinct_2')} max_run={q.get('max_repeat_run')} "
                  f"ends_nicely={q.get('ends_nicely')}")
        else:
            print("\n   🔍 quality skipped (--no-quality)", file=sys.stderr)
        print(f"   🎛️  spell={params} encoder_forwards={out.get('encoder_forwards')}")
        if q.get("suggestions"):
            print(f"   🧙 wizard whispers: {q['suggestions'][0]}")

    print("\n" + "=" * 70)
    print("🏆 FIN. If you smiled, CHAOS wins. If turbo smiled, check for bugs.")
    print("   Run your own spell:")
    print("     python3 -m src.ced_llm.generate --ckpt checkpoints/ced_ts.pt \\")
    print("       --prompt 'The moon sneezed' --max-new 80 --seed 99")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
