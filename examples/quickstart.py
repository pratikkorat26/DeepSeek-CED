"""Quickstart example: train-smoke, then generate 2 prompts end to end.

Runs fully offline on CPU in seconds:

    python3 examples/quickstart.py

Steps:
  1. Train smoke (``train --smoke`` toy run, asserts loss goes DOWN) and
     save the toy checkpoint to a temp file.
  2. Load that checkpoint back and generate 2 prompts greedily,
     proving the encoder ran exactly ONCE per generation.
"""

import os
import sys
import tempfile

# Make `examples/quickstart.py` work from the repo root (no install needed).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    from src.ced_llm.generate import _try_load_ckpt, generate_greedy
    from src.ced_llm.train import main as train_main

    print("=" * 70)
    print("STEP 1/2: train smoke (toy run, loss must go DOWN)")
    print("=" * 70)
    with tempfile.TemporaryDirectory(prefix="ced_quickstart_") as tmp:
        ckpt = os.path.join(tmp, "smoke.pt")
        rc = train_main(["--smoke", "--ckpt", ckpt, "--seed", "0"])
        assert rc == 0, "train --smoke failed (rc=%r)" % (rc,)
        assert os.path.exists(ckpt), "train --smoke saved no checkpoint"

        print("=" * 70)
        print("STEP 2/2: generate 2 prompts from the smoke checkpoint")
        print("=" * 70)
        model, tok, _ = _try_load_ckpt(ckpt, device="cpu")
        if tok is None:  # smoke ckpts carry no tokenizer; conjure the twin
            from src.ced_llm.generate import _build_smoke_model_and_tokenizer

            _, tok = _build_smoke_model_and_tokenizer()

        for i, prompt in enumerate(
            ["Once upon a time", "The little bunny found a"], start=1
        ):
            out = generate_greedy(model, tok, prompt, max_new_tokens=24,
                                  device="cpu")
            print("-" * 70)
            print("PROMPT %d/2: %r" % (i, prompt))
            print(out["text"])
            print("(tokens=%d encoder_forwards=%d -- the encoder ran ONCE)" % (
                len(out["token_ids"]), out["encoder_forwards"]))
            assert out["encoder_forwards"] == 1, "KV-reuse violated!"

    print("=" * 70)
    print("QUICKSTART OK: trained smoke, generated 2 prompts. Shine! ⚡📚")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
