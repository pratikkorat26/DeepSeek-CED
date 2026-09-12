# CONTRIBUTING.md — How to Enter the Thunderdome Without Dying ⚡

Welcome, brave contributor. This repo is small, dense, offline-first, and
protected by 22 tests with VERY specific failure messages. Follow these rules
and you'll leave a legend. Break them and you'll leave a cautionary tale in
`THUNDERDOME_DOCS.md`.

## 0. The one command that matters

```bash
pytest -q
```

22 passed or it didn't happen. Run it BEFORE you push, AFTER you push, and at
least once while staring dramatically out a window.

## 1. Know thy lanes (Thunderdome boundaries are sacred)

This repo runs rival lanes. Respect them:

- **Train/model/attention/encoder/decoder** — the speed lane. Benchmark before
  AND after; keep `test_kv_reuse.py` parity ≤1e-4 or revert.
- **`generate.py`** — the story lane. Keep `generate_greedy` backward-compatible
  (defaults `temp 0.0 / top_k 0 / top_p 1.0 / rep 1.0`); greedy stays pure argmax.
- **Docs/tests** — the lore lane (hi 👋). Docs changes must keep every command
  copy-paste-runnable. Test changes must FAIL on the bug they claim to catch
  (mutation-check by hand: break the code, watch the test scream, fix it back).

If your PR touches a lane that isn't yours, say so LOUDLY in the PR description
and bring numbers (timings) or stories (samples). Surprises get reverted.

## 2. Setup (30 seconds)

```bash
pip install -r requirements.txt   # torch>=2.0 + pytest required; rest optional
pytest -q                         # confirm 22 green on YOUR box before changing anything
python3 -m src.ced_llm.train --smoke
python3 -m src.ced_llm.generate --smoke
```

No internet? No problem. Everything falls back to synthetic data + word-level
tokenizer. If your change NEEDS the network, it will be rejected with prejudice
and a haiku.

## 3. The CED contracts (violations fail tests and hearts)

1. **Encoder runs ONCE per generation.** `init_decode_cache` → one forward.
   `forward_step` never re-encodes. `encoder_forward_count` delta must be 1.
2. **Logits align 1:1; the CALLER shifts.** `x_in=tok[:, :-1]`, `y=tok[:, 1:]`,
   `CE(logits, y, ignore_index=pad)`. Don't "fix" the shift inside the model.
3. **Causality lives in self-attn; cross-attn is GLOBAL.** Full-logits prefix
   invariance does NOT hold (by design). Don't add a test asserting it. Don't
   "fix" cross-attn to be causal. Read `ARCHITECTURE.md` §2.2 first.
4. **Length contract:** output is `prompt + max_new` unless it ends in EOS.
   Early-stop-without-EOS is a bug. `T > max_seq_len` raises `ValueError` —
   callers truncate (keep the tail).
5. **Greedy is sacred:** `temperature=0` = pure argmax, deterministic, ignores
   top-k/top-p. Sampled paths take `seed=` for reproducibility.

## 4. PR checklist (copy-paste this, check every box)

- [ ] `pytest -q` → 22 passed (paste the line)
- [ ] `python3 -m src.ced_llm.train --smoke` → SMOKE OK (paste loss line)
- [ ] `python3 -m src.ced_llm.generate --smoke` → SMOKE OK (`encoder_forwards=1`)
- [ ] No new required dependencies (or justified + fallback-covered)
- [ ] Docs updated if behavior/flags changed (`README.md` commands re-tested)
- [ ] New tests for new behavior (and each new test FAILS without the fix)
- [ ] No secrets, no checkpoints, no 500MB "oops" files committed
- [ ] No drive-by refactors of someone else's lane

## 5. Style (we're small, be kind to readers)

- Python, type hints where cheap, docstrings on public functions.
- Keep the tiny-CPU-smoke ethos: default paths must run offline in <120s.
- Failure messages should NAME the violated contract and the fix
  (see `tests/test_kv_reuse.py` for the gold standard of disappointed errors).
- Comments explain WHY, not WHAT. If the code needs a paragraph, the code may
  be wrong — or legendary. Ask in the PR.

## 6. Reporting bugs (make the Lore Master happy)

Include: command run, FULL error output, `pytest -q` result, torch version,
and whether you were online. Best bug reports get immortalized. Worst ones get
immortalized differently.

---
*Encoder runs once. Tests run always. See you in the Thunderdome.* ⚡📚
