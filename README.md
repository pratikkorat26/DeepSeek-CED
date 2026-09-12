# DeepSeek CED: Minimal Dense Context-Encoder-Decoder LLM ⚡📚

[![python](https://img.shields.io/badge/python-%3E%3D3.10-blue)](pyproject.toml)
[![torch](https://img.shields.io/badge/torch-%3E%3D2.0-orange)](requirements.txt)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> **CED = Context-Encoder-Decoder.** The encoder reads your prefix **ONCE** into
> memory. The decoder then generates token after token while **reusing cached
> keys/values** — the encoder never runs again. One encode. Infinite vibes.
> Future contributors: you will cry tears of joy. That's a promise, not a threat.

**What this repo is:** data + training + generation for a minimal **dense**
encoder-decoder-style LM on TinyStories. No MoE. No GQA/MLA/RoPE. No network
required. CPU-friendly, fully offline-capable via synthetic fallback data and a
word-level tokenizer. If your laptop can run a toaster, it can run this.

**What this repo is NOT:** a 70B moat, a chatbot with feelings, or a reason to
buy GPUs. (Turbo tried. Chaos cast spells. Docs won anyway.)

---

## ⚡ Quickstart (60 seconds to glory)

```bash
pip install -r requirements.txt
# torch>=2.0 and pytest are required; datasets/tiktoken are optional (fallback-covered)

pytest -q                                        # 35 tests, ~2s, all green or riot
python3 -m src.ced_llm.train --smoke             # toy run: asserts loss goes DOWN
python3 -m src.ced_llm.generate --smoke          # tiny model, 16 tokens, encoder runs x1
```

Expected smoke output (don't panic if numbers wiggle slightly by torch version):

```
 35 passed in ~2.2s
[train] SMOKE OK (loss 6.25 -> 4.83, the line goes down, stonks 📉📈)
[generate] SMOKE OK (tokens=... encoder_forwards=1, the encoder took ONE nap... er, pass)
```

### Real TinyStories run (downloads if online, synthetic fallback if not)

```bash
# Train (2000 examples, ~500 steps, CPU-ok):
python3 -m src.ced_llm.train --data tinystories --seq-len 128 --batch-size 8 \
  --steps 500 --lr 3e-4 --max-examples 2000 --ckpt ckpt.pt --seed 0

# Generate, greedy (deterministic, boring, turbo-approved):
python3 -m src.ced_llm.generate --ckpt ckpt.pt --prompt "Once upon a time" --max-new 64

# Generate, delightful (sampled, chaos-approved):
python3 -m src.ced_llm.generate --ckpt ckpt.pt --prompt "The little bunny" \
  --max-new 64 --temperature 0.8 --top-k 40

# Carnival mode (3 seeded stories + quality scores, if demo_stories.py exists):
python3 demo_stories.py --smoke
```

### Optimizer (AdamW vs Muon)

Default is **AdamW** (`--lr 3e-4` + cosine decay, grad clip 1.0). Feeling
dangerous? **Muon** orthogonalizes matrix gradients via Newton-Schulz and
usually reaches lower loss in fewer steps:

```bash
python3 -m src.ced_llm.train --data tinystories --steps 500 --ckpt ckpt.pt \
  --optimizer muon --muon-lr 0.02
```

Convention: 2D params (matrices) → Muon at `--muon-lr` (Muon scale, ~0.02);
biases/norms → AdamW at `--lr`. Zero new dependencies — `src/ced_llm/optim.py`
is a self-contained reimplementation. The optimizer choice is recorded in
every run's `config.json`, so comparisons stay honest.

---

## 🏛️ Architecture (the badass diagram)

```
                        ┌─────────────────────────────────────────────┐
                        │          ENCODER  (runs EXACTLY ONCE)        │
                        │                                             │
  tokens ──▶ embed+pos ──▶ CausalEncoder (N_enc pre-norm blocks)       │
                        │       │                                     │
                        │       ▼                                     │
                        │  h_enc  ──▶ kv_proj_k ──▶ K_g  ╗             │
                        │  h_enc  ──▶ kv_proj_v ──▶ V_g  ╝── MEMORY    │
                        └─────────────────────────────────────────────┘
                                   init_decode_cache() ▲ ONE call
                                                       │ k_glob/v_glob
                                                       │ (+ per-layer self-KV)
                        ┌──────────────────────────────┴──────────────┐
                        │        DECODER LOOP (runs N times)          │
                        │                                             │
  next token ──▶ CausalDecoder (N_dec layers):                        │
                 self-attn (causal + KV cache)  ──┐                    │
                 cross-attn (GLOBAL over K_g/V_g) ─┼─▶ logits ─▶ sample│
                        ▲                         │      │            │
                        └──── forward_step() ─────┘      ▼            │
                                                 next token ─────────┘
                                                 (encoder NOT re-run,
                                                  encoder_forward_count == 1)
```

**The three laws of CED:**

1. **Encode once.** `model.init_decode_cache(prompt)` runs the encoder a single
   time and projects `h_enc → (K_g, V_g)` a single time. (`ARCHITECTURE.md` has
   the full liturgy with per-step ASCII traces.)
2. **Reuse everything.** Every `model.forward_step(token, cache)` reuses the
   identical `k_glob`/`v_glob` tensors (object identity preserved — we check
   `data_ptr()`) plus per-layer self-attention KV caches.
3. **Prove it.** `model.encoder_forward_count` must be exactly `1` after any
   generation. The tests assert the delta `== 1`. Re-encode and the tests will
   find you.

**Loss convention (read this before touching train.py):** `forward` returns
logits aligned 1:1 with `input_ids`. The CALLER shifts:
`x_in = input_ids[:, :-1]`, `y = input_ids[:, 1:]`,
`loss = CE(logits, y, ignore_index=pad_token_id)`.

**Causality quirk (by design, not a bug):** the decoder cross-attends GLOBALLY
to every encoder position, so changing the last input token legitimately shifts
prefix logits via cross-attention. Full-logits prefix invariance does NOT hold.
What MUST hold (and is tested): encoder prefix states are invariant to
last-token changes, and decoder prefix outputs are invariant to future-hidden
changes with fixed globals. See `tests/test_causality.py`.

---

## 📊 Benchmarks

Measured on Apple M4, torch 2.14, `benchmarks/speed.py` (tok/s, best of runs;
MPS tiny-shape timing is noisy ±15–30%, CPU is stable):

| Workload | Command | Apple M4 result |
|---|---|---|
| Full test suite (35 tests) | `pytest -q` | ~2.1s, 35/35 green |
| Train smoke (60 toy steps) | `python3 -m src.ced_llm.train --smoke` | loss `6.25 → 4.83`, asserts final < initial |
| Generate smoke (16 tokens) | `python3 -m src.ced_llm.generate --smoke` | `encoder_forwards=1`, proves KV-reuse |
| Train, MPS (d=128, 2+2, B=8, T=128) | `benchmarks/speed.py --device mps` | **~89k tok/s** (11.4 ms/step) |
| Train, CPU (same shape) | `benchmarks/speed.py --device cpu` | ~61k tok/s (16.6 ms/step) |
| Generate, MPS (same shape) | `benchmarks/speed.py --device mps` | **~470 tok/s** (2.1 ms/token) |
| Generate, MPS batched ×8 | `benchmarks/speed.py --device mps --gen-loop forge --gen-batch-size 8` | **~4000 tok/s** (~8.5× single) |
| Generate, CPU (same shape) | `benchmarks/speed.py --device cpu` | ~4700 tok/s (0.21 ms/token) |
| Train smoke, MPS stacked flags | `train --smoke --device mps --fused --grad-clip 0 --loss-sync-every 8` | wall 1.81s → **1.51s (+20%)** |

Speed levers (all opt-in except the freebies): fused QKV + static KV cache +
mask fast paths (inference, exact parity), `--fused`/`--foreach` AdamW,
`--grad-clip 0`, `--loss-sync-every 8`, `--device mps`, batch decode.
Defaults (CPU fp32) are bit-identical: smoke canary `6.2551 → 4.8283` before
and after. Raw JSON: `benchmarks/baseline_*` (pre-tune) vs `benchmarks/tuned_*`.

**Reproduce on your machine:**

```bash
pytest -q
time python3 -m src.ced_llm.train --smoke
time python3 -m src.ced_llm.generate --smoke
python3 benchmarks/speed.py --device auto
```

---

## 📏 Eval (perplexity + samples)

`src/ced_llm/eval.py` scores any checkpoint (or a tiny random model) and shows
its work: mean loss + perplexity, then 3 generation samples — each proving
`encoder_forwards=1` on the way out.

```bash
python3 -m src.ced_llm.eval --smoke          # tiny random model, synthetic data, <60s CPU
python3 -m src.ced_llm.eval --ckpt ckpt.pt   # eval YOUR trained checkpoint
python3 -m src.ced_llm.eval --data tinystories --max-examples 200 --max-new 32
```

What it does:

1. **Perplexity.** Builds an eval loader (`--seq-len`, `--batch-size`,
   `--max-examples`) from synthetic fallback texts, or a TinyStories sample
   with `--data tinystories` (tries HF download when online, falls back to
   synthetic offline — never crashes, never hangs the suite).
2. **3 generation samples.** Fixed prompts (`brave little bunny` /
   `forgetful dragon` / `talking frog`, seeds 7/42/1234): greedy short takes
   in `--smoke`, delightful sampled stories + magic scores otherwise.

Handy flags: `--ckpt` (ckpts saved by `train --ckpt`), `--seq-len`,
`--batch-size`, `--max-examples`, `--max-new`, `--seed`, `--device`.
`--seq-len` is auto-capped to the model's `max_seq_len` so eval never reports
a bogus loss on rows the model can't score.

New here and want the whole arc in one command? `python3
examples/quickstart.py` runs train-smoke, saves the toy checkpoint, then
generates 2 prompts from it end to end.

---

## 📈 Tracking (every run leaves footprints)

Training (and optionally eval) logs to a timestamped run directory — no
account, no server, no network. Just files:

```bash
python3 -m src.ced_llm.train --data tinystories --steps 500 --ckpt ckpt.pt
# → runs/run-20260912-103000/{config.json, metrics.jsonl, summary.json}

python3 -m src.ced_llm.train --smoke --run-name debug1   # custom run name
python3 -m src.ced_llm.train --steps 500 --no-track      # disable logging
python3 -m src.ced_llm.eval --ckpt ckpt.pt --run-dir runs --run-name eval1
```

* `config.json` — hyperparams + CLI args + git hash (best-effort).
* `metrics.jsonl` — one JSON object per logged step (`--log-every N`,
  default 20): `{"step": 20, "loss": 4.21, "lr": 0.00029}`.
* `summary.json` — final aggregates (init/final loss, eval ppl, ckpt path).

Compare runs with zero dependencies:

```bash
python3 -c "
import json
for run in ['runs/a', 'runs/b']:
    rows = [json.loads(l) for l in open(run + '/metrics.jsonl')]
    print(run, 'steps:', len(rows), 'last loss:', round(rows[-1].get('loss', 0), 4))
"
```

Want live curves? `pip install tensorboard`, add `--tensorboard` to any
tracked command, then `tensorboard --logdir runs`. Missing package only
warns — the JSONL logs always work.

---

## 🗺️ File map (know thy dungeon)

```
src/ced_llm/config.py     # CEDConfig dataclass + validate() (hyperparams, max_seq_len, pad id)
src/ced_llm/attention.py  # Dense CausalSelfAttention + GlobalCrossAttention (SDPA, no MoE)
src/ced_llm/encoder.py    # CausalEncoder: stack of pre-norm blocks
src/ced_llm/decoder.py    # CausalDecoder: self-attn + global cross-attn + FFN, KV-cache stepping
src/ced_llm/model.py      # CEDForLM: embed ONCE, encode_once, init_decode_cache, forward_step
src/ced_llm/data.py       # SimpleTokenizer, TinyStories loader + synthetic fallback, encode_pack
src/ced_llm/train.py      # compute_loss / train_one_epoch / evaluate / CLI (--smoke toy run, --run-dir tracking)
src/ced_llm/optim.py      # Muon (Newton-Schulz) + AdamW hybrid factory, torch-only, zero new deps
src/ced_llm/tracking.py   # RunTracker: offline JSONL runs + optional TensorBoard mirror
src/ced_llm/eval.py       # perplexity + 3 samples CLI (--smoke, --ckpt, --run-dir)
src/ced_llm/generate.py   # generate_greedy (cache-once + step loop) / sampling spells / CLI
benchmarks/speed.py       # tok/s harness: train+gen on cpu/mps, fp32/fp16, batch decode
tests/test_causality.py   # Encoder + decoder self-attn causality proofs
tests/test_data.py        # Tokenizer round-trip, pack, batch shapes, offline fallback
tests/test_generate.py    # Greedy determinism, KV-reuse counter, length-or-EOS-stop
tests/test_kv_reuse.py    # Full-vs-incremental parity, ptr stability, global dependence
tests/test_training.py    # Toy overfit, grad flow to encoder, padding invariance
tests/test_edgecases_docs.py  # DOC-BOSS: empty prompt, single token, truncation, temp=0 (7 tests)
tests/test_tracking.py      # RunTracker layout, disabled mode, TB fallback, smoke end-to-end
tests/test_optim.py         # Newton-Schulz band, Muon descent, hybrid routing, muon smoke
ARCHITECTURE.md           # Deep dive: encoder-once + KV-reuse with ASCII traces
CONTRIBUTING.md           # How to contribute without summoning demons
requirements.txt          # torch+pytest required; datasets/tiktoken optional (commented)
```

---

## 🧪 Testing (the shield wall)

```bash
pytest -q                          # everything, ~2s
pytest tests/test_edgecases_docs.py -q   # just the DOC-BOSS edge cases
pytest tests/test_kv_reuse.py -q         # just the cache paranoia
```

| Test file | What it proves | Catches |
|---|---|---|
| `test_causality.py` (2) | Encoder/decoder prefix invariance | Bidirectional-mask slop |
| `test_data.py` (4) | Round-trip, pack+BOS/EOS, batch shapes, offline fallback | Tokenizer/loss-mask slop |
| `test_generate.py` (3) | Determinism, encoder x1, length-or-EOS | RNG leaks, re-encode loops, truncation bugs |
| `test_kv_reuse.py` (3) | Parity ≤1e-4, ptr stability, global dependence | Cache bugs, recompute-cheating, dead cross-attn |
| `test_training.py` (3) | Overfit 50%+80%, grad flow, pad invariance | Broken loss/shift/detached encoder |
| `test_edgecases_docs.py` (7) | Empty→BOS, single-token, truncation contract, temp=0 | Demo-day crashes |

---

## 🔧 Troubleshooting (the resurrection scroll)

**`ImportError: torch>=2.0 is required`**
→ You skipped quickstart step 1. `pip install -r requirements.txt`. No torch, no thunder.

**`ValueError: sequence length 64 exceeds max_seq_len 32`**
→ Working as designed! The model FAILS LOUDLY past `max_seq_len` instead of
silently hallucinating. Fix: keep the LAST `max_seq_len` tokens
(`ids[:, -max_seq_len:]`), or lower `--seq-len` / raise `max_seq_len` in config.
`encode_pack` already truncates training rows to `seq_len` with BOS/EOS kept.

**Empty prompt generates something weird / `""` → crash?**
→ Should NOT crash: `generate_greedy` seeds empty prompts with BOS and returns
`[BOS, ...tokens]`. If you call the model directly with a 0-length tensor,
that's on you — seed with BOS first. (`test_empty_prompt_bos_seed_no_crash`.)

**Output stops early (shorter than prompt + max_new)?**
→ Only legal if it ends with EOS. Early-stop-without-EOS is a bug and the tests
fail loudly on it. Pass `--no-eos-stop` (if your CLI has it) to force full length.

**Different runs give different text with temperature=0?**
→ Impossible unless dropout is on during eval or something reseeds. Greedy is
pure argmax and ignores top-k/top-p. Check `model.eval()` and see
`test_temperature_zero_determinism`. With temperature>0, pass `seed=N` for
reproducibility (`test_seeded_sampling_is_reproducible`).

**Offline / no internet / `datasets` missing?**
→ Fine. The loader falls back to synthetic TinyStories-style texts and
`SimpleTokenizer` needs zero downloads. `test_tinystories_loader_fallback_offline`
proves it with imports force-broken.

**Loss is NaN / not decreasing in smoke?**
→ Smoke uses fixed seeds; `train --smoke` asserts final < initial. If you
changed hyperparams, check LR (default smoke is tuned), then check the shift
convention (`x_in=tok[:, :-1]`, `y=tok[:, 1:]`). Then blame turbo. (Kidding.
Mostly.)

**`tiktoken` / `datasets` pip errors?**
→ They're OPTIONAL. Comment them out; everything still runs. `requirements.txt`
says so in the comments. Docs said it twice. Docs are never wrong.

---

## 🤝 Contributing

Read `CONTRIBUTING.md` (short, funny, strict about scope and tests), then read
`ARCHITECTURE.md` (long, funny, strict about the encoder running once). PRs that
break `pytest -q` will be mocked in `THUNDERDOME_DOCS.md`. This is both a warning
and an invitation.

## 📜 Thunderdome scrolls

- `THUNDERDOME_TURBO.md` — the speed demon's manifesto (train/model/attention).
- `THUNDERDOME_CHAOS.md` — the story wizard's spellbook (generate.py).
- `THUNDERDOME_DOCS.md` — the lore master's chronicle (docs + tests, i.e. this
  README's muscle). Double trash-talk inside. Enter at own risk.

---
*Built offline-first for TinyStories. Encoder runs once. Docs run forever.* ⚡📚
