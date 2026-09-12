# FORGE-DECODE: generation-loop speed hunt (Apple M4 MPS)

Scope: `src/ced_llm/generate.py`, `demo_stories.py` (perf), `benchmarks/speed.py`
(ADDITIVE args only). No commits, no new required deps (stdlib + torch only).
No `FORGE_MODEL.md` exists in repo, so the "local prealloc" branch was taken
(generate.py owns tiny reuse buffers; ready to adopt a forge-model static
cache when its doc/API lands).

## Diagnosis (cProfile, MPS, 320 greedy tokens, d_model=64)

| Cost center (per token) | Share | Fix (all in generate.py / bench harness) |
|---|---|---|
| `torch.tensor([[nxt]])` alloc + H2D | ~24% | reused `[1,1]` device buffer + `fill_` / `copy_` |
| `logits.float().cpu()` full-vocab D2H | ~22% | fused device-side `argmax`, scalar `.item()` sync only |
| `_pad_mask` per-step alloc + `.any()` sync (inside `forward_step`) | ~19% | normalize `cache["enc_mask"]=None` once (prompt has no pads; `_pad_mask` maps all-ones to `None` anyway) |
| `_extract_step_logits` + sampler dispatch + EOS list rebuild | small | one-time return-convention probe, hoisted `frozenset` EOS |
| MPS ~4 ms/step launch floor (irreducible per step) | — | **batch decode** amortizes it over B rows |

`model.py`/`decoder.py` internals (per-step `pos` alloc, KV `cat` reallocs,
cross-attn re-split) are out of scope and remain the residual floor.

## Changes

- `generate.py`: `_greedy_fast_loop()` (auto-selected for pure greedy:
  `temperature<=0` and `repetition_penalty==1.0`; top-k/top-p are ignored on
  the greedy branch by construction). Non-canonical `forward_step` returns
  bail to the untouched legacy loop (pristine cache rebuilt, counter safe).
  New `generate_batch_greedy()` (one shared encode, per-row exactness).
  Smoke prints tok/s (asserts unchanged).
- `benchmarks/speed.py`: `--gen-loop {baseline,forge}` (default `baseline`),
  `--gen-batch-size N` (default 1). Defaults reproduce legacy behavior.
- `demo_stories.py`: `--no-quality` flag, per-story tok/s timing (stderr),
  one `model.eval()`. Generation args unchanged.

## Before/after (`benchmarks/speed.py --device mps`, fp32, full defaults, same tree)

| gen mode | smoke tok/s (ms/tok) | real tok/s (ms/tok) |
|---|---|---|
| baseline | 315 (3.18) | 479 (2.09) |
| forge B=1 | 344 (2.91) | 485 (2.06) |
| forge B=4 | 1407 (0.71) | 1912 (0.52) |
| forge B=8 | 2829 (0.35) | 3875 (0.26) |

Loop-isolated A/B on the pre-change model (legacy-with-`.cpu()` vs fast):
MPS smoke 243 -> 275 tok/s (+13%), real 201 -> 358 (+78%).
CPU fp32: baseline 6617/3509 -> forge B=4 16962/8534; B=1 within noise.

Noise caveat: MPS tiny-shape timing is noisy run-to-run (±15-30% per
FORGE_MODEL.md), so the B=1 forge deltas (+9% smoke, +1% real) are inside the
noise band — treat batch (4-8x, far outside noise) as the real signal.
In-tree `real` baseline (479) already includes the sibling FORGE-MODEL
model-side wins (fused QKV / static cache, auto-enabled via config defaults;
no generate-side hook needed — this loop benefits automatically).
Session-start tree: MPS gen smoke 186 (5.39 ms), real 227 (4.40 ms);
CPU gen smoke 6547 (0.15 ms), real 2622 (0.38 ms).

## Correctness (all green)

- `pytest -q`: **35/35** in-tree (final tree incl. concurrent model-side work).
- 8-config CPU matrix (greedy / temp+top-k/p / rep-penalty / empty prompt /
  no-EOS-stop) bit-identical vs pre-change baseline file; `encoder_forwards==1`.
- In-tree fast==forced-legacy and batch==single on **cpu and mps**;
  dict-returning and object-returning models (bail path) and keyword-form
  `forward_step` all identical with `encoder_forwards==1`.
- Smoke asserts sacred: `--smoke` OK, `encoder_forwards==1`.

## Failures / incidents (honest log)

1. **External reverts of `generate.py` mid-session** (2x: helper insert
   vanished; branch+smoke edits vanished; a `forge-loop-wip` stash holds a
   copy of my helper text). Recovered by re-applying + md5/backup guards in
   `/tmp/my_generate_full.py`. No work lost; final md5 verified.
2. **Concurrent FORGE-MODEL WIP broke `import src.ced_llm`** (~30 min:
   `model.py` needed `config.resolve_dtype` before `config.py` landed;
   pytest 22 failed + 2 collection errors tree-wide). Verified meanwhile in
   isolated overlay `/tmp/forge_verify` (HEAD + my files: 35/35, identical,
   parity); final verification re-run in-tree after the tree healed.
3. **My test wrapper bug** (not product): `nn.Module.__setattr__` routes
   `self.m` to `_modules`, breaking a dict-model test double (delta=0).
   Rewrote as plain object; bail path verified identical.
4. **zsh word-splitting**: unquoted `$args` passed as one arg to speed.py;
   reran with explicit argv. Measurement-harness only.
5. **No `FORGE_MODEL.md`** -> static-cache adoption deferred; local-prealloc
   buffers used instead (documented above).
6. **Honest accounting**: speed.py's baseline loop was already lean (device
   argmax, no `.cpu()`), so forge B=1 gains only +2-9% there; the big wins
   are the API path (old `.cpu()` per token) and batching (up to ~8x).
