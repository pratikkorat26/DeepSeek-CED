# FORGE-LOOP — train-step speed hunt (Apple M4 MPS)

Scope: `src/ced_llm/train.py` loop bodies, `optim.py`, `tracking.py` overhead,
`data.py` loader options (backward-compatible APIs only). No new required deps.
No commits. Default CPU fp32 numerics bit-identical (flags or pure-overhead
removals only).

Machine: Apple M4, torch 2.8.0, MPS available+built. All tok/s are
`benchmarks/speed.py` or loop-mirror microbenches, best-of-2/3, same process
conditions noted. MPS timing noise is ±5–10% run-to-run; only deltas well
outside noise are claimed as wins.

## Baselines (before)

Task brief baselines (fp32): mps train smoke 21783 tok/s, real 84029 tok/s;
cpu train smoke 59855, real 61196.

Re-measured on this machine (`benchmarks/speed.py --steps 60`, HEAD without
FORGE-LOOP changes, clean worktree `/private/tmp/forge_base`):

| harness | smoke train | real train |
|---|---|---|
| `speed.py --device mps --dtype fp32` | ~17350 tok/s (7.15 ms/step) | ~74859 tok/s (13.57 ms/step) |
| `speed.py --device mps --dtype fp16` | ~14564 tok/s (8.51 ms/step) | ~69402 tok/s (14.64 ms/step) |
| `speed.py --device cpu --dtype fp32` | ~48795 tok/s (2.54 ms/step) | ~44978 tok/s (22.59 ms/step) |

`pytest -q`: 35/35. `train --smoke --seed 0` (cpu fp32): init 6.2551 →
final 4.8283, avg step 5.6046 (this exact triple is the bit-identical canary).

Note: `benchmarks/speed.py` builds its own vanilla-AdamW loop and bypasses
`train_one_epoch` / `build_optimizer` / `RunTracker`, so it is the
before/after regression gate (must stay neutral), NOT where loop wins show up.
Loop wins are measured with train-loop-mirror microbenches below + timed
`train --smoke` runs.

## Attempts (win AND fail)

Train-loop-mirror bench = `_fixed_toy_loader` + `_make_model` + `compute_loss`
+ `zero_grad/backward/clip/step`, MPS-synchronized timers. Smoke shape =
B4/T32/d64/1+1 layers; real shape = B8/T128/d128/2+2 layers.

| # | attempt | result (MPS best-of-3) | verdict |
|---|---|---|---|
| 1 | AdamW `fused=True` (smoke) | 19304 → 22554 tok/s (**+17–20%**); real 86340 → 88252 (**+2–3%**) | **WIN** — opt-in `--fused` |
| 2 | AdamW `foreach=True` (smoke) | 19338 → 20509 (**+6%**); real 86340 → 85019 (~0%, noise) | **HALF-WIN** — opt-in `--foreach`, fused dominates |
| 3 | `capturable=True` | `AssertionError: ... must be on ['cuda','xpu','hpu','privateuseone']` | **FAIL** — intentionally not exposed (MPS unsupported) |
| 4 | fp16 autocast alone (intel said nothing) | smoke 17350 → 14564 (**-16%**); real 74859 → 69402 (**-7%**); loop-mirror fp16 15062 → 12612 (**-16%**) | **FAIL (confirmed)** — still shipped as `--dtype fp16` flag, never default |
| 5 | bf16 autocast alone | smoke → 14407 (**-17%**); loop-mirror 15062 → ~12400 (**-18%**) | **FAIL (confirmed)** — shipped as `--dtype bf16` flag only |
| 6 | `torch.compile` (inductor, tiny shapes) | first 5 steps **5.75 s** (eager ~0.04 s); 30 steps 6.6 s; dynamo `recompile_limit (8)` hit on `model.encoder_forward_count` int attr | **FAIL on tiny (confirmed)** — shipped as opt-in `--compile` (off by default) with `allow_unspec_int_on_nn_module` guard + eager fallback |
| 7 | per-step `.item()` sync removal | sync 15322 → nosync 16194 (**+6–7%** smoke) | **WIN** — (a) pure-overhead dedup (3×→1× `.item()` per logged/printed step, identical value) always on; (b) `--loss-sync-every K` tensor-accumulate flag for the rest |
| 8 | duplicate `.item()` in smoke/non-smoke loops (`total` + `tracker.log` + `print` each called `float(loss.item())`) | 1 MPS sync instead of 3 on logged steps | **WIN** — always on, bit-identical (`loss_val` reused) |
| 9 | skip `clip_grad_norm_` (`--grad-clip 0`) | fused smoke 17062 → 21030 (**+23%** on top of fused, **+40%** over default); real fused 79697 → 84000 (**+8%** over default) | **WIN via flag** — default stays 1.0 (identical); `<=0` disables |
| 10 | fused + noclip + nosync combined | smoke default 15062 → **23016 (+53%)** | **WIN** (stacked flags) |
| 11 | Muon NS steps 5 → 3 → 2 | 49.3 → 39.5 → 34.7 ms/step (**-20%** of Muon time) but Muon still **~6–7× slower** than AdamW (6.35 ms/step) on smoke | **PARTIAL** — `--muon-ns-steps` flag helps Muon users; not a default win; default stays 5 (identical) |
| 12 | tracker: persistent handle + `buffer_rows` | 200 logs: 1.1 ms → 0.7 ms (**-36%** tracking overhead ≈ 0.002 ms/step) | **WIN (tiny, free)** — default `buffer_rows=1` keeps file bytes + read-your-writes identical; `--tracker-buffer N` batches |
| 13 | DataLoader `num_workers/persistent/prefetch/pin` | tinystories tiny-epoch: nw=0 3.4 ms/epoch; nw=2 fragile under macOS spawn from stdin + slower for tiny batches | **NEUTRAL/FAIL for tiny** — shipped backward-compat (`--num-workers/--prefetch-factor/--pin-memory`, defaults 0/None/off = identical) for large-scale use |
| 14 | `--device mps` wiring (train CLI was hardcoded `cpu`) | smoke cpu 1.87 s wall → mps 2.07 s wall (MPS **slower** on smoke); real-shape MPS faster than CPU (13.4 vs 22.6 ms/step) | **WIN for real shapes** — `--device {cpu,mps,cuda,auto}`, default `cpu` (identical) |
| 15 | TB `flush()` every `log()` | batched with file flush (write-through default flushes, buffered defers to drain/`flush()`/`close()`) | **WIN (tiny)** — default behavior identical |

## What shipped

- `src/ced_llm/optim.py`: `build_optimizer(..., foreach=None, fused=None,
  ns_steps=5)` + `_maybe_adamw()` with MPS-safe fallback (fused rejected →
  retry without fused → vanilla). Defaults (`None`/`5`) construct exactly
  `torch.optim.AdamW(params, lr)` / `Muon(..., ns_steps=5)` as before.
- `src/ced_llm/train.py`:
  - `--device {cpu,mps,cuda,auto}` (default `cpu`), `--dtype {fp32,fp16,bf16}`
    (default `fp32` = nullcontext, identical), `--foreach`, `--fused`,
    `--muon-ns-steps 5`, `--grad-clip 1.0` (`<=0` disables),
    `--loss-sync-every 1` (tensor-accumulate when >1),
    `--tracker-buffer 1`, `--compile` (off, guarded), `--num-workers 0`,
    `--prefetch-factor`, `--pin-memory`.
  - `train_one_epoch(..., dtype="fp32", loss_sync_every=1)` — defaults take
    the exact old code path (per-step `float(loss.item())`, clip 1.0, no
    autocast).
  - Smoke + non-smoke loops: single `loss_val = float(loss.item())` reused
    for avg + tracker + print; autocast-wrapped forward; compile hook;
    evaluate passes `dtype`.
  - `_new_optimizer` forwards `--fused/--foreach/--muon-ns-steps`;
    `_new_tracker` forwards `--tracker-buffer`.
- `src/ced_llm/tracking.py`: `RunTracker(..., buffer_rows=1)` — persistent
  append handle (saves open/close per `log()`), write-through default keeps
  bytes + `load_metrics()` visibility identical; `flush()`/`close()` drain;
  TB flush batched with file flush.
- `src/ced_llm/data.py`: `get_dataloader(..., num_workers=0,
  persistent_workers=False, prefetch_factor=None, pin_memory=False)` —
  `prefetch_factor` only forwarded when `num_workers>0` (torch raises
  otherwise); all defaults reproduce the old `DataLoader(ds, batch_size,
  shuffle)` exactly. `eval.py` call sites unaffected (keyword args).

## Final numbers (after, `/private/tmp/forge_clean` = HEAD + these 4 files)

- `speed.py --device mps --steps 60` (regression gate, harness bypasses loop):
  smoke ~17303–17522, real ~75517–76615 → **neutral vs base within noise**
  (base smoke 16708–17799, real 63726–76194). No regression.
- Loop-mirror best-of-3: smoke default 15062 → fused 17062 → fused+noclip
  21030 → fused+noclip+nosync **23016 (+53%)**; real default 77742 → fused
  79697 (**+2.5%**) → fused+noclip **84000 (+8%)**.
- Recommended: `train --device mps --fused --grad-clip 0` (+40% smoke,
  +8% real); add `--loss-sync-every 8` when 1-ulp avg freedom is acceptable.
- `pytest -q`: **35/35** in the isolated clean tree; `eval --smoke` OK
  (3 samples, `encoder_forwards==1`).
- Bit-identical canary: `--smoke --seed 0` cpu fp32 → init **6.2551**,
  final **4.8283**, avg **5.6046** before AND after.

## Caveats

- Main working tree currently contains a concurrent out-of-scope `decoder.py`
  WIP that breaks `pytest` there (`NameError: _sdpa`); all green runs above
  are from the isolated HEAD+4-files worktree. My diff touches only
  `train.py`, `optim.py`, `tracking.py`, `data.py` (+ this doc).
- MPS wall-clock noise is large; all claimed wins are best-of-3 deltas well
  above the ±5–10% noise band, and fails (fp16/bf16/compile) were re-measured
  to confirm the intel rather than assumed.
