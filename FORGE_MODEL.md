# FORGE-MODEL — model-kernel speed hunt (Apple M4 MPS)

Scope: `attention.py`, `encoder.py`, `decoder.py`, `model.py`, `config.py` only.
Never touched: `train.py`, `generate.py`, `optim.py`. No new REQUIRED deps
(only `torch` + stdlib `os`/`typing`). No git commits. `pytest -q`: 35/35 green.
Default numerics preserved (fp32; parity tests green, see §5).

## 1. Before / after (`benchmarks/speed.py --device mps`, fp32, torch 2.8.0)

Committed file baselines (torch 2.14.0, for reference):
- `baseline_mps_fp32.json`: train_smoke 21423 tok/s (5.79ms), train_real 89041
  (11.41ms), gen_smoke 395 (2.53ms), gen_real 340 (2.94ms.
- `baseline_cpu_fp32.json`: train_smoke 58780 (2.11ms), train_real 59828
  (16.98ms), gen_smoke 8713 (0.11ms), gen_real 4721 (0.21ms).

Fresh BEFORE (2026-09-12, torch 2.8.0, worktree with concurrent
train/optim/tracking edits, WITHOUT model-kernel edits):
- MPS: train_smoke 17257 (7.19ms), train_real 76030 (13.36ms),
  gen_smoke 256 (3.91ms), gen_real 231 (4.33ms).
- CPU: train_smoke 43676 (2.84ms), train_real 42097 (24.13ms),
  gen_smoke 6736 (0.15ms), gen_real 3590 (0.28ms).

AFTER (same machine, torch 2.8.0, WITH model-kernel edits; 4 runs to show MPS
variance — MPS tiny-shape timing is noisy ±15-30% run-to-run):
- Run A: MPS train_smoke 15287 (8.11), train_real 66641 (15.25),
  gen_smoke 293 (3.41, +14% tok/s), gen_real 419 (2.38, +81%);
  CPU train 43764/41790, gen_smoke 5448, gen_real 3182.
- Run B: MPS train 16865 (7.35)/68293 (14.88), gen 272 (3.68)/400 (2.50).
- Run C: MPS train 17489 (7.09)/59248 (17.15), gen 239 (4.18)/444 (2.25).
- Run D: MPS train 17545 (7.07)/75434 (13.47), gen 260 (3.85)/341 (2.93).

Summary: MPS `gen_real` (d=128, 2+2 layers) consistently wins (+48% to +92%
tok/s across runs). MPS `gen_smoke` (d=64, 1+1) is within noise (239-293 vs
256 before; Run A +14%). Train is within noise / slightly slower on some runs
(see Failures). CPU gen regresses ~11-19% on tiny shapes (strided static views
vs contiguous cat; CPU is compute-bound, MPS is overhead-bound — trade-off
accepted for MPS target, parity still exact).

Steady-state micro (warmed MPS kernels, 1+1 / d=64, N=32, same episode):
- cat+separate (original semantics): 3.619 ms/tok.
- cat+fused QKV only: 1.085 ms/tok (3.3×).
- static+separate only: 0.370 ms/tok (9.8×).
- static+fused (shipped default): 0.357 ms/tok (10.1×).
- Warmed full sweep (max_len=128): N=16 0.409, N=32 0.368, N=64 0.352 ms/tok.
`speed.py` default warmup (8 toks) does NOT warm all lengths 1..64, so its
numbers include MPS kernel compilations and read 3-4 ms/tok cold vs 0.35 ms/tok
warmed. Report both; do not compare cold to warmed.

## 2. SDPA backend selection on MPS (measured, then made zero-overhead)

Probe (MPS, torch 2.8.0, `torch.nn.attention.sdpa_kernel`):
- Tiny causal [2,4,32,16]: MATH 0.0429, FLASH 0.0398, EFFICIENT 0.0374,
  CUDNN 0.0371, DEFAULT 0.0399 ms (all within noise).
- Decode [1,4,1,16]×[1,4,8,16]: MATH 0.0252, FLASH 0.0256, EFFICIENT 0.0246,
  CUDNN 0.0252, DEFAULT 0.0250 ms.
Conclusion: on MPS all backends lower to the same kernel; SDPA itself is <1%
of the ~3.9 ms/tok decode cost → decode is OVERHEAD-bound (dispatch/alloc/sync),
not math-bound.

Shipped: `attention.preferred_sdpa_backend(device)` + `_sdpa()` wrapper.
Default (`CED_SDPA_BACKEND=auto`, the default) calls
`F.scaled_dot_product_attention` DIRECTLY with no context (zero per-forward
overhead, bit-identical, MATH-equivalent on MPS). Explicit
`CED_SDPA_BACKEND=math|flash|efficient|cudnn` forces that backend via
`sdpa_kernel` context with fallback to direct on failure. Per-call override
via `_backend=` kwarg for experiments. No per-step context when auto.

## 3. Mask fast paths

Kept: `_is_no_pad_mask(None)` → True (no sync), `is_causal=True` /
`attn_mask=None` fused kernels, no `[B,1,T,T]` alloc on no-pad path.
Added:
- `model._pad_mask_cached()`: `id(tensor)`-keyed (strong ref, anti-ABA,
  cap 32) sharing of `attention_mask==0` conversions. Fixed toy loaders reuse
  the same 8 mask objects → after 8 forwards, hits avoid per-forward `==0`
  alloc + `.any()` D2H sync. Semantics identical to `_pad_mask` (None vs bool).
- `model.init_decode_cache` precomputes `glob_mask` once; `forward_step`
  reuses `cache["glob_mask"]` (no per-step recompute+sync). `history_key_mask`
  stays `None` when `attention_mask is None` (bench path → no alloc/sync).
- `causal_attend_mask` unchanged (still uses `_get_causal_ok` cache); padded
  path still materializes fused mask (rare, correctness first).

## 4. Fused QKV projections (3 launches → 1, bit-identical)

`CausalSelfAttention` (`attention.py`) + `DecoderLayer._qkv_self`
(`decoder.py`): single `F.linear(x, cat([Wq,Wk,Wv]))` then `chunk(3)`.
- State dict UNCHANGED (`q_proj/k_proj/v_proj` still separate `nn.Linear`
  bias=False) → old checkpoints load exactly.
- Train/grad-enabled: fresh `cat` every forward (autograd stays attached;
  grad flow verified by `test_gradient_flow_to_encoder`).
- Inference (`eval` + `no_grad`): cached stacked `[3D,D]` weight keyed by
  `(ptr, _version, device, dtype)` per triple; reused until weights change
  (optimizer `copy_`/`add_` bumps `_version`; `load_state_dict` copy also
  bumps). `use_fused_qkv` flag (default True, from `config.use_fused_qkv`;
  `CausalSelfAttention(..., fused_qkv=True)` 4th arg defaults True so old
  `CausalSelfAttention(d,nhead,drop)` calls keep working).
- Measured parity: fused vs separate full-forward maxdiff 0.0 (CPU and MPS).

## 5. STATIC preallocated KV decode cache (fill-in-place, exact-parity proof)

`model.init_decode_cache` adds (backward-compat extras, old keys untouched):
`self_k_buf/self_v_buf` ([None]×layers, lazy `[B,H,S,Dh]` via `torch.empty`),
`self_decode_len=0`, `self_static_capacity=max_seq_len`, `glob_mask`,
`batch`, `_pos_buf` (`[B,1]` long, `fill_` per step), `use_static`
(from `config.use_static_cache`, default True).

`DecoderLayer._self_step_static` + `CausalDecoder.forward_static` +
`model.forward_step` static branch:
- `narrow(2,cur_len,1).copy_(k_new/v_new)` (O(1), no `cat` realloc/O(T) copy);
  attention uses `narrow(2,0,cur_len+1)` views (no copy).
- Buffers lazily allocated with `k_new` dtype/device → correct under fp32
  default and under MPS autocast fp16/bf16 (when explicitly requested).
- Mismatch (batch/device/dtype/capacity changed) with `cur_len>0` raises
  `ValueError`; with `cur_len==0` reallocates (fresh episode). `cur_len+1 >
  capacity` raises `decode position exceeds max_seq_len` (same message).
- Compat: after each static step `cache["self_k_list/v_list"]` are updated to
  the prefix views (shapes `[B,H,cur_len,Dh]`, values identical) and
  `self_decode_len` bumped, so old readers see correct shapes/values.
  Old caches (no static keys) or `use_static=False` fall back to legacy `cat`
  path (same messages). Static failures fall back to `cat` (correctness over
  speed; never breaks parity).

Parity proof (tiny cfg 128/32/1+1/2 heads, CPU and MPS):
- full vs incremental (static default): maxdiff 8.94e-08, `allclose`
  (atol 1e-4) True on CPU and MPS.
- static vs cat incremental: maxdiff 0.0 (exact).
- full vs cat incremental: 8.94e-08 (same pre-existing full-vs-step delta;
  static adds zero extra error).
- `k_glob/v_glob` `data_ptr` stable across 4 steps; `encoder_forward_count==1`.
- `pytest -q`: 35/35 (includes `test_kv_reuse` parity, single-encode/ptr,
  `test_generate`, `test_causality`, `test_training` padding invariance
  diff 4.7e-07 <1e-4).
“Bit-identical” = default fp32 path passes all parity/invariance thresholds
with zero additional error from fused/static (fused 0.0, static-vs-cat 0.0);
the 8.9e-08 full-vs-step residual is pre-existing kernel-order noise, not
introduced here.

## 6. Hoisted `forward_step` per-token validation

- `max_seq_len` hoisted to local; `cur_len` from cached int
  `self_decode_len` (no `first_k.size(2)` tensor op on hot path; old-cache
  fallback preserved).
- `pos`: reused `_pos_buf.fill_(cur_len)` (no `torch.tensor([[cur_len]])`
  alloc+`expand` per token; fallback allocates once then caches).
- `glob_mask`: reused from init (no per-step `_pad_mask` alloc+sync).
- `tok_emb/pos_emb` hoisted to locals; `layer_caches` list build kept only
  for legacy `cat` path (static path passes bufs directly).
- Validation messages preserved: `next_ids must have shape [B, 1]`,
  `batch size of next_ids and cache must match`,
  `decode position exceeds max_seq_len`, history-mask shape check.

## 7. Dtype policy `resolve_dtype(device, flag)` defaulting to fp32

`config.resolve_dtype(device="cpu", flag="fp32") -> torch.dtype`:
pass-through for `torch.dtype`, `None`→fp32, case-insensitive
`fp32/fp16/bf16/auto` (+aliases), unknown→fp32 (never crashes).
`auto` conservatively returns fp32 (explicit opt-in required for fp16/bf16).
`CEDConfig` gains `dtype="fp32"`, `use_fused_qkv=True`,
`use_static_cache=True`, `sdpa_backend="auto"` (all defaulted so old
`CEDConfig(vocab,...)` constructions and `validate()` keep working; new fields
validated leniently). `CEDForLM` validates/stores `_resolved_dtype` at init
(default fp32). Benchmarks keep explicit `--dtype` + MPS autocast; model
buffers follow runtime `k_new` dtype (correct under autocast).

## 8. Failures / non-wins (honest)

- Train tok/s: no consistent win; runs vary ±15-27% on MPS tiny shapes
  (e.g. real 59248 vs 75434 across identical runs). Fused A/B in isolation:
  5.19→5.06 ms/step (-2.5%, within noise). Concurrent `train.py` edits by
  another lane also move train numbers; do not attribute train deltas to this
  lane alone. No train correctness regression (overfit/grad/pad tests green).
- CPU gen tiny shapes: -11% to -19% (static strided views help MPS
  overhead-bound decode but cost CPU contiguous fast path). Accepted for MPS
  target; parity exact, tests green.
- MPS smoke gen: within noise (239-293 vs 256 before); real gen wins clearly.
  Blame short warmup (8) + per-length MPS compilation; warmed steady-state is
  0.35 ms/tok (10× over cat+separate 3.62 ms/tok).
- Full-vs-step residual 8.9e-08 (not 0.0) — pre-existing, unchanged by this
  lane (static-vs-cat 0.0 proves it).
- `torch.compile` / fp16 / flash-backend “wins” not claimed: measured no
  benefit on tiny MPS shapes (compile slower, backends identical, fp16 changes
  numerics) → all left opt-in/off by default.
- Files touched: `attention.py`, `encoder.py`, `decoder.py`, `model.py`,
  `config.py` (+ this doc). `train.py`/`generate.py`/`optim.py` untouched by
  this lane (worktree shows other lanes’ concurrent edits there — not mine).
