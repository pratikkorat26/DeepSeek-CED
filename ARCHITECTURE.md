# ARCHITECTURE.md — CED: The Encoder That Encodes Once ⚡

> *In the beginning there was the prefix. And the encoder read it. ONCE. And it
> was good. And the decoder said "more tokens," and the encoder said "no, I
> already did mine, reuse the cache," and thus was KV-reuse born.*
> — The Lore Master (DOC-BOSS), probably carving this into a stone tablet.

This is the deep dive for `src/ced_llm/`. Read `README.md` for the quickstart;
read this when you want to know **why the encoder runs exactly once** and how
the decoder gets away with never asking twice.

---

## 1. The 30-second gospel

CED = **Context-Encoder-Decoder** (dense, v1 — no MoE, no GQA/MLA, no RoPE, no
sparse anything; positions are learned embeddings in `model.py`).

```
tokens ──▶ embed+pos ──▶ encoder layers (N_enc) ──┐
                                                  ├─▶ memory (computed ONCE)
prompt ──▶ encode_once / init_decode_cache ───────┘
                                                        │
next token ──▶ decoder layers (N_dec, causal+KV cache) ─┼─▶ logits ─▶ sample ─▶ next token
                 ▲                                      │
                 └────────── forward_step loop ─────────┘  (encoder NOT re-run)
```

Three ideas, that's the whole religion:

1. **Embed once, share everywhere.** `forward()` computes `x_emb` a single time
   and feeds the SAME tensor to encoder and decoder paths.
2. **Encode once, project once.** `encode_once` → causal encoder → final states
   `h_enc` → exactly one projection to shared global keys/values `K_g`/`V_g`.
3. **Decode incrementally, reuse globally.** `forward_step` reuses the identical
   `k_glob`/`v_glob` tensors (same `data_ptr()`) plus per-layer self-KV caches.

---

## 2. Component map (who does what)

```
src/ced_llm/
├── config.py     # CEDConfig dataclass: vocab, d_model, n_enc/dec, nhead, dim_ff,
│                 #   max_seq_len, dropout, pad_token_id, layer_norm_eps + validate()
├── attention.py  # CausalSelfAttention (is_causal SDPA fast path + padded path)
│                 #   GlobalCrossAttention (EVERY query sees EVERY non-pad encoder pos)
├── encoder.py    # CausalEncoder: N_enc × (LN → causal self-attn → resid → LN → GELU FFN → resid)
├── decoder.py    # CausalDecoder: N_dec × (LN → causal self-attn → LN → GLOBAL cross-attn → LN → FFN)
│                 #   + incremental per-layer (K, V) self-cache for forward_step
├── model.py      # CEDForLM: _embed, encode_once, forward, init_decode_cache, forward_step,
│                 #   encoder_forward_count (the snitch that counts encodes)
├── data.py       # SimpleTokenizer (offline word-level), load_tinystories (+synthetic fallback),
│                 #   encode_pack (BOS/EOS + truncate + pad), dataloaders, toy batch
├── train.py      # Caller-shifts loss, train_one_epoch, evaluate, --smoke CLI
└── generate.py   # generate_greedy: cache-once + step loop, sampling spells, --smoke CLI
```

### 2.1 Config (`config.py`)

| Field | Meaning | Typical tiny / real |
|---|---|---|
| `vocab_size` | Token embeddings + LM head outputs | 64–128 (tests) / 6702–8000 (TinyStories) |
| `d_model` | Model width (must divide by `nhead`) | 32 (tests) / 256 (real) |
| `n_enc_layers` / `n_dec_layers` | Encoder / decoder depth (≥1) | 1+1 (tests/smoke) / 2+2 (real) |
| `nhead` | Dense attention heads | 2 / 8 |
| `dim_ff` | FFN hidden dim | 64 / 1024 |
| `max_seq_len` | Learned position table size = HARD length cap | 32 / 128–1024 |
| `dropout` | Dropout p in [0, 1) | 0.0 (tests) / 0.1 (real) |
| `pad_token_id` | Ignored by LM loss (`ignore_index`) | 0 / 50256 |
| `layer_norm_eps` | All LayerNorms' eps | 1e-5 |

`validate()` raises `ValueError`/`TypeError` on nonsense. Past `max_seq_len`,
the model raises `ValueError("sequence length ... exceeds max_seq_len ...")` —
LOUDLY, by design (see §6).

### 2.2 Attention (`attention.py`)

Both modules are dense multi-head attention over `scaled_dot_product_attention`:

```
CausalSelfAttention:                      GlobalCrossAttention:
  Q = query prefix                          Q = decoder hidden (prefix)
  K,V = SAME prefix                         K,V = K_g / V_g (FULL encoder memory)
  mask = causal (lower-tri)                 mask = NON-causal (every query → every non-pad key)
         + pad-key blocking                          + pad-key blocking only
```

- **Fast path:** no pads → `is_causal=True` / `attn_mask=None` SDPA kernel, no
  `[B,1,T,T]` bool materialized. **Padded path:** explicit causal+pad mask.
  Math identical; the fast path just skips allocating masks for air.
- **Dropout guard:** `training and p != 0` else skip dispatch (identity).
- Why full-model logits are NOT prefix-invariant (test_causality.py's famous
  footnote): cross-attention is global, so changing input `[:, -1]` changes
  `k_glob/v_glob[:, -1]`, which shifts prefix logits through cross-attn by
  ~2e-2 on tiny configs. That's CORRECT. Causality lives in the SELF-attention;
  context-mixing lives in the CROSS-attention. Don't "fix" this. Ever.

### 2.3 Encoder (`encoder.py`)

```
x [B,T,D] ──▶ ┌ EncoderBlock × N_enc ──────────────────────────┐ ──▶ h_enc [B,T,D]
              │  h = x + drop( self_attn( LN(x), pad_mask ) )   │   (causal: pos t
              │  y = h + drop( FFN( LN(h) ) )   // GELU          │    sees only ≤t)
              └────────────────────────────────────────────────┘
```

Causal mask = lower-triangular. Prefix states `[:, :-1]` are bit-invariant to
last-token changes (tested). Final `h_enc` is the ONLY thing the decoder ever
learns about the prompt — via one projection.

### 2.4 Decoder (`decoder.py`)

Each layer, pre-norm, three sub-blocks:

```
h ──▶ h + drop( causal_self_attn( LN(h) ) )        // prefix-only, KV-cached
  ──▶ h + drop( cross_attn( LN(h), K_g, V_g ) )    // GLOBAL, K_g/V_g shared+reused
  ──▶ h + drop( FFN( LN(h) ) )                     // GELU
```

- **Self-attn** supports two modes: `_self_full` (training/`forward`, whole
  sequence at once) and `_self_step` (inference, one column at a time, appending
  to the per-layer `(K, V)` cache of shape `[B,H,T_past,Dh]`).
- **Cross-attn** has NO cache of its own because it needs none: `K_g`/`V_g` are
  frozen for the whole generation. Every layer reads the same two tensors.
- **FFN** is the usual `D → dim_ff → D` GELU sandwich.

---

## 3. The sacred flows

### 3.1 Training (`forward` + caller shift)

```
input_ids [B,T] ──▶ _embed ──▶ x_emb [B,T,D] ──┬──▶ encoder ──▶ h_enc ──▶ K_g, V_g
                                               └──▶ decoder(x_emb, K_g, V_g) ──▶ h_dec
                                                        ──▶ norm ──▶ lm_head ──▶ logits [B,T,V]

loss (CALLER shifts):  x_in = tok[:, :-1], y = tok[:, 1:]
                       CE( model(x_in)["logits"], y, ignore_index=pad )
```

Shapes at `B=2, T=8, D=32, V=128`: `x_emb [2,8,32]` → `h_enc [2,8,32]` →
`K_g,V_g [2,8,32]` → `logits [2,7,128]` after shift. Padding: `attention_mask`
blocks pads in self AND cross attention; pad labels are ignored in the loss
(`test_padding_invariance` proves padded loss == unpadded loss to 1e-4).

### 3.2 Generation, step by step (THE encoder-once liturgy)

Setup: prompt `"hello world"` → ids `[h, w]` (say `[4, 5]`), `max_new = 3`.

```
STEP 0 — init_decode_cache (THE ONE ENCODE):
─────────────────────────────────────────────
  input [1,2] ──▶ _embed ──▶ encoder ──▶ h_enc [1,2,D]
                                        ──▶ K_g [1,2,D], V_g [1,2,D]   (projected ONCE)
  cache = { k_glob, v_glob,               // ← SAME tensors forever (ptr frozen)
            self_kv: [None × N_dec],      // ← empty, will grow
            len: 2 }
  encoder_forward_count: 0 ──▶ 1   (and NEVER moves again)
  prev = last prompt col [w]

STEP 1 — forward_step([w]):
───────────────────────────
  decoder_one_col(w | self_kv=[None,None], K_g,V_g) ──▶ logits [1,1,V] ──▶ argmax ──▶ n1
  self_kv: [None,None] ──▶ [(K=[w],V=[w]) × N_dec]     // prompt cols live in K_g/V_g;
  k_glob/v_glob: UNTOUCHED (same data_ptr)             // self-cache holds only STEP inputs
  generated = [h, w, n1], prev = [n1]

STEP 2 — forward_step([n1]):
────────────────────────────
  decoder_one_col(n1 | self_kv=[w],[K_g,V_g]) ──▶ logits ──▶ n2
  self_kv grows: [w] ──▶ [w, n1]
  k_glob/v_glob: UNTOUCHED
  generated = [h, w, n1, n2], prev = [n2]

STEP 3 — forward_step([n2]):  ... same ...  ──▶ n3
  generated = [h, w, n1, n2, n3]   (2 + 3 = 5 = prompt + max_new ✔)
  encoder_forward_count STILL 1 ✔   (delta asserted == 1, else RuntimeError)
```

Why each step is O(1) instead of O(T): self-attention only scores the new
column against cached `K_past` (`[B,H,T_past,Dh]` append, no recompute), and
cross-attention scores one query row against frozen `K_g`/`V_g`. The encoder's
O(T²) cost is paid exactly once no matter how many tokens you generate — that's
the entire economic argument for CED.

### 3.3 Cache anatomy (what `init_decode_cache` returns)

```python
cache = {
    "k_glob": Tensor[B, T_prompt, D],   # shared global keys   (FROZEN, ptr-stable)
    "v_glob": Tensor[B, T_prompt, D],   # shared global values (FROZEN, ptr-stable)
    "self_kv": [(K, V) | None] * N_dec, # per-layer self caches, each [B,H,T_step,Dh]
    # ... plus whatever bookkeeping the impl needs (lengths, masks)
}
```

Contract (asserted by `test_single_encode_and_ptr_stability`):
- After K `forward_step` calls, `encoder_forward_count == 1`.
- `cache["k_glob"].data_ptr()` and `cache["v_glob"].data_ptr()` NEVER change.
- `forward_step` returns `logits [B,1,V]` per step; concatenating T steps equals
  full `forward` logits to 1e-4 (`test_parity_full_vs_incremental`).

---

## 4. Sampling spells (`generate.py`, chaos's playground)

```
logits [V] ──▶ repetition_penalty (÷/× by penalty, default 1.0 = off)
           ──▶ / temperature
                 ├── temp <= 0  ──▶ argmax            (DETERMINISTIC, ignores top-k/p)
                 └── temp > 0   ──▶ top-k filter ──▶ top-p filter ──▶ multinomial
```

- `generate_greedy(model, tok, prompt, max_new_tokens, device, temperature=0.0,
  top_k=0, top_p=1.0, repetition_penalty=1.0, seed=None, stop_on_eos=True)`
  returns `{text, token_ids, encoder_forwards}` and RAISES unless the encoder
  delta is exactly 1.
- Empty prompt `""` → seeded with BOS (never crashes; `test_edgecases_docs.py`).
- Stops early ONLY on a known EOS id; otherwise output is exactly
  `prompt_len + max_new`. Early-stop-without-EOS fails loudly.
- `seed=int` reseeds `torch`+`random` for reproducible mischief.

---

## 5. Data pipeline (`data.py`, offline-first or death)

```
TinyStories (HF datasets, if present+online)
   │  else: synthetic fallback generator (TinyStories-flavored word salad)
   ▼
texts ──▶ SimpleTokenizer(texts, vocab_size)   // lowercase+split, <pad><unk><bos><eos>
   │         encode: str → [ids] (no specials)   decode: [ids] → str (round-trips)
   ▼
encode_pack(texts, tok, seq_len):  [BOS] + ids[:seq_len-2] + [EOS] + [PAD…]  → exactly seq_len
   ▼
get_dataloader / get_toy_batch → {input_ids [B,T] long, attention_mask [B,T], ...}
```

`SimpleTokenizer.encode("")` → `[]`; `encode_pack([""], tok, 8)` →
`[BOS, EOS, PAD×6]`. Long inputs truncate content FIRST, then add specials, so
BOS/EOS survive truncation. The model itself does NOT auto-truncate overlong
prompts at generate time — it raises `ValueError` naming `max_seq_len` (callers
keep the tail; tested).

---

## 6. Limits & sharp edges (read before bleeding)

| Edge | Behavior | Rule |
|---|---|---|
| `T > max_seq_len` | `ValueError: sequence length … exceeds max_seq_len …` | Keep last `max_seq_len` tokens |
| Empty prompt `""` | BOS-seeded, `[BOS]+gen` | Don't pass 0-length tensors to the model directly |
| Single-token prompt | Works; prefix preserved | Same as above, but cuter |
| `temperature=0` | Pure argmax, ignores top-k/p, deterministic | For stories use temp 0.7–0.9 + seed |
| `max_new_tokens=0` | Returns prompt ids unchanged | Valid, boring, tested-adjacent |
| Pads in batch | Masked in attn, ignored in loss | Always pass `attention_mask` with pads |
| Full-logits prefix invariance | Does NOT hold (global cross-attn, by design) | Test self-causality instead |

---

## 7. Perf notes (for the stopwatch crowd)

Complexity per generation of `N` new tokens over prompt length `P`:

| Cost | Naïve (re-encode every step) | CED (this repo) |
|---|---|---|
| Encoder forwards | N | **1** |
| Encoder FLOPs | O(N · P² · D) | **O(P² · D)** once |
| Per-step self-attn | O((P+i)²) recompute | **O(P+i)** vs cache |
| Per-step cross-attn | — | O(P) vs frozen K_g/V_g |
| Memory overhead | — | K_g,V_g `[B,P,D]` + self-KV `[B,H,N,Dh]`/layer |

To reproduce the numbers for `README.md`'s benchmark table on YOUR box:

```bash
pytest -q
time python3 -m src.ced_llm.train --smoke
time python3 -m src.ced_llm.generate --smoke
```

Micro-bench template (forward vs step, CPU):

```python
import time, torch
from src.ced_llm.config import CEDConfig
from src.ced_llm.model import CEDForLM
cfg = CEDConfig(vocab_size=512, d_model=128, n_enc_layers=2, n_dec_layers=2,
                nhead=8, dim_ff=512, max_seq_len=128, dropout=0.0, pad_token_id=0)
m = CEDForLM(cfg).eval()
x = torch.randint(1, 512, (8, 128))
t0=time.perf_counter()
for _ in range(30):
    with torch.no_grad(): m(x)
print("forward avg:", (time.perf_counter()-t0)/30*1e3, "ms")
```

---

## 8. Invariants (the test suite is the spec)

- `test_causality.py` — self-attention is causal; cross-attention is global.
- `test_data.py` — tokenizer round-trips; packs are exact; loader works offline.
- `test_generate.py` — greedy deterministic; encoder x1; length-or-EOS.
- `test_kv_reuse.py` — parity ≤1e-4; ptr-stable globals; decoder USES globals.
- `test_training.py` — toy overfits (loss halves, acc >80%); grads reach encoder;
  pads don't leak.
- `test_edgecases_docs.py` — empty/single/long/temp=0 all behave (DOC-BOSS seal).

Break any of these and `pytest -q` will name you in the failure message. The
failure messages are long, specific, and faintly disappointed in you. As is
tradition. As it should be.

---
*Encoder runs once. Cache is forever. Docs are eternal.* ⚡📚
