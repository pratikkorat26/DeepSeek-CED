"""Speed benchmark for CED on Apple Silicon (offline, no deps beyond torch).

Measures train tok/s and generate tok/s at smoke size AND realistic size,
on CPU or MPS, with proper device sync around timers.

Usage:
    python3 benchmarks/speed.py                      # auto device, quick
    python3 benchmarks/speed.py --device mps --steps 100
    python3 benchmarks/speed.py --device cpu --json-out benchmarks/latest.json
    python3 benchmarks/speed.py --dtype fp16 --device mps
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

try:
    from src.ced_llm.train import (
        _fixed_toy_loader,
        _make_config,
        _make_model,
        compute_loss,
        set_seed,
    )
except ImportError:
    from ced_llm.train import (  # type: ignore
        _fixed_toy_loader,
        _make_config,
        _make_model,
        compute_loss,
        set_seed,
    )


def _sync(device):
    try:
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
    except Exception:
        pass


def bench_train(device="cpu", dtype="fp32", steps=60, batch_size=4,
                seq_len=32, d_model=64, n_enc=1, n_dec=1, nhead=4,
                vocab_size=512, warmup=5):
    set_seed(0)
    dim_ff = 4 * d_model
    loader = _fixed_toy_loader(vocab_size, seq_len, batch_size,
                               num_batches=8, seed=0)
    config = _make_config(vocab_size, d_model, n_enc, n_dec, nhead,
                          dim_ff, seq_len, 0.0, 0)
    model = _make_model(config)
    try:
        model.to(device)
    except Exception:
        return None
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    use_amp = dtype in ("fp16", "bf16") and device == "mps"
    amp_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
    batches = list(loader)

    def one_step(batch):
        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="mps", dtype=amp_dtype):
                loss, _ = compute_loss(model, batch)
        else:
            loss, _ = compute_loss(model, batch)
        loss.backward()
        opt.step()
        return loss

    model.train()
    for b in batches[:warmup]:
        one_step(_move(b, device))
    _sync(device)
    toks = 0
    t0 = time.time()
    n = 0
    while n < steps:
        for b in batches:
            if n >= steps:
                break
            one_step(_move(b, device))
            toks += batch_size * (seq_len - 1)
            n += 1
    _sync(device)
    dt = max(1e-9, time.time() - t0)
    return {"tok_s": toks / dt, "ms_step": dt / max(1, n) * 1000.0, "steps": n}


def _move(batch, device):
    try:
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()}
    except Exception:
        return batch


def bench_generate(device="cpu", dtype="fp32", max_new=64, d_model=64,
                   n_enc=1, n_dec=1, nhead=4, vocab_size=512, seq_len=64,
                   warmup_new=8, gen_loop="baseline", gen_batch_size=1):
    set_seed(0)
    dim_ff = 4 * d_model
    config = _make_config(vocab_size, d_model, n_enc, n_dec, nhead,
                          dim_ff, seq_len, 0.0, 0)
    model = _make_model(config)
    try:
        model.to(device)
    except Exception:
        return None
    model.eval()
    # FORGE-DECODE additive options (defaults preserve the legacy CLI/behavior):
    # gen_loop="forge" reuses one [B,1] input buffer + skips the per-step
    # mask-sync; gen_batch_size>1 amortizes MPS launch cost over B rows.
    try:
        _batch = max(1, int(gen_batch_size or 1))
    except Exception:
        _batch = 1
    _forge = str(gen_loop or "baseline").strip().lower() == "forge"
    prompt = torch.randint(4, vocab_size, (_batch, 8), device=device)
    use_amp = dtype in ("fp16", "bf16") and device == "mps"
    amp_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    def gen(n_tokens):
        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="mps", dtype=amp_dtype):
                    return _gen_loop(n_tokens)
            return _gen_loop(n_tokens)

    def _gen_loop(n_tokens):
        if _forge:
            return _gen_loop_forge(n_tokens)
        cache = model.init_decode_cache(prompt)
        tok = prompt[:, -1:]
        for _ in range(n_tokens):
            logits, cache = model.forward_step(tok, cache)
            tok = logits[:, -1:].argmax(-1)
        return tok

    def _gen_loop_forge(n_tokens):
        cache = model.init_decode_cache(prompt)
        try:
            if isinstance(cache, dict) and cache.get("enc_mask") is not None:
                cache["enc_mask"] = None
        except Exception:
            pass
        tok = prompt[:, -1:].clone()
        for _ in range(n_tokens):
            logits, cache = model.forward_step(tok, cache)
            tok.copy_(logits[:, -1:].argmax(-1))
        return tok

    gen(warmup_new)
    _sync(device)
    t0 = time.time()
    gen(max_new)
    _sync(device)
    dt = max(1e-9, time.time() - t0)
    toks = _batch * max_new
    return {"tok_s": toks / dt, "ms_token": dt / toks * 1000.0,
            "tokens": toks, "batch_size": _batch, "gen_loop": "forge" if _forge else "baseline"}


def main(argv=None):
    p = argparse.ArgumentParser(description="CED speed benchmark (offline)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    p.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--gen-tokens", type=int, default=64)
    p.add_argument("--json-out", default=None)
    # FORGE-DECODE additive perf options (defaults = legacy behavior).
    p.add_argument("--gen-loop", default="baseline", choices=["baseline", "forge"],
                   help="gen decode loop: baseline (legacy) or forge (buffer reuse, no per-step mask sync)")
    p.add_argument("--gen-batch-size", type=int, default=1,
                   help="batch rows for gen decode (amortizes MPS launch cost)")
    args = p.parse_args(argv)
    device = args.device
    if device == "auto":
        try:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        except Exception:
            device = "cpu"
    out = {"device": device, "dtype": args.dtype,
           "torch": torch.__version__,
           "gen_loop": args.gen_loop, "gen_batch_size": args.gen_batch_size}
    print("== CED speed bench: device=%s dtype=%s ==" % (device, args.dtype))
    for tag, kw in (("smoke", {}),
                    ("real", {"d_model": 128, "n_enc": 2, "n_dec": 2,
                              "seq_len": 128, "batch_size": 8})):
        r = bench_train(device, args.dtype, steps=args.steps, **kw)
        if r is None:
            print("%-12s train: device failed" % tag)
            continue
        out["train_" + tag] = r
        print("%-12s train: %8.0f tok/s  (%5.2f ms/step)" % (tag, r["tok_s"], r["ms_step"]))
    for tag, kw in (("smoke", {}),
                    ("real", {"d_model": 128, "n_enc": 2, "n_dec": 2,
                              "seq_len": 128})):
        r = bench_generate(device, args.dtype, max_new=args.gen_tokens,
                           gen_loop=args.gen_loop,
                           gen_batch_size=args.gen_batch_size, **kw)
        if r is None:
            print("%-12s gen:   device failed" % tag)
            continue
        out["gen_" + tag] = r
        print("%-12s gen:   %8.0f tok/s  (%5.2f ms/token)" % (tag, r["tok_s"], r["ms_token"]))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print("wrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
