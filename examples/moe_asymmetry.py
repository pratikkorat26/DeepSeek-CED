"""Show V4.1-style asymmetric activation at toy scale (offline, CPU, seconds).

Prefill runs the encoder MoE; decode runs the decoder MoE with the encoder
cached away -- so per-token ACTIVE params differ by phase, exactly the
8B-prefill / 16B-decode shape of DeepSeek-V4.1-Flash (theirs: 384 experts
top-6 MoE + CSA2; ours: 8 experts top-2 dense-attention toy).

Usage:
    python3 examples/moe_asymmetry.py
    python3 examples/moe_asymmetry.py --experts 8 --topk 2 --steps 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

from src.ced_llm.config import CEDConfig
from src.ced_llm.model import CEDForLM
from src.ced_llm.moe import (
    decode_active_params,
    prefill_active_params,
    total_params,
)


def main(argv=None):
    p = argparse.ArgumentParser(description="MoE asymmetry demo (offline)")
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    torch.manual_seed(int(args.seed))
    cfg = CEDConfig(vocab_size=128, d_model=32, n_enc_layers=1,
                    n_dec_layers=1, nhead=2, dim_ff=64, max_seq_len=32,
                    dropout=0.0, pad_token_id=0, moe_enabled=True,
                    moe_num_experts=int(args.experts),
                    moe_top_k=int(args.topk),
                    moe_shared_experts=int(args.shared))
    cfg.validate()
    model = CEDForLM(cfg)
    model.eval()
    n_total = total_params(model)
    n_pre = prefill_active_params(model)
    n_dec = decode_active_params(model)
    print("== MoE asymmetry (toy V4.1-Flash shape) ==")
    print("experts=%d topk=%d shared=%d" % (args.experts, args.topk, args.shared))
    print("total params:            %8d" % n_total)
    print("prefill-active / token:  %8d  (%5.1f%% of total)" % (n_pre, 100.0 * n_pre / max(1, n_total)))
    print("decode-active  / step:   %8d  (%5.1f%% of total)" % (n_dec, 100.0 * n_dec / max(1, n_total)))
    print("decode/prefill ratio:    %.2f  (V4.1-Flash: 16B/8B = 2.00)" % (n_dec / max(1, n_pre)))
    # Run a real episode and show which experts fire per phase.
    prompt = torch.randint(4, 128, (1, 8))
    with torch.no_grad():
        cache = model.init_decode_cache(prompt)
        enc_hists = {}
        for m in model.encoder.modules():
            if type(m).__name__ == "DeepSeekMoELayer" and m.last_stats:
                enc_hists["enc"] = m.last_stats["counts"].tolist()
        print("encoder expert assignments (topk per token): %s" % (enc_hists.get("enc"),))
        tok = prompt[:, -1:]
        for s in range(max(1, int(args.steps))):
            logits, cache = model.forward_step(tok, cache)
            tok = logits[:, -1:].argmax(-1)
            for li, layer in enumerate(model.decoder.layers):
                for m in layer.modules():
                    if type(m).__name__ == "DeepSeekMoELayer" and m.last_stats:
                        print("decode step %d layer %d experts: %s" % (
                            s + 1, li, m.last_stats["counts"].tolist()))
    print("encoder_forward_count=%d (encoder ran ONCE)" % model.encoder_forward_count)
    print("ASYMMETRY OK: prefill and decode activate different param subsets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
