"""Training for minimal dense CED LLM (offline-first, CPU smoke).

Loss convention (caller shifts):
    logits_out = model(input_ids=x_in, attention_mask=mask_in)['logits']
    loss = CE(logits.view(-1, V), y.view(-1), ignore_index=pad)
where x_in = input_ids[:, :-1], y = input_ids[:, 1:].

If src/ced_llm/{config,model}.py are missing, a minimal fallback dense
Transformer LM is used (shim) so `--smoke` still runs offline.
"""

import argparse
import math
import os
import random
import sys

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception as e:
    raise ImportError("torch>=2.0 is required") from e

# -- data imports (support `python -m src.ced_llm.train` and pytest layouts) --
try:
    from .data import (
        SimpleTokenizer,
        encode_pack,
        get_dataloader,
        get_toy_batch,
        load_tinystories,
    )
except Exception:
    try:
        from src.ced_llm.data import (  # type: ignore
            SimpleTokenizer,
            encode_pack,
            get_dataloader,
            get_toy_batch,
            load_tinystories,
        )
    except Exception:
        # Last-resort: direct file load (defensive; keeps smoke alive).
        import importlib.util as _ilu

        _here = os.path.dirname(os.path.abspath(__file__))
        _spec = _ilu.spec_from_file_location(
            "_ced_data_fallback", os.path.join(_here, "data.py")
        )
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)  # type: ignore
        SimpleTokenizer = _mod.SimpleTokenizer
        encode_pack = _mod.encode_pack
        get_dataloader = _mod.get_dataloader
        get_toy_batch = _mod.get_toy_batch
        load_tinystories = _mod.load_tinystories

# -- tracking import (same layout tolerance; no-op fallback keeps smoke alive) --
try:
    from .tracking import RunTracker, load_metrics  # type: ignore
except Exception:
    try:
        from src.ced_llm.tracking import RunTracker, load_metrics  # type: ignore
    except Exception:

        class RunTracker:  # offline fallback: no-op tracker, same API
            def __init__(self, *a, **k):
                self.dir = None

            @property
            def active(self):
                return False

            def log(self, *a, **k):
                return None

            def close(self, *a, **k):
                return None

            @classmethod
            def disabled(cls):
                return cls()

        def load_metrics(*a, **k):
            return []


# -- optimizer import (same layout tolerance; AdamW fallback keeps smoke alive) --
try:
    from .optim import build_optimizer  # type: ignore
except Exception:
    try:
        from src.ced_llm.optim import build_optimizer  # type: ignore
    except Exception:
        build_optimizer = None


def _resolve_device(name):
    """Resolve --device (cpu/mps/auto/cuda) to a concrete string (never raises)."""
    try:
        s = str(name or "cpu").lower()
    except Exception:
        return "cpu"
    if s == "auto":
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        try:
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return s


def _autocast_ctx(device, dtype):
    """Null context for fp32 (default, bit-identical); autocast otherwise."""
    try:
        d = str(dtype or "fp32").lower()
    except Exception:
        d = "fp32"
    if d in ("fp32", "fp", "float32", "", "none"):
        from contextlib import nullcontext
        return nullcontext()
    try:
        dev = str(device or "cpu").lower()
        if dev not in ("mps", "cuda", "cpu"):
            dev = "cpu"
        amp = torch.bfloat16 if d in ("bf16", "bfloat16") else torch.float16
        # CPU autocast with fp16 is slow/unsupported on some builds; only
        # MPS/CUDA take the fast path. CPU+bf16 is allowed.
        if dev == "cpu" and amp is torch.float16:
            from contextlib import nullcontext
            return nullcontext()
        return torch.autocast(device_type=dev, dtype=amp)
    except Exception:
        from contextlib import nullcontext
        return nullcontext()


def _maybe_compile(model, want):
    """Opt-in torch.compile (default off). Never breaks training on failure."""
    try:
        if not bool(want):
            return model
    except Exception:
        return model
    try:
        # The encoder_forward_count int attr triggers dynamo recompiles on
        # tiny shapes; allow unspec ints so one graph survives counting.
        try:
            import torch._dynamo.config as _dc
            _dc.allow_unspec_int_on_nn_module = True
        except Exception:
            pass
        return torch.compile(model)
    except Exception as e:
        print("[train] WARNING: --compile requested but failed (%r); using eager." % (e,),
              file=sys.stderr)
        return model


def _new_optimizer(args, model, lr):
    """Build the CLI-selected optimizer (AdamW default; Muon optional)."""
    try:
        if build_optimizer is not None:
            kw = dict(
                muon_lr=float(getattr(args, "muon_lr", 0.02)),
                momentum=float(getattr(args, "muon_momentum", 0.95)),
            )
            # foreach/fused fast paths: opt-in only (default None == vanilla).
            try:
                if bool(getattr(args, "fused", False)):
                    kw["fused"] = True
                elif bool(getattr(args, "foreach", False)):
                    kw["foreach"] = True
            except Exception:
                pass
            try:
                kw["ns_steps"] = int(getattr(args, "muon_ns_steps", 5))
            except Exception:
                pass
            return build_optimizer(
                getattr(args, "optimizer", "adamw"),
                model.parameters(),
                lr=float(lr),
                **kw,
            )
    except Exception:
        pass
    return torch.optim.AdamW(model.parameters(), lr=float(lr))


def _new_tracker(args, extra_config):
    """Build a RunTracker from CLI args (disabled with --no-track)."""
    try:
        if bool(getattr(args, "no_track", False)):
            return RunTracker.disabled()
        cfg = {"seed": int(getattr(args, "seed", 0))}
        try:
            cfg.update(dict(extra_config or {}))
        except Exception:
            pass
        try:
            buf = int(getattr(args, "tracker_buffer", 1) or 1)
        except Exception:
            buf = 1
        try:
            return RunTracker(
                run_dir=getattr(args, "run_dir", "runs"),
                run_name=getattr(args, "run_name", None),
                config=cfg,
                tensorboard=bool(getattr(args, "tensorboard", False)),
                buffer_rows=buf,
            )
        except TypeError:
            # Fallback tracker without buffer_rows (offline no-op shim).
            return RunTracker(
                run_dir=getattr(args, "run_dir", "runs"),
                run_name=getattr(args, "run_name", None),
                config=cfg,
                tensorboard=bool(getattr(args, "tensorboard", False)),
            )
    except Exception:
        try:
            return RunTracker.disabled()
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Model loading: prefer real CED model, else fallback shim.
# ---------------------------------------------------------------------------
_CEDConfig = None
_CEDForLM = None
_MODEL_IMPORT_ERROR = None

try:
    try:
        from .config import CEDConfig as _CC  # type: ignore

        _CEDConfig = _CC
    except Exception:
        from src.ced_llm.config import CEDConfig as _CC  # type: ignore

        _CEDConfig = _CC
except Exception as e:
    _MODEL_IMPORT_ERROR = e

try:
    try:
        from .model import CEDForLM as _CM  # type: ignore

        _CEDForLM = _CM
    except Exception:
        from src.ced_llm.model import CEDForLM as _CM  # type: ignore

        _CEDForLM = _CM
except Exception as e:
    _MODEL_IMPORT_ERROR = e


# -- Minimal fallback dense LM (shim; used only if real model is missing). --
from dataclasses import dataclass, asdict  # noqa: E402


@dataclass
class _FallbackConfig:
    vocab_size: int = 512
    d_model: int = 64
    n_enc_layers: int = 1
    n_dec_layers: int = 1
    nhead: int = 4
    dim_ff: int = 128
    max_seq_len: int = 32
    dropout: float = 0.0
    pad_token_id: int = 0


def _resolve_nhead(d_model, want):
    try:
        want = int(want)
    except Exception:
        want = 4
    if want > 0 and d_model % want == 0:
        return want
    for h in (4, 2, 8, 1):
        try:
            h = int(h)
        except Exception:
            continue
        if h > 0 and d_model % h == 0:
            return h
    return 1


class _FallbackLM(nn.Module):
    """Tiny dense decoder-only Transformer exposing the CEDForLM API.

    Methods: forward, encode_once, init_decode_cache, forward_step,
             reset_encoder_counter, encoder_forward_count.
    forward_step does NOT increment the encoder counter (KV-reuse demo).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder_forward_count = 0
        d = int(getattr(config, "d_model", 64))
        V = int(getattr(config, "vocab_size", 512))
        max_len = int(getattr(config, "max_seq_len", 32))
        nhead = _resolve_nhead(d, getattr(config, "nhead", 4))
        dim_ff = int(getattr(config, "dim_ff", 4 * d))
        drop = float(getattr(config, "dropout", 0.0))
        n_layers = max(
            1, int(getattr(config, "n_enc_layers", 1)) + int(getattr(config, "n_dec_layers", 1))
        )
        self.token_emb = nn.Embedding(V, d)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=drop,
            batch_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, V, bias=False)
        self._cache = None

    def reset_encoder_counter(self):
        self.encoder_forward_count = 0

    def _forward_hidden(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        device = input_ids.device
        max_len = self.pos_emb.num_embeddings
        if T > max_len:
            input_ids = input_ids[:, -max_len:]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -max_len:]
            B, T = input_ids.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        pad_id = getattr(self.config, "pad_token_id", 0)
        key_padding_mask = None
        try:
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
                if not bool(key_padding_mask.any()):
                    key_padding_mask = None
            else:
                m = input_ids == int(pad_id)
                if bool(m.any()):
                    key_padding_mask = m
        except Exception:
            key_padding_mask = None
        try:
            h = self.layers(
                x, mask=None, src_key_padding_mask=key_padding_mask, is_causal=True
            )
        except TypeError:
            causal = torch.triu(
                torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
            )
            fm = torch.zeros(T, T, device=device)
            fm.masked_fill_(causal, float("-inf"))
            h = self.layers(x, mask=fm, src_key_padding_mask=key_padding_mask)
        return self.ln(h)

    def forward(self, input_ids, attention_mask=None, labels=None):
        try:
            self.encoder_forward_count += 1
        except Exception:
            self.encoder_forward_count = 1
        h = self._forward_hidden(input_ids, attention_mask)
        logits = self.lm_head(h)
        out = {"logits": logits}
        if labels is not None:
            try:
                pad = int(getattr(self.config, "pad_token_id", 0))
            except Exception:
                pad = 0
            out["loss"] = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=pad,
            )
        return out

    def encode_once(self, input_ids, attention_mask=None):
        try:
            self.encoder_forward_count += 1
        except Exception:
            self.encoder_forward_count = 1
        h = self._forward_hidden(input_ids, attention_mask)
        try:
            self._cache = {
                "memory": h.detach(),
                "input_ids": input_ids.detach().clone(),
            }
        except Exception:
            self._cache = {"memory": h, "input_ids": input_ids}
        return h

    def init_decode_cache(self, input_ids, attention_mask=None):
        enc = self.encode_once(input_ids, attention_mask)
        try:
            self._cache = {
                "memory": enc.detach(),
                "input_ids": input_ids.detach().clone(),
                "attention_mask": (
                    attention_mask.detach().clone()
                    if isinstance(attention_mask, torch.Tensor)
                    else attention_mask
                ),
            }
        except Exception:
            self._cache = {"memory": enc, "input_ids": input_ids}
        return self._cache

    def forward_step(self, next_token_ids, cache):
        # Do NOT touch encoder_forward_count (KV reuse).
        if not isinstance(next_token_ids, torch.Tensor):
            next_token_ids = torch.tensor(next_token_ids, dtype=torch.long)
        device = self.token_emb.weight.device
        try:
            next_token_ids = next_token_ids.to(device)
        except Exception:
            pass
        if next_token_ids.dim() == 1:
            next_token_ids = next_token_ids.unsqueeze(1)
        c = cache if isinstance(cache, dict) else self._cache
        if isinstance(c, dict) and isinstance(c.get("input_ids"), torch.Tensor):
            try:
                prev = c["input_ids"].to(device)
            except Exception:
                prev = c["input_ids"]
            full = torch.cat([prev, next_token_ids], dim=1)
            max_len = self.pos_emb.num_embeddings
            if full.size(1) > max_len:
                full = full[:, -max_len:]
            c["input_ids"] = full
            h = self._forward_hidden(full, None)
            logits = self.lm_head(h[:, -1:, :])
            c["memory"] = h
            self._cache = c
            return {"logits": logits, "cache": c}
        h = self._forward_hidden(next_token_ids, None)
        logits = self.lm_head(h)
        new_c = {"input_ids": next_token_ids, "memory": h}
        self._cache = new_c
        return {"logits": logits, "cache": new_c}


def _make_config(vocab_size, d_model, n_enc, n_dec, nhead, dim_ff, max_seq_len, dropout, pad_id):
    if _CEDConfig is not None:
        try:
            return _CEDConfig(
                vocab_size=vocab_size,
                d_model=d_model,
                n_enc_layers=n_enc,
                n_dec_layers=n_dec,
                nhead=nhead,
                dim_ff=dim_ff,
                max_seq_len=max_seq_len,
                dropout=dropout,
                pad_token_id=pad_id,
            )
        except TypeError:
            # Try alternate field names defensively.
            try:
                return _CEDConfig(
                    vocab_size=vocab_size,
                    d_model=d_model,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                    pad_token_id=pad_id,
                )
            except Exception:
                pass
        except Exception:
            pass
    return _FallbackConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_enc_layers=n_enc,
        n_dec_layers=n_dec,
        nhead=nhead,
        dim_ff=dim_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pad_token_id=pad_id,
    )


def _make_model(config):
    if _CEDForLM is not None:
        try:
            return _CEDForLM(config)
        except Exception as e:
            print(
                "[train] WARNING: real CEDForLM init failed (%r); using fallback shim." % (e,),
                file=sys.stderr,
            )
    if isinstance(config, _FallbackConfig):
        return _FallbackLM(config)
    # Real config but no real model class: adapt to fallback.
    try:
        fc = _FallbackConfig(
            vocab_size=int(getattr(config, "vocab_size", 512)),
            d_model=int(getattr(config, "d_model", 64)),
            n_enc_layers=int(getattr(config, "n_enc_layers", 1)),
            n_dec_layers=int(getattr(config, "n_dec_layers", 1)),
            nhead=int(getattr(config, "nhead", 4)),
            dim_ff=int(getattr(config, "dim_ff", 256)),
            max_seq_len=int(getattr(config, "max_seq_len", 32)),
            dropout=float(getattr(config, "dropout", 0.0)),
            pad_token_id=int(getattr(config, "pad_token_id", 0)),
        )
    except Exception:
        fc = _FallbackConfig()
    return _FallbackLM(fc)


def _config_to_dict(cfg):
    try:
        if isinstance(cfg, dict):
            return dict(cfg)
    except Exception:
        pass
    try:
        return asdict(cfg)  # dataclass
    except Exception:
        pass
    out = {}
    for k in (
        "vocab_size",
        "d_model",
        "n_enc_layers",
        "n_dec_layers",
        "nhead",
        "dim_ff",
        "max_seq_len",
        "dropout",
        "pad_token_id",
    ):
        try:
            out[k] = getattr(cfg, k)
        except Exception:
            pass
    return out


# ---------------------------------------------------------------------------
# Core training ops
# ---------------------------------------------------------------------------
def _infer_pad_id(model, override=None):
    if override is not None:
        try:
            return int(override)
        except Exception:
            pass
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            if isinstance(cfg, dict) and "pad_token_id" in cfg:
                return int(cfg["pad_token_id"])
            v = getattr(cfg, "pad_token_id", None)
            if v is not None:
                return int(v)
    except Exception:
        pass
    return 0


def _extract_logits(model_out):
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)) and len(model_out) > 0:
        return model_out[0]
    if hasattr(model_out, "logits"):
        try:
            return model_out.logits
        except Exception:
            pass
    if isinstance(model_out, torch.Tensor):
        return model_out
    raise TypeError("model forward did not return logits (dict['logits'] expected)")


def compute_loss(model, batch, pad_token_id=None):
    """Shifted next-token CE loss.

    logits_out = model(input_ids=x_in, attention_mask=mask_in)['logits']
    loss = CE(logits.view(-1,V), y.view(-1), ignore_index=pad)
    Returns (loss, logits).
    """
    try:
        x = batch["input_ids"]
    except Exception:
        # Support tuple-style batches defensively.
        x = batch[0]
    try:
        mask = batch.get("attention_mask", None) if isinstance(batch, dict) else batch[1]
    except Exception:
        mask = None
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.long)
    if mask is not None and not isinstance(mask, torch.Tensor):
        try:
            mask = torch.as_tensor(mask, dtype=torch.long)
        except Exception:
            mask = None
    # Move batch to model device only when needed (avoids re-alloc/copy).
    try:
        p = next(model.parameters())
        dev = p.device
        try:
            if x.device != dev:
                x = x.to(dev, non_blocking=True)
        except Exception:
            x = x.to(dev)
        if isinstance(mask, torch.Tensor):
            try:
                if mask.device != dev:
                    mask = mask.to(dev, non_blocking=True)
            except Exception:
                try:
                    mask = mask.to(dev)
                except Exception:
                    pass
    except Exception:
        pass
    if x.size(1) < 2:
        raise ValueError("seq_len must be >= 2 for shifted LM loss")
    x_in = x[:, :-1]
    y = x[:, 1:]
    mask_in = mask[:, :-1] if isinstance(mask, torch.Tensor) else None
    pad = _infer_pad_id(model, pad_token_id)
    out = model(input_ids=x_in, attention_mask=mask_in)
    logits = _extract_logits(out)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=int(pad)
    )
    return loss, logits


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device="cpu",
    max_steps=None,
    scheduler=None,
    pad_token_id=None,
    grad_clip=1.0,
    log_every=20,
    dtype="fp32",
    loss_sync_every=1,
    **kwargs,
):
    """One training epoch. Returns {'loss': avg_loss, 'steps': n}.

    Overhead flags (defaults preserve historical CPU fp32 numerics exactly):
      dtype: 'fp32' (default, no autocast) or 'fp16'/'bf16' autocast on
        MPS/CUDA. fp16/bf16 change numerics; measure before assuming a win.
      loss_sync_every: 1 (default) syncs loss.item() every step (identical
        avg). >1 accumulates the loss on-device and syncs once per K steps
        + once at the end (fewer MPS syncs; avg may differ in last ulp).
    """
    # Back-compat: allow pad_token_id passed positionally via kwargs aliases.
    if pad_token_id is None:
        pad_token_id = kwargs.get("pad_id", kwargs.get("pad", None))
    try:
        model.train()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass
    try:
        _sync_every = int(kwargs.get("loss_sync_every", loss_sync_every) or 1)
    except Exception:
        _sync_every = 1
    try:
        _dtype = str(kwargs.get("dtype", dtype) or "fp32")
    except Exception:
        _dtype = "fp32"
    if _sync_every < 1:
        _sync_every = 1
    total = 0.0
    n = 0
    # Deferred-sync accumulator (only used when _sync_every > 1).
    _accum = None
    _accum_n = 0
    try:
        max_steps = int(max_steps) if max_steps is not None else None
    except Exception:
        max_steps = None
    # Resolve target device once (avoids per-step torch.device + .to dispatch).
    try:
        _tgt = torch.device(device) if not isinstance(device, torch.device) else device
    except Exception:
        _tgt = None
    for batch in dataloader:
        # Move to device only when needed.
        try:
            if isinstance(batch, dict):
                items = batch.items()
            else:
                items = (("input_ids", batch[0]), ("attention_mask", batch[1]))
            nb = {}
            for k, v in items:
                if isinstance(v, torch.Tensor) and _tgt is not None:
                    try:
                        nb[k] = v if v.device == _tgt else v.to(_tgt, non_blocking=True)
                    except Exception:
                        nb[k] = v
                else:
                    nb[k] = v
            batch = nb
        except Exception:
            pass
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, _dtype):
            loss, _ = compute_loss(model, batch, pad_token_id=pad_token_id)
        try:
            loss.backward()
        except Exception:
            optimizer.zero_grad()
            continue
        try:
            if grad_clip is not None:
                _gc = float(grad_clip)
                if _gc > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _gc)
        except Exception:
            pass
        optimizer.step()
        if scheduler is not None:
            try:
                scheduler.step()
            except Exception:
                pass
        n += 1
        if _sync_every <= 1:
            # loss.item() already syncs; detach().cpu() is redundant overhead.
            total += float(loss.item())
        else:
            try:
                d = loss.detach()
                _accum = d if _accum is None else (_accum + d)
                _accum_n += 1
                if (_accum_n % _sync_every) == 0 or (
                    max_steps is not None and n >= max_steps
                ):
                    total += float(_accum.item())
                    _accum = None
                    _accum_n = 0
                    # NOTE: total currently holds the SUM; avg divides by n
                    # at the end, but batched item() summed K losses at once
                    # (same sum, fewer syncs; fp32 rounding may differ 1ulp).
            except Exception:
                total += float(loss.item())
        if max_steps is not None and n >= max_steps:
            break
    # Flush any deferred accumulator. total holds the sum of synced chunks
    # plus (below) the trailing partial chunk.
    try:
        if _accum is not None and _accum_n > 0:
            # _accum is a sum of _accum_n losses; total already holds the sum
            # of prior chunks, so add this chunk's sum once.
            # To recover the true sum we tracked total as sum-of-chunks, but
            # above we did total += float(chunk_sum) per flush, so just add.
            total += float(_accum.item())
    except Exception:
        pass
    avg = total / max(1, n)
    return {"loss": avg, "steps": n}


@torch.no_grad()
def evaluate(model, dataloader, device="cpu", pad_token_id=None, max_batches=None, **kwargs):
    """Mean loss + perplexity. Returns {'loss': float, 'ppl': float}."""
    if pad_token_id is None:
        pad_token_id = kwargs.get("pad_id", kwargs.get("pad", None))
    try:
        _dtype = str(kwargs.get("dtype", "fp32") or "fp32")
    except Exception:
        _dtype = "fp32"
    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to(device)
    except Exception:
        pass
    total = 0.0
    n = 0
    try:
        max_batches = int(max_batches) if max_batches is not None else None
    except Exception:
        max_batches = None
    try:
        _tgt = torch.device(device) if not isinstance(device, torch.device) else device
    except Exception:
        _tgt = None
    for batch in dataloader:
        try:
            if isinstance(batch, dict):
                items = batch.items()
            else:
                items = (("input_ids", batch[0]), ("attention_mask", batch[1]))
            nb = {}
            for k, v in items:
                if isinstance(v, torch.Tensor) and _tgt is not None:
                    try:
                        nb[k] = v if v.device == _tgt else v.to(_tgt, non_blocking=True)
                    except Exception:
                        nb[k] = v
                else:
                    nb[k] = v
            batch = nb
        except Exception:
            pass
        try:
            with _autocast_ctx(device, _dtype):
                loss, _ = compute_loss(model, batch, pad_token_id=pad_token_id)
        except Exception:
            continue
        # Under @torch.no_grad, detach().cpu() is pure overhead.
        total += float(loss.item())
        n += 1
        if max_batches is not None and n >= max_batches:
            break
    avg = total / max(1, n)
    try:
        ppl = math.exp(min(avg, 20.0))
    except Exception:
        ppl = float("inf")
    return {"loss": avg, "ppl": ppl}


def set_seed(seed):
    try:
        seed = int(seed)
    except Exception:
        seed = 0
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed % (2 ** 32))
    except Exception:
        pass
    try:
        torch.manual_seed(seed)
    except Exception:
        pass
    return seed


def _fixed_toy_loader(vocab_size, seq_len, batch_size, num_batches=8, seed=0):
    """Deterministic FIXED toy batches (reused each epoch so loss can drop)."""
    g = torch.Generator()
    try:
        g.manual_seed(int(seed))
    except Exception:
        pass
    batches = []
    for _ in range(max(1, int(num_batches))):
        ids = torch.randint(4, max(5, int(vocab_size)), (int(batch_size), int(seq_len)), generator=g)
        batches.append(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        )

    class _FixedLoader:
        def __init__(self, bs):
            self.bs = bs

        def __iter__(self):
            # No per-step clone: batches are treated read-only by the
            # training loop (slicing/views + .to() never mutate in place),
            # so sharing tensors avoids a full [B, T] alloc+copy per step.
            for b in self.bs:
                yield {"input_ids": b["input_ids"], "attention_mask": b["attention_mask"]}

        def __len__(self):
            return len(self.bs)

    return _FixedLoader(batches)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train minimal dense CED LM")
    p.add_argument("--smoke", action="store_true", help="offline smoke: tiny toy run")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data", type=str, default="toy", choices=["toy", "tinystories"])
    p.add_argument("--max-examples", type=int, default=500)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    # Extra (optional) model-size flags; smoke overrides them.
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-enc", type=int, default=2)
    p.add_argument("--n-dec", type=int, default=2)
    p.add_argument("--nhead", type=int, default=4)
    # Run tracking (offline JSONL by default; TensorBoard only with --tensorboard).
    p.add_argument("--run-dir", type=str, default="runs",
                   help="parent dir for tracked runs (each run gets a subdir)")
    p.add_argument("--run-name", type=str, default=None,
                   help="run subdir name (default: run-YYYYMMDD-HHMMSS)")
    p.add_argument("--no-track", action="store_true",
                   help="disable run tracking (no files written)")
    p.add_argument("--tensorboard", action="store_true",
                   help="mirror scalars to TensorBoard (needs pip install tensorboard)")
    p.add_argument("--log-every", type=int, default=20,
                   help="log a metrics row every N steps")
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "muon"],
                   help="adamw = AdamW everywhere; muon = Muon for matrices + AdamW for rest")
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="Muon learning rate for 2D params (Muon scale, not AdamW scale)")
    p.add_argument("--muon-momentum", type=float, default=0.95,
                   help="Muon momentum coefficient")
    # FORGE-LOOP speed flags (all defaults preserve CPU fp32 numerics exactly).
    p.add_argument("--device", type=str, default="cpu",
                   choices=["cpu", "mps", "cuda", "auto"],
                   help="train device (default cpu; auto picks mps/cuda if available)")
    p.add_argument("--dtype", type=str, default="fp32",
                   choices=["fp32", "fp16", "bf16"],
                   help="fp32 default (identical); fp16/bf16 autocast on mps/cuda (measure!)")
    p.add_argument("--foreach", action="store_true",
                   help="AdamW foreach=True fast path (opt-in; ~+6%% MPS smoke)")
    p.add_argument("--fused", action="store_true",
                   help="AdamW fused=True fast path (opt-in; ~+20%% MPS smoke, ~+2%% real)")
    p.add_argument("--muon-ns-steps", type=int, default=5,
                   help="Muon Newton-Schulz iters (default 5 identical; 3 is cheaper)")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="grad clip norm (default 1.0 identical; <=0 disables for speed)")
    p.add_argument("--loss-sync-every", type=int, default=1,
                   help="sync loss.item() every K steps (default 1 identical; >1 fewer MPS syncs)")
    p.add_argument("--tracker-buffer", type=int, default=1,
                   help="tracker rows buffered before flush (default 1 identical)")
    p.add_argument("--compile", action="store_true",
                   help="opt-in torch.compile (measured SLOWER on tiny MPS shapes)")
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers for tinystories path (default 0 identical)")
    p.add_argument("--prefetch-factor", type=int, default=None,
                   help="prefetch per worker (only when --num-workers>0)")
    p.add_argument("--pin-memory", action="store_true",
                   help="DataLoader pin_memory (opt-in)")
    return p


def main(argv=None):
    args = build_argparser().parse_args(argv)
    set_seed(args.seed)
    device = _resolve_device(getattr(args, "device", "cpu"))
    try:
        _dtype = str(getattr(args, "dtype", "fp32") or "fp32").lower()
    except Exception:
        _dtype = "fp32"
    try:
        _grad_clip = float(getattr(args, "grad_clip", 1.0))
        if not (_grad_clip > 0):
            _grad_clip = None  # <=0 disables clipping (opt-in speed path)
    except Exception:
        _grad_clip = 1.0

    if args.smoke:
        steps = 60
        seq_len = 32
        batch_size = 4
        vocab_size = 512
        d_model, n_enc, n_dec, nhead, dim_ff = 64, 1, 1, 4, 256
        lr = 1e-3  # slightly higher to guarantee visible decrease on toy data
        print("[train] SMOKE mode: d_model=64 1+1 layers seq=32 batch=4 steps=%d device=%s dtype=%s" % (steps, device, _dtype))
        loader = _fixed_toy_loader(vocab_size, seq_len, batch_size, num_batches=8, seed=args.seed)
        eval_loader = loader
        config = _make_config(vocab_size, d_model, n_enc, n_dec, nhead, dim_ff, seq_len, 0.0, 0)
        model = _make_model(config)
        model.to(device)
        model = _maybe_compile(model, getattr(args, "compile", False))
        opt = _new_optimizer(args, model, lr)
        # Initial loss.
        init = evaluate(model, eval_loader, device=device, dtype=_dtype)
        print("[train] smoke init loss=%.4f ppl=%.2f" % (init["loss"], init["ppl"]))
        tracker = _new_tracker(args, {
            "mode": "smoke", "steps": steps, "seq_len": seq_len,
            "batch_size": batch_size, "lr": lr, "d_model": d_model,
            "n_enc": n_enc, "n_dec": n_dec, "nhead": nhead, "dim_ff": dim_ff,
            "vocab_size": vocab_size, "init_loss": float(init["loss"]),
            "optimizer": str(getattr(args, "optimizer", "adamw")),
            "muon_lr": float(getattr(args, "muon_lr", 0.02)),
        })
        if tracker is not None and getattr(tracker, "active", False):
            print("[track] run dir: %s" % tracker.dir)
            tracker.log(0, {"loss": float(init["loss"]), "ppl": float(init["ppl"])})
        # Train: cycle the fixed loader until `steps` optimizer updates.
        # NOTE: no per-step batch .to(device) here -- batches are already CPU
        # and compute_loss moves only when needed, saving a dict alloc + 2x
        # .to() dispatches per step.
        model.train()
        total, n = 0.0, 0
        while n < steps:
            for batch in loader:
                if n >= steps:
                    break
                opt.zero_grad(set_to_none=True)
                with _autocast_ctx(device, _dtype):
                    loss, _ = compute_loss(model, batch)
                loss.backward()
                try:
                    if _grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                except Exception:
                    pass
                opt.step()
                # Single .item() per step (was 2-3x when tracking/printing):
                # one MPS sync instead of three, identical value reused.
                loss_val = float(loss.item())
                total += loss_val
                n += 1
                if tracker is not None and getattr(tracker, "active", False) and n % 10 == 0:
                    tracker.log(n, {"loss": loss_val})
        final = evaluate(model, eval_loader, device=device, dtype=_dtype)
        print("[train] smoke final loss=%.4f ppl=%.2f (avg step loss=%.4f)" % (final["loss"], final["ppl"], total / max(1, n)))
        if not (final["loss"] < init["loss"]):
            # Retry once with more steps/higher LR before failing (deterministic).
            print("[train] WARNING: loss did not decrease; retrying with lr*3 ...")
            for batch in loader:
                opt2 = _new_optimizer(args, model, lr * 3)
                for _ in range(20):
                    opt2.zero_grad(set_to_none=True)
                    with _autocast_ctx(device, _dtype):
                        l, _ = compute_loss(model, batch)
                    l.backward()
                    try:
                        if _grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
                    except Exception:
                        pass
                    opt2.step()
                break
            final2 = evaluate(model, eval_loader, device=device, dtype=_dtype)
            print("[train] retry final loss=%.4f" % final2["loss"])
            final = final2
        if not (final["loss"] < init["loss"]):
            print(
                "[train] SMOKE FAILED: final loss %.4f not < init %.4f"
                % (final["loss"], init["loss"]),
                file=sys.stderr,
            )
            return 1
        print("[train] SMOKE OK: loss decreased %.4f -> %.4f" % (init["loss"], final["loss"]))
        if args.ckpt:
            try:
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": _config_to_dict(config),
                        "vocab_size": vocab_size,
                    },
                    args.ckpt,
                )
                print("[train] saved ckpt to %s" % args.ckpt)
            except Exception as e:
                print("[train] WARNING: ckpt save failed: %r" % (e,), file=sys.stderr)
        try:
            if tracker is not None and getattr(tracker, "active", False):
                tracker.log(max(1, n), {"loss": float(final["loss"]),
                                        "ppl": float(final["ppl"])})
                tracker.close({"init_loss": float(init["loss"]),
                               "final_loss": float(final["loss"]),
                               "final_ppl": float(final["ppl"]),
                               "avg_step_loss": total / max(1, n),
                               "steps": n, "ckpt": args.ckpt})
        except Exception:
            pass
        return 0

    # ---- Non-smoke path ----------------------------------------------------
    steps = max(1, int(args.steps))
    seq_len = max(2, int(args.seq_len))
    batch_size = max(1, int(args.batch_size))
    lr = float(args.lr)
    try:
        log_every = max(1, int(getattr(args, "log_every", 20) or 20))
    except Exception:
        log_every = 20

    if args.data == "tinystories":
        # Build tokenizer from a small corpus sample, then dataloader.
        print("[train] loading TinyStories (offline-safe) ...")
        sample_texts = []
        try:
            for ex in load_tinystories(split="train", streaming=True, max_examples=min(2000, int(args.max_examples))):
                t = ex.get("text", "") if isinstance(ex, dict) else str(ex)
                if t.strip():
                    sample_texts.append(t)
                if len(sample_texts) >= 500:
                    break
        except Exception:
            sample_texts = []
        if not sample_texts:
            sample_texts = ["Once upon a time there was a little bunny."]
        tokenizer = SimpleTokenizer(sample_texts, vocab_size=8000)
        vocab_size = int(tokenizer.vocab_size)
        try:
            _nw = int(getattr(args, "num_workers", 0) or 0)
        except Exception:
            _nw = 0
        try:
            _pf = getattr(args, "prefetch_factor", None)
            _pf = int(_pf) if _pf is not None else None
        except Exception:
            _pf = None
        loader = get_dataloader(
            tokenizer,
            split="train",
            seq_len=seq_len,
            batch_size=batch_size,
            max_examples=int(args.max_examples),
            shuffle=True,
            num_workers=_nw,
            persistent_workers=bool(_nw > 0),
            prefetch_factor=_pf,
            pin_memory=bool(getattr(args, "pin_memory", False)),
        )
        tok_vocab = tokenizer.to_dict()
    else:
        vocab_size = 512
        tokenizer = None
        tok_vocab = None
        # Deterministic fixed toy stream (reused) for stable training.
        loader = _fixed_toy_loader(vocab_size, seq_len, batch_size, num_batches=16, seed=args.seed)

    # Resolve model size: d_model must be divisible by nhead.
    d_model = int(args.d_model)
    nhead = _resolve_nhead(d_model, int(args.nhead))
    dim_ff = 4 * d_model
    config = _make_config(
        vocab_size, d_model, int(args.n_enc), int(args.n_dec),
        nhead, dim_ff, seq_len, 0.0, 0,
    )
    model = _make_model(config)
    model.to(device)
    model = _maybe_compile(model, getattr(args, "compile", False))
    opt = _new_optimizer(args, model, lr)
    tracker = _new_tracker(args, {
        "mode": "train", "data": args.data, "steps": steps, "seq_len": seq_len,
        "batch_size": batch_size, "lr": lr, "max_examples": int(args.max_examples),
        "d_model": d_model, "n_enc": int(args.n_enc), "n_dec": int(args.n_dec),
        "nhead": nhead, "dim_ff": dim_ff, "vocab_size": vocab_size,
        "log_every": log_every,
        "optimizer": str(getattr(args, "optimizer", "adamw")),
        "muon_lr": float(getattr(args, "muon_lr", 0.02)),
        "device": device, "dtype": _dtype,
    })
    if getattr(tracker, "active", False):
        print("[track] run dir: %s" % tracker.dir)

    # Cosine-ish decay (manual; constant also acceptable per spec).
    total, n = 0.0, 0
    model.train()
    it = 0
    while it < steps:
        for batch in loader:
            if it >= steps:
                break
            # No per-step batch .to(device) dict rebuild here: compute_loss
            # moves only when device differs (loader is already CPU).
            # Simple cosine decay on top of base lr.
            try:
                frac = it / max(1, steps)
                cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * frac))
                for pg in opt.param_groups:
                    pg["lr"] = cur_lr
            except Exception:
                pass
            opt.zero_grad(set_to_none=True)
            with _autocast_ctx(device, _dtype):
                loss, _ = compute_loss(model, batch)
            loss.backward()
            try:
                if _grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _grad_clip)
            except Exception:
                pass
            opt.step()
            # Single sync per step: reuse loss_val for avg + tracker + print.
            loss_val = float(loss.item())
            total += loss_val
            it += 1
            if getattr(tracker, "active", False) and (it == 1 or it % log_every == 0):
                try:
                    _lr_log = float(opt.param_groups[0]["lr"])
                except Exception:
                    _lr_log = lr
                tracker.log(it, {"loss": loss_val, "lr": _lr_log})
            if it == 1 or it % 100 == 0:
                try:
                    _lr_now = float(opt.param_groups[0]["lr"])
                except Exception:
                    _lr_now = lr
                print("[train] step=%d/%d loss=%.4f lr=%.2e" % (it, steps, loss_val, _lr_now), flush=True)
        # If loader exhausted but steps remain and loader is finite, loop again.
        if it < steps:
            # For HF-backed loaders len may be 1 epoch; rebuild is overkill:
            # just continue cycling the same loader object (it re-iterates).
            if n > 10:
                break
            n += 1
            continue
        break
    print("[train] done steps=%d avg_loss=%.4f" % (it, total / max(1, it)))
    ev_loss, ev_ppl = None, None
    try:
        ev = evaluate(model, loader, device=device, max_batches=20, dtype=_dtype)
        ev_loss, ev_ppl = float(ev["loss"]), float(ev["ppl"])
        print("[train] eval loss=%.4f ppl=%.2f" % (ev_loss, ev_ppl), flush=True)
    except Exception as e:
        print("[train] WARNING: eval failed: %r" % (e,))
    if args.ckpt:
        try:
            payload = {
                "model_state": model.state_dict(),
                "config": _config_to_dict(config),
                "vocab_size": vocab_size,
            }
            if tok_vocab is not None:
                payload["tokenizer_vocab"] = tok_vocab
            torch.save(payload, args.ckpt)
            print("[train] saved ckpt to %s" % args.ckpt)
        except Exception as e:
            print("[train] WARNING: ckpt save failed: %r" % (e,), file=sys.stderr)
            return 1
    try:
        if getattr(tracker, "active", False):
            tracker.log(max(1, it), {"loss": total / max(1, it)})
            tracker.close({"steps": it, "avg_step_loss": total / max(1, it),
                           "eval_loss": ev_loss, "eval_ppl": ev_ppl,
                           "ckpt": args.ckpt})
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
