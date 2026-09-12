"""Run tracking for CED training/eval (offline-first).

Every tracked run writes a self-contained directory::

    runs/<run-name>/
        config.json     # hyperparams + CLI args + git hash (best-effort)
        metrics.jsonl   # one JSON object per logged step: {"step": N, ...}
        summary.json    # final aggregates written by close()

No third-party service, no account, no network -- plain files you can
``cat``, ``grep`` or plot with five lines of Python. TensorBoard mirroring
is optional: pass ``tensorboard=True`` (needs ``pip install tensorboard``);
when the package is missing we warn once and keep the JSONL logs.

Usage from train/eval CLIs::

    python3 -m src.ced_llm.train --data tinystories --steps 500 --run-dir runs
    python3 -m src.ced_llm.train --smoke --run-dir runs --run-name debug1
    python3 -m src.ced_llm.train --steps 500 --no-track        # disable
    tensorboard --logdir runs                                 # if installed

Compare two runs without any dependency::

    python3 -c "
    import json
    for run in ['runs/a', 'runs/b']:
        rows = [json.loads(l) for l in open(run + '/metrics.jsonl')]
        print(run, 'steps:', len(rows), 'last loss:', rows[-1].get('loss'))
    "
"""

import datetime
import json
import os
import subprocess


def _utc_stamp():
    try:
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    except Exception:
        return "unknown-time"


def _git_hash():
    """Short git SHA of the working tree, or None (never raises)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8", "replace").strip() or None
    except Exception:
        return None


def _safe_jsonable(obj):
    """Coerce config values to JSON-safe primitives (never raises)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return "<unprintable>"


class RunTracker:
    """File-backed run logger with optional TensorBoard mirror."""

    def __init__(self, run_dir="runs", run_name=None, config=None,
                 tensorboard=False, enabled=True):
        self.enabled = bool(enabled)
        self.tb_wanted = bool(tensorboard)
        self.tb_writer = None
        self.tb_active = False
        self.dir = None
        self._metrics_path = None
        self._warned_tb = False
        if not self.enabled:
            return
        name = str(run_name) if run_name else "run-%s" % _utc_stamp()
        # Sanitize: keep it a single path component.
        name = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name) or "run"
        self.dir = os.path.join(str(run_dir), name)
        try:
            os.makedirs(self.dir, exist_ok=True)
        except Exception:
            self.enabled = False
            return
        self._metrics_path = os.path.join(self.dir, "metrics.jsonl")
        cfg = _safe_jsonable(dict(config or {}))
        cfg["_git_hash"] = _git_hash()
        cfg["_created_utc"] = _utc_stamp()
        try:
            with open(os.path.join(self.dir, "config.json"), "w") as f:
                json.dump(cfg, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            pass
        if self.tb_wanted:
            self._init_tensorboard()

    def _init_tensorboard(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except Exception:
            print("[track] WARNING: --tensorboard requested but the "
                  "'tensorboard' package is not installed; continuing with "
                  "JSONL logs only (pip install tensorboard to enable).")
            self._warned_tb = True
            return
        try:
            self.tb_writer = SummaryWriter(log_dir=self.dir)
            self.tb_active = True
        except Exception as e:
            print("[track] WARNING: TensorBoard init failed (%r); JSONL only." % (e,))
            self.tb_writer = None

    @classmethod
    def disabled(cls):
        """No-op tracker honoring the same API (for --no-track)."""
        obj = cls.__new__(cls)
        obj.enabled = False
        obj.tb_wanted = False
        obj.tb_writer = None
        obj.tb_active = False
        obj.dir = None
        obj._metrics_path = None
        obj._warned_tb = False
        return obj

    @property
    def active(self):
        return bool(self.enabled and self.dir)

    def log(self, step, metrics):
        """Append one metrics row. Never raises; no-op when disabled."""
        if not self.active:
            return
        try:
            step = int(step)
        except Exception:
            step = -1
        row = {"step": step}
        try:
            for k, v in dict(metrics or {}).items():
                try:
                    row[str(k)] = float(v) if isinstance(v, bool) is False and isinstance(
                        v, (int, float)) else v
                except Exception:
                    row[str(k)] = str(v)
        except Exception:
            return
        try:
            with open(self._metrics_path, "a") as f:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            pass
        if self.tb_active and self.tb_writer is not None:
            try:
                for k, v in row.items():
                    if k == "step":
                        continue
                    if isinstance(v, (int, float)):
                        self.tb_writer.add_scalar(str(k), float(v), step)
                try:
                    self.tb_writer.flush()
                except Exception:
                    pass
            except Exception:
                pass

    def close(self, summary=None):
        """Write summary.json and close the TB writer. Never raises."""
        if not self.active:
            return None
        payload = _safe_jsonable({"summary": dict(summary or {})})
        payload["summary"]["_closed_utc"] = _utc_stamp()
        path = os.path.join(self.dir, "summary.json")
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
        except Exception:
            return None
        if self.tb_writer is not None:
            try:
                self.tb_writer.close()
            except Exception:
                pass
            self.tb_active = False
        return path


def load_metrics(run_dir):
    """Read metrics.jsonl back into a list of dicts (empty list on any error)."""
    rows = []
    try:
        with open(os.path.join(str(run_dir), "metrics.jsonl")) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return rows
