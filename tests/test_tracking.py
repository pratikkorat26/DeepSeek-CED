"""Tests for offline run tracking (JSONL by default, TensorBoard optional).

Covers RunTracker file layout, disabled mode, the TensorBoard-missing
fallback, and an end-to-end train --smoke run writing metrics.jsonl.
All offline, all CPU, all fast.
"""

import json
import os

try:
    from src.ced_llm.tracking import RunTracker, load_metrics
except ImportError:
    from ced_llm.tracking import RunTracker, load_metrics


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def test_tracker_writes_config_metrics_summary(tmp_path):
    t = RunTracker(run_dir=str(tmp_path), run_name="demo",
                   config={"lr": 3e-4, "steps": 10})
    assert t.active
    assert os.path.isfile(os.path.join(t.dir, "config.json"))
    cfg = _read_json(os.path.join(t.dir, "config.json"))
    assert cfg["lr"] == 3e-4 and cfg["steps"] == 10
    t.log(1, {"loss": 5.0, "lr": 3e-4})
    t.log(2, {"loss": 4.5, "lr": 3e-4})
    rows = load_metrics(t.dir)
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[1]["loss"] == 4.5
    summary_path = t.close({"final_loss": 4.5})
    assert summary_path is not None and os.path.isfile(summary_path)
    summary = _read_json(summary_path)
    assert summary["summary"]["final_loss"] == 4.5


def test_tracker_disabled_writes_nothing(tmp_path):
    t = RunTracker.disabled()
    assert not t.active
    t.log(1, {"loss": 1.0})  # must not raise
    assert t.close({"x": 1}) is None
    assert os.listdir(str(tmp_path)) == []


def test_tracker_tensorboard_missing_still_logs_jsonl(tmp_path):
    # tensorboard may or may not be installed; either way JSONL must land.
    t = RunTracker(run_dir=str(tmp_path), run_name="tbprobe",
                   config={}, tensorboard=True)
    assert t.active
    t.log(1, {"loss": 2.0})
    t.close({})
    rows = load_metrics(t.dir)
    assert len(rows) == 1 and rows[0]["loss"] == 2.0


def test_load_metrics_missing_dir_returns_empty(tmp_path):
    assert load_metrics(str(tmp_path / "nope")) == []


def test_train_smoke_tracks_run(tmp_path):
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    rc = train_main(["--smoke", "--run-dir", str(tmp_path),
                     "--run-name", "smoke1"])
    assert rc == 0
    run_dir = os.path.join(str(tmp_path), "smoke1")
    rows = load_metrics(run_dir)
    assert len(rows) >= 2  # init row + periodic rows + final row
    assert os.path.isfile(os.path.join(run_dir, "config.json"))
    summary = _read_json(os.path.join(run_dir, "summary.json"))
    assert summary["summary"]["final_loss"] < summary["summary"]["init_loss"]


def test_train_smoke_no_track_writes_nothing(tmp_path, monkeypatch):
    try:
        from src.ced_llm.train import main as train_main
    except ImportError:
        from ced_llm.train import main as train_main
    monkeypatch.chdir(tmp_path)
    rc = train_main(["--smoke", "--no-track"])
    assert rc == 0
    assert os.listdir(str(tmp_path)) == []
