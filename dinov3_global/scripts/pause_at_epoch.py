"""Pause this comparison at a completed epoch, retaining resumable state."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import psutil
import torch


def write(path, value):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--epoch", type=int, required=True)
    args = ap.parse_args()
    root = args.run.resolve()
    status_path = root / "status.json"
    status = json.loads(status_path.read_text())
    if status["current_job"] != "v4_safe/fold0" or status["state"] != "running":
        raise ValueError("Expected the first active comparison fold")
    supervisor = psutil.Process(status["pid"])
    child = psutil.Process(status["child_pid"])
    if not any(Path(arg).name == "compare_city.py" for arg in supervisor.cmdline()):
        raise ValueError("Supervisor command does not match comparison runner")
    if not any(Path(arg).name == "crossval_city.py" for arg in child.cmdline()):
        raise ValueError("Child command does not match city training")
    child_args = child.cmdline()
    if Path(child_args[child_args.index("--out") + 1]).resolve() != root / "v4_safe":
        raise ValueError("Child belongs to another experiment")
    if child_args[child_args.index("--fold-indices") + 1] != "0":
        raise ValueError("Child is not fold 0")
    audit = root / "throughput_audit"
    audit.mkdir(exist_ok=True)
    write(audit / "pause_request.json", {"requested_epoch": args.epoch,
          "supervisor_pid": supervisor.pid, "child_pid": child.pid,
          "requested_at": datetime.now(timezone.utc).isoformat(),
          "reason": "User requested an isolated throughput audit and safe resume"})
    # The child is independent. Stop queue dispatch, while letting the current
    # epoch and validation complete without touching its training process.
    supervisor.terminate()
    supervisor.wait(timeout=15)
    status.update(state="pausing_at_epoch", queue_paused=True, pause_target_epoch=args.epoch,
                  updated_at=datetime.now(timezone.utc).isoformat())
    write(status_path, status)
    print(f"Queue held; training child {child.pid} continues until saved epoch {args.epoch}", flush=True)
    fold = root / "v4_safe/fold0"
    deadline = time.monotonic() + 3600
    while child.is_running() and time.monotonic() < deadline:
        history = fold / "history.jsonl"
        lines = history.read_text().splitlines() if history.exists() else []
        try:
            last_epoch = json.loads(lines[-1])["epoch"] if lines else 0
        except (json.JSONDecodeError, KeyError):
            last_epoch = 0
        if last_epoch >= args.epoch:
            state = torch.load(fold / "last.pt", map_location="cpu", weights_only=True)
            if state["epoch"] < args.epoch or not all(k in state for k in ("head", "ema", "opt", "scaler", "rng_state")):
                raise RuntimeError("Checkpoint is incomplete; training has not been stopped")
            child.terminate()
            child.wait(timeout=20)
            for name in ("last.pt", "best.pt", "history.jsonl"):
                shutil.copy2(fold / name, audit / ("preserved_" + name))
            status.update(state="paused_for_throughput", pid=None, child_pid=None,
                          checkpoint_epoch=state["epoch"], updated_at=datetime.now(timezone.utc).isoformat())
            write(status_path, status)
            print(f"Paused at epoch {state['epoch']}; complete state preserved in {audit}", flush=True)
            return
        time.sleep(5)
    raise RuntimeError("Training exited or pause timed out before a verified checkpoint")


if __name__ == "__main__":
    main()
