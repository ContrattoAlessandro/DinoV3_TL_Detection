"""Run the resumable twelve-epoch pilot and all fixed-checkpoint evaluations."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.evaluation import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = REPO / "runs/v6_axial_pilot"
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    if status_path.exists() and not args.resume:
        raise ValueError("Pilot already exists; use --resume")
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(
        status="running", started_at_utc=datetime.now(timezone.utc).isoformat(), completed_stages=[])
    status.update(status="running", pid=os.getpid())
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

    def stage(name, command):
        if name in status["completed_stages"]:
            return
        status.update(stage=name, stage_started_at_unix=time.time())
        write_json(status_path, status)
        print(f"Starting {name}", flush=True)
        with (root / f"{name}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-u", *command], cwd=REPO, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    with (root / f"{name}.log").open("rb") as tail:
                        tail.seek(max(0, (root / f"{name}.log").stat().st_size - 2048))
                        lines = tail.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
                    progress = lines[-1] if lines else "initializing"
                    status.update(progress=progress, updated_at_unix=time.time(), child_pid=process.pid)
                    write_json(status_path, status)
                    print(f"{name}: {progress}", flush=True)
            if code:
                status.update(status="failed", exit_code=code)
                write_json(status_path, status)
                raise RuntimeError(f"{name} failed with exit code {code}; see {root / (name + '.log')}")
        status["completed_stages"].append(name)
        write_json(status_path, status)
        print(f"Completed {name}", flush=True)

    stage("training", ["scripts/cross_validate.py", "--config", "configs/spatial_axial.yaml",
                       "--out", str(root / "city_cv"), "--folds", "4", "--fold-indices", "0",
                       "--epochs", "12", "--device", "cuda", "--runtime-loader", "configs/runtime.json",
                       "--fold-manifest", "metadata/dtld_city_folds.json", *(["--resume"] if args.resume else [])])
    checkpoint = root / "city_cv/fold0/best.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    frozen = dict(architecture="v6_axial", checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
                  epoch=state["epoch"], seed=state["cfg"]["optim"]["seed"], weights="EMA",
                  selection="highest fold-0 validation mAP at T=1", epochs_trained=12)
    selected = root / "selected_checkpoint.json"
    if selected.exists() and json.loads(selected.read_text()) != frozen:
        raise ValueError("Selected checkpoint changed")
    write_json(selected, frozen)
    stage("dtld_test", ["scripts/evaluate.py", "--ckpt", str(checkpoint), "--out",
                        str(root / "evaluations/dtld_test"), "--device", "cuda",
                        "--runtime-loader", "configs/runtime.json"])
    stage("atlas", ["scripts/evaluate_ood.py", "--labels", "metadata/atlas_labels.json",
                    "--images", "datasets/atlas_relevance_expanded", "--ckpt", str(checkpoint),
                    "--include-uncertain", "--device", "cuda", "--out",
                    str(root / "evaluations/atlas/v6_axial_fold0")])
    stage("vzc", ["scripts/evaluate_vzc.py", "--ckpt", str(checkpoint), "--model-names", "v6_axial_fold0",
                  "--out", str(root / "evaluations/vzc"), "--batch", "4", "--device", "cuda"])
    stage("benchmark", ["scripts/benchmark_axial.py", "--pilot", str(root)])
    stage("comparison", ["scripts/compare_axial_pilot.py", "--pilot", str(root)])
    if sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    status.update(status="complete", stage="complete", completed_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(status_path, status)
    print("Pilot training, evaluations, and comparison complete", flush=True)


if __name__ == "__main__":
    main()
