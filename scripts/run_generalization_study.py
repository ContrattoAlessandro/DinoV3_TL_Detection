"""Resumable DTLD-only A–D pilots, winning folds, frozen evaluation and acceptance."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.evaluation import sha256_file, write_json


def immutable(path, value):
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise ValueError(f"Frozen provenance changed: {path}")
    write_json(path, value)


def choose_pilot(reports, benchmark):
    eligible = [
        (tuple(rep["selection_key"]), name)
        for name, rep in reports.items()
        if rep["eligible"] and benchmark["models"]["A" if name == "B" else name]["eligible"]
    ]
    # Stable A,B,C,D ordering resolves a complete validation tie.
    return max(eligible, key=lambda pair: (pair[0], -"ABCD".index(pair[1])))[1] if eligible else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs/generalization_study")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    if status_path.exists() and not args.resume:
        raise ValueError("Study already exists; use --resume")
    status = (
        json.loads(status_path.read_text())
        if status_path.exists()
        else dict(state="running", started_at=datetime.now(timezone.utc).isoformat(), completed_stages=[])
    )
    if status["state"] in ("complete", "unsuccessful"):
        print(f"Study already {status['state']}; v5 promotion decision is frozen")
        return
    source_paths = [
        *sorted((REPO / "src/dinov3_global").glob("*.py")),
        *[
            REPO / "scripts" / name
            for name in (
                "run_generalization_study.py",
                "cross_validate.py",
                "evaluate.py",
                "evaluate_ood.py",
                "evaluate_vzc.py",
                "benchmark_generalization.py",
                "compare_generalization.py",
                "smoke_generalization.py",
            )
        ],
    ]
    inputs = [
        *[REPO / f"configs/generalization_{n}.yaml" for n in "abcd"],
        REPO / "configs/runtime.json",
        REPO / "metadata/dtld_city_folds.json",
        REPO / "datasets/DTLD/v2.0/DTLD_train.json",
        REPO / "metadata/atlas_labels.json",
        REPO / "datasets/VZC_TLD/download_manifest.json",
        *[
            REPO / f"runs/v5/city_cv/fold{f}/{name}"
            for f in range(4)
            for name in ("best.pt", "val_predictions.npz")
        ],
    ]
    inputs += [
        REPO / "runs/v5/evaluations/dtld_test/predictions.npz",
        REPO / "runs/v5/evaluations/vzc/labels_snapshot.json",
    ]
    inputs += [
        REPO / f"runs/v5/evaluations/{suffix}"
        for f in range(4)
        for suffix in (
            f"dtld_test/member{f}/predictions.npz",
            f"atlas/v5_fold{f}/predictions.csv",
            f"vzc/v5_fold{f}/predictions.npz",
        )
    ]
    plan = dict(
        protocol="DTLD-only, four fresh fold-0 seed-0 pilots, winning remaining folds seeds 1–3",
        max_full_training_runs=7,
        epochs=12,
        baseline="retained v5",
        external_status="exploratory",
        selection="DTLD validation mean RR/RG recall gated by matching v5 mAP and NoR recall minus .02; mAP then earlier epoch ties",
        primary_temperature=1.0,
        source_sha256={p.relative_to(REPO).as_posix(): sha256_file(p) for p in source_paths},
        input_sha256={p.relative_to(REPO).as_posix(): sha256_file(p) for p in inputs},
    )
    immutable(root / "plan.json", plan)
    status.update(state="running", pid=os.getpid())
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")

    def verify():
        for group in ("source_sha256", "input_sha256"):
            if any(sha256_file(REPO / path) != digest for path, digest in plan[group].items()):
                raise ValueError("Study sources/inputs changed; continuation is a different experiment")

    def stage(name, command):
        verify()
        if name in status["completed_stages"]:
            return
        status.update(stage=name, stage_started_at=time.time())
        write_json(status_path, status)
        print(f"Starting {name}", flush=True)
        with (root / f"{name}.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-B", "-u", *command],
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            status["child_pid"] = process.pid
            write_json(status_path, status)
            while True:
                try:
                    code = process.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    with (root / f"{name}.log").open("rb") as tail:
                        tail.seek(max(0, tail.seek(0, 2) - 2048))
                        lines = tail.read().decode("utf-8", errors="replace").replace("\r", "\n").splitlines()
                    status.update(progress=lines[-1] if lines else "initializing", updated_at=time.time())
                    write_json(status_path, status)
                    print(f"{name}: {status['progress']}", flush=True)
            if code:
                raise RuntimeError(f"{name} failed with exit code {code}; see {root / (name + '.log')}")
        verify()
        status["completed_stages"].append(name)
        status.update(child_pid=None, updated_at=time.time())
        write_json(status_path, status)

    def train(name, folds):
        stage(
            "train_" + name + "_" + "".join(map(str, folds)),
            [
                "scripts/cross_validate.py",
                "--config",
                f"configs/generalization_{name.lower()}.yaml",
                "--out",
                str(root / name),
                "--folds",
                "4",
                "--fold-indices",
                *map(str, folds),
                "--epochs",
                "12",
                "--device",
                "cuda",
                "--runtime-loader",
                "configs/runtime.json",
                "--fold-manifest",
                "metadata/dtld_city_folds.json",
                *(["--resume"] if args.resume else []),
            ],
        )

    def unsuccessful(reason):
        write_json(
            root / "decision.json",
            dict(
                promote=False,
                retain="v5",
                reason=reason,
                external_evaluation="not performed; no eligible complete candidate",
            ),
        )
        status.update(
            state="unsuccessful", reason=reason, completed_at=datetime.now(timezone.utc).isoformat()
        )
        write_json(status_path, status)

    try:
        smoke = root / "verification/smoke/report.json"
        if smoke.exists():
            if json.loads(smoke.read_text())["status"] != "complete":
                raise ValueError("GPU verification incomplete")
        else:
            stage("smoke", ["scripts/smoke_generalization.py", "--out", str(smoke.parent)])
        if not (root / "benchmark.json").exists():
            stage("benchmark", ["scripts/benchmark_generalization.py", "--out", str(root / "benchmark.json")])
        benchmark = json.loads((root / "benchmark.json").read_text())
        immutable(
            root / "latency_gate.json",
            dict(
                benchmark_sha256=sha256_file(root / "benchmark.json"),
                models={
                    k: dict(eligible=v["eligible"], ratios=v["latency_ratios"])
                    for k, v in benchmark["models"].items()
                },
            ),
        )
        pilots = {}
        for name in "ABCD":
            if name == "D" and not benchmark["models"]["D"]["eligible"]:
                pilots[name] = dict(eligible=False, reason="Completed pipeline exceeds 2x latency budget")
            else:
                train(name, [0])
                pilots[name] = json.loads((root / name / "fold0/fold_report.json").read_text())
            write_json(root / "pilots.json", pilots)
        winner = choose_pilot(pilots, benchmark)
        if winner is None:
            unsuccessful("No pilot meets the validation and latency gates")
            return
        immutable(
            root / "pilot_winner.json",
            dict(
                winner=winner,
                selection_key=pilots[winner]["selection_key"],
                pilot_results_sha256=sha256_file(root / "pilots.json"),
            ),
        )
        train(winner, [1, 2, 3])
        reports = [json.loads((root / winner / f"fold{f}/fold_report.json").read_text()) for f in range(4)]
        if not all(r["eligible"] for r in reports):
            unsuccessful("Winning configuration has at least one city fold without an eligible checkpoint")
            return
        checkpoints = [root / winner / f"fold{f}/best.pt" for f in range(4)]
        frozen = dict(
            winner=winner,
            configuration_sha256=sha256_file(REPO / f"configs/generalization_{winner.lower()}.yaml"),
            checkpoints=[
                dict(path=str(p), sha256=sha256_file(p), epoch=r["best_epoch"], seed=r["seed"])
                for p, r in zip(checkpoints, reports)
            ],
            rule="Fixed before any official DTLD test, ATLAS or VZC evaluation; T=1",
        )
        immutable(root / "frozen_candidate.json", frozen)
        evaluation = root / "evaluations"
        dtld = Path(status.get("dtld_output", evaluation / "dtld_test"))
        if (dtld / "status.json").exists() and json.loads((dtld / "status.json").read_text()).get(
            "status"
        ) == "complete":
            if "dtld_test" not in status["completed_stages"]:
                status["completed_stages"].append("dtld_test")
        elif dtld.exists() and any(dtld.iterdir()):
            # Keep interrupted artifacts intact; a fresh attempt uses the same frozen heads.
            attempt = 1
            while (evaluation / f"dtld_test_attempt{attempt}").exists():
                attempt += 1
            dtld = evaluation / f"dtld_test_attempt{attempt}"
        status["dtld_output"] = str(dtld)
        stage(
            "dtld_test",
            [
                "scripts/evaluate.py",
                "--ckpt",
                *map(str, checkpoints),
                "--out",
                str(dtld),
                "--device",
                "cuda",
                "--runtime-loader",
                "configs/runtime.json",
            ],
        )
        for f, p in enumerate(checkpoints):
            stage(
                f"atlas_fold{f}",
                [
                    "scripts/evaluate_ood.py",
                    "--labels",
                    "metadata/atlas_labels.json",
                    "--images",
                    "datasets/atlas_relevance_expanded",
                    "--ckpt",
                    str(p),
                    "--include-uncertain",
                    "--device",
                    "cuda",
                    "--out",
                    str(evaluation / f"atlas/candidate_fold{f}"),
                ],
            )
        stage(
            "vzc",
            [
                "scripts/evaluate_vzc.py",
                "--ckpt",
                *map(str, checkpoints),
                "--model-names",
                *[f"candidate_fold{f}" for f in range(4)],
                "--out",
                str(evaluation / "vzc"),
                "--batch",
                "4",
                "--device",
                "cuda",
            ],
        )
        if any(sha256_file(p) != row["sha256"] for p, row in zip(checkpoints, frozen["checkpoints"])):
            raise ValueError("Frozen checkpoint changed during evaluation")
        stage("comparison", ["scripts/compare_generalization.py", "--study", str(root), "--dtld", str(dtld)])
        status.update(state="complete", stage="complete", completed_at=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
    except BaseException as error:
        status.update(state="failed", error=str(error), updated_at=time.time())
        write_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
