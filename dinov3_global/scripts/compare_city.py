"""Run paired v4-safe/v5 city folds serially on one GPU.

The shared manifest is checked before training. Progress and subprocess logs
are saved after every job; rerun with --resume after an interruption. ATLAS
and the official DTLD test split are never loaded by this comparison.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dinov3_global.scripts.crossval_city import atomic_json
from dinov3_global.scripts.runtime_loader import validate_runtime


def now():
    return datetime.now(timezone.utc).isoformat()


def process_alive(pid):
    """Read process state without signalling another process on Windows."""
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        api = ctypes.WinDLL("kernel32", use_last_error=True)
        api.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        api.OpenProcess.restype = wintypes.HANDLE
        api.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        api.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = api.OpenProcess(0x1000, False, int(pid))
        if not handle:
            if ctypes.get_last_error() == 5:
                raise RuntimeError(f"cannot verify process {pid}; refusing concurrent resume")
            return False
        try:
            code = wintypes.DWORD()
            if not api.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError(f"cannot query process {pid}")
            return code.value == 259
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def source_hashes():
    files = list((REPO / "dinov3_global/dinov3_global").glob("*.py"))
    files += [Path(__file__).resolve(), REPO / "dinov3_global/scripts/crossval_city.py"]
    files += [REPO / "dinov3_global/scripts/runtime_loader.py"]
    files += [REPO / f"dinov3_global/configs/{v}.yaml" for v in ("v4_safe", "v5")]
    return {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=REPO / "dinov3_global/runs/city_comparison_20260930")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=0, help="smoke only; 0 uses full train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--runtime-loader", type=Path)
    ap.add_argument("--execution-revision", type=Path, help="record a verified user-authorized execution-only change")
    args = ap.parse_args()
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    if status_path.exists() and not args.resume:
        raise ValueError("comparison already exists; use --resume or a new --out")
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if any(process_alive(previous.get(k)) for k in ("pid", "child_pid")):
            raise ValueError("previous supervisor or training child is still running")
    sources = source_hashes()
    source_path = root / "source_manifest.json"
    if source_path.exists() and json.loads(source_path.read_text()) != sources:
        if args.execution_revision is None:
            raise ValueError("training code/config changed; use a new comparison directory or a verified execution revision")
        revision = json.loads(args.execution_revision.read_text())
        previous_sources = json.loads(source_path.read_text())
        if revision["previous_source_hashes"] != previous_sources or revision["new_source_hashes"] != sources:
            raise ValueError("execution revision does not match old and new sources")
        changed = {k for k in previous_sources.keys() | sources.keys() if previous_sources.get(k) != sources.get(k)}
        allowed = {"dinov3_global/scripts/crossval_city.py", "dinov3_global/scripts/compare_city.py",
                   "dinov3_global/scripts/runtime_loader.py"}
        if not changed.issubset(allowed):
            raise ValueError("execution revision changes model/data/loss/optimizer code or scientific configs")
        proof = json.loads(Path(revision["validation_report"]).read_text())
        if not proof.get("passed"):
            raise ValueError("execution equivalence checks have not passed")
        selected_runtime_path = args.runtime_loader or (root / "runtime_loader.json")
        selected_runtime = validate_runtime(json.loads(selected_runtime_path.read_text()))
        if proof.get("runtime_loader") != selected_runtime or revision.get("runtime_loader") != selected_runtime:
            raise ValueError("execution proof does not cover the selected runtime settings")
        archive = root / "execution_revisions"
        archive.mkdir(exist_ok=True)
        revision_name = f"revision_{len(list(archive.glob('revision_*.json'))) + 1}.json"
        atomic_json(archive / revision_name, revision)
    atomic_json(source_path, sources)
    runtime_path = root / "runtime_loader.json"
    runtime = None
    if args.runtime_loader is not None:
        runtime = validate_runtime(json.loads(args.runtime_loader.read_text()))
        if runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
            raise ValueError("runtime loader settings differ from existing execution; create a new revision")
        atomic_json(runtime_path, runtime)
    elif runtime_path.exists():
        runtime = validate_runtime(json.loads(runtime_path.read_text()))
    common = ["--folds", str(args.folds), "--epochs", str(args.epochs),
              "--warmup-epochs", str(min(3, args.epochs)), "--early-stop-patience", "0",
              "--max-items", str(args.max_items), "--device", args.device,
              "--fold-manifest", str(root / "fold_manifest.json")]
    if runtime is not None:
        common += ["--runtime-loader", str(runtime_path)]
    plan = {"epochs": args.epochs, "folds": args.folds, "max_items": args.max_items,
            "warmup_epochs": min(3, args.epochs), "early_stop_patience": 0,
            "selection_metric": "mAP at T=1 on EMA", "seed": "0 + fold index",
            "variants": ["v4_safe", "v5"],
            "order": [[fi, variant] for fi in range(args.folds) for variant in ("v4_safe", "v5")],
            "source": "official DTLD train only; city-disjoint configuration selection"}
    plan_path = root / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("schedule differs from existing comparison")
    atomic_json(plan_path, plan)
    status = {"state": "auditing", "pid": os.getpid(), "started_at": now(),
              "updated_at": now(), "current_job": None, "completed_jobs": []}
    if runtime is not None:
        status["runtime_loader"] = runtime
    atomic_json(status_path, status)
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1", TQDM_MININTERVAL="10")
    script = REPO / "dinov3_global/scripts/crossval_city.py"

    def command(variant, extra):
        return [sys.executable, "-u", str(script), "--config",
                str(REPO / f"dinov3_global/configs/{variant}.yaml"),
                "--out", str(root / variant), *common, *extra]

    # Both configurations must accept the exact same manifest before GPU work.
    for variant in plan["variants"]:
        audit_log = root / f"audit_{variant}.log"
        with audit_log.open("a", encoding="utf-8") as log:
            result = subprocess.run(command(variant, ["--audit-only"]), cwd=REPO, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(state="failed", updated_at=now(), failure_log=str(audit_log),
                          exit_code=result.returncode)
            atomic_json(status_path, status)
            raise RuntimeError(f"Audit failed; see {audit_log}")
    try:
        for fi, variant in plan["order"]:
            if source_hashes() != sources:
                raise RuntimeError("training code/config changed during the comparison")
            job = f"{variant}/fold{fi}"
            completed_path = root / variant / f"fold{fi}" / "fold_report.json"
            if completed_path.exists() and args.resume:
                status["completed_jobs"].append(job)
                continue
            log_path = root / f"{variant}_fold{fi}.log"
            status.update(state="running", current_job=job, updated_at=now(),
                          current_log=str(log_path))
            atomic_json(status_path, status)
            extra = ["--fold-indices", str(fi)] + (["--resume"] if args.resume else [])
            print(f"{now()} Starting {job}; log -> {log_path}", flush=True)
            with log_path.open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command(variant, extra), cwd=REPO, env=env,
                                         stdout=log, stderr=subprocess.STDOUT)
                status.update(child_pid=child.pid, updated_at=now())
                atomic_json(status_path, status)
                rc = child.wait()
            if rc:
                status.update(state="failed", exit_code=rc, updated_at=now())
                atomic_json(status_path, status)
                raise RuntimeError(f"{job} exited {rc}; see {log_path}")
            status["completed_jobs"].append(job)
            status.update(updated_at=now(), child_pid=None)
            atomic_json(status_path, status)
        comparison = {}
        for variant in plan["variants"]:
            comparison[variant] = json.loads((root / variant / "cv_city_report.json").read_text())
        deltas = {f"fold{fi}": comparison["v5"]["folds"][f"fold{fi}"]["metrics"]["mAP"] -
                  comparison["v4_safe"]["folds"][f"fold{fi}"]["metrics"]["mAP"]
                  for fi in range(args.folds)}
        atomic_json(root / "comparison.json", {"variants": comparison,
                    "v5_minus_v4_safe_mAP": deltas,
                    "selection_note": "Validation selects epochs; these scores are not an unbiased test or a paper replication"})
        status.update(state="complete", current_job=None, updated_at=now(), child_pid=None)
        atomic_json(status_path, status)
        print(f"Comparison complete -> {root / 'comparison.json'}", flush=True)
    except BaseException:
        if status["state"] != "failed":
            status.update(state="interrupted", updated_at=now())
            atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
