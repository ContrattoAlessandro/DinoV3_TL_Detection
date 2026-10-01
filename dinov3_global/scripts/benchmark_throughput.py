"""Benchmark the actual two-view training loop on disposable checkpoint copies.

No scientific run artifact is overwritten. The timed loader terminates fit()
before validation/checkpoint writes. Batch-size changes are exploratory only.
"""
import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dinov3_global.dinov3_global import engine
from dinov3_global.dinov3_global.data_global import DTLDGlobalDataset, class_frequencies, collate_global
from dinov3_global.scripts.crossval_city import atomic_json


class BenchmarkFinished(Exception):
    pass


class TimedLoader:
    def __init__(self, loader, warmup, steps):
        self.loader = loader
        self.dataset = loader.dataset
        self.warmup = warmup
        self.steps = steps
        self.measuring = False
        self.wait_seconds = 0
        self.images = 0
        self.events = defaultdict(list)
        self.samples = []
        self.paths = []
        self.sample_stop = threading.Event()

    def sample_resources(self):
        proc = psutil.Process()
        while not self.sample_stop.is_set():
            rss = proc.memory_info().rss
            for child in proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
            row = {"process_tree_rss_mib": rss / 1024**2,
                   "system_ram_percent": psutil.virtual_memory().percent,
                   "system_cpu_percent": psutil.cpu_percent()}
            try:
                output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                           "--format=csv,noheader,nounits"], text=True, timeout=5,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                vals = [float(v.strip()) for v in output.splitlines()[0].split(",")]
                row.update(gpu_util_percent=vals[0], gpu_memory_mib=vals[1], gpu_power_watts=vals[2])
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self.samples.append(row)
            self.sample_stop.wait(1)

    def __len__(self):
        return self.warmup + self.steps

    def __iter__(self):
        start = time.perf_counter()
        iterator = iter(self.loader)
        sampler_thread = None
        try:
            for bi in range(len(self)):
                if bi == self.warmup:
                    torch.cuda.synchronize()
                    self.startup_and_warmup_seconds = time.perf_counter() - start
                    torch.cuda.reset_peak_memory_stats()
                    self.measuring = True
                    self.start = time.perf_counter()
                    sampler_thread = threading.Thread(target=self.sample_resources, daemon=True)
                    sampler_thread.start()
                t0 = time.perf_counter()
                batch = next(iterator)
                self.paths.extend(batch["path"])
                if self.measuring:
                    self.wait_seconds += time.perf_counter() - t0
                    self.images += len(batch["label"])
                yield batch
            torch.cuda.synchronize()
            self.elapsed_seconds = time.perf_counter() - self.start
            self.peak_allocated_mib = torch.cuda.max_memory_allocated() / 1024**2
            self.peak_reserved_mib = torch.cuda.max_memory_reserved() / 1024**2
            raise BenchmarkFinished
        finally:
            self.measuring = False
            self.sample_stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=6)
            del iterator


def measured_call(timer, label, function):
    def wrapped(*args, **kwargs):
        if not timer.measuring:
            return function(*args, **kwargs)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = function(*args, **kwargs)
        end.record()
        timer.events[label].append((start, end))
        return result
    return wrapped


def run_case(case, cfg, state, model, dataset, log_pi, root, warmup, steps):
    print(f"Starting {case}", flush=True)
    torch.set_num_threads(case["cpu_threads"])
    model.decoder.load_state_dict(state["head"])
    torch.cuda.empty_cache()
    current_cfg = copy.deepcopy(cfg)
    current_cfg["optim"].update(batch=case["batch"], accum=16 // case["batch"])
    nw = case["workers"]
    loader = DataLoader(dataset, batch_size=case["batch"], sampler=engine._make_sampler(dataset, cfg["data"].get("sampler", "none")),
                        collate_fn=collate_global, num_workers=nw, pin_memory=case["pin"],
                        **({"prefetch_factor": case["prefetch"], "persistent_workers": False} if nw else {}))
    timed = TimedLoader(loader, warmup, steps)
    functions = ("_to_device", "train_view", "global_loss", "token_losses", "consistency_kl")
    originals = {name: getattr(engine, name) for name in functions}
    forward = model.forward
    backward = torch.Tensor.backward
    try:
        for name, fn in originals.items():
            setattr(engine, name, measured_call(timed, name, fn))
        model.forward = measured_call(timed, "model_forward", forward)
        torch.Tensor.backward = measured_call(timed, "backward", backward)
        try:
            engine.fit(current_cfg, model, timed, None, str(root / "disposable_fit"), torch.device("cuda"), log_pi,
                       start_ep=state["epoch"], opt_state=copy.deepcopy(state["opt"]),
                       scaler_state=copy.deepcopy(state["scaler"]), ema_state=state["ema"],
                       best_init=state["best_score"], stale_init=state["stale_epochs"],
                       rng_state=state["rng_state"])
            raise RuntimeError("Timing sentinel did not stop fit before validation")
        except BenchmarkFinished:
            pass
        result = {**case, "status": "ok", "effective_batch": 16,
                  "images_per_second": timed.images / timed.elapsed_seconds,
                  "seconds_per_batch": timed.elapsed_seconds / steps,
                  "loader_wait_ms_per_batch": timed.wait_seconds * 1000 / steps,
                  "startup_and_warmup_seconds": timed.startup_and_warmup_seconds,
                  "peak_allocated_mib": timed.peak_allocated_mib,
                  "peak_reserved_mib": timed.peak_reserved_mib,
                  "stage_ms_per_batch": {label: sum(a.elapsed_time(b) for a, b in pairs) / steps
                                         for label, pairs in timed.events.items()},
                  "resource_samples": timed.samples}
        head_hash = hashlib.sha256()
        for name, tensor in sorted(model.decoder.state_dict().items()):
            head_hash.update(name.encode())
            head_hash.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        rng = engine.capture_rng_state()
        rng["torch"] = rng["torch"].tolist()
        rng["cuda"] = [s.tolist() for s in rng["cuda"]]
        result.update(final_head_sha256=head_hash.hexdigest(),
                      sampled_paths_sha256=hashlib.sha256(json.dumps(timed.paths).encode()).hexdigest(),
                      final_rng_sha256=hashlib.sha256(json.dumps(rng, sort_keys=True).encode()).hexdigest())
        for name in ("gpu_util_percent", "gpu_power_watts", "process_tree_rss_mib"):
            values = [r[name] for r in timed.samples if name in r]
            result["mean_" + name] = statistics.mean(values) if values else None
        print(f"RESULT {case['name']}: {result['images_per_second']:.2f} images/s; "
              f"data wait {result['loader_wait_ms_per_batch']:.2f} ms/batch; "
              f"VRAM peak {result['peak_allocated_mib']:.0f} MiB", flush=True)
        return result
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {case['name']}", flush=True)
        return {**case, "status": "oom", "error": str(e).splitlines()[0]}
    finally:
        for name, fn in originals.items():
            setattr(engine, name, fn)
        model.forward = forward
        torch.Tensor.backward = backward
        model.zero_grad(set_to_none=True)
        del loader, timed
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=96)
    ap.add_argument("--quick", action="store_true", help="two cases for smoke verification")
    args = ap.parse_args()
    if args.steps % 4 or args.warmup % 4:
        raise ValueError("warmup and measured steps must align with all accumulation windows")
    root = args.run.resolve()
    status = json.loads((root / "status.json").read_text())
    if status["state"] != "paused_for_throughput":
        raise ValueError("Scientific queue must be paused before GPU throughput measurements")
    audit = root / "throughput_audit"
    state = torch.load(audit / "preserved_last.pt", map_location="cpu", weights_only=True)
    cfg = state["cfg"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.backends.cudnn.benchmark = True
    data = cfg["data"]
    full = DTLDGlobalDataset(str(REPO / data["label_dir"]), str(REPO / data["img_root"]), "train",
              crop_sides=data["crop_sides"], label_crop_sides=data["label_crop_sides"],
              target_hw=tuple(data["target_hw"]), label_policy=data["label_policy"], policy_weight=data["policy_weight"])
    cities = json.loads((root / "fold_manifest.json").read_text())["folds"][0]["val_cities"]
    full.items = [it for it in full.items if it["city"] not in cities]
    log_pi = torch.log(torch.tensor(class_frequencies(full), dtype=torch.float32) + 1e-8)
    model = engine._build_model(cfg, torch.device("cuda"), str(REPO))
    default_threads = torch.get_num_threads()
    def case(name, workers=0, pin=False, threads=1, prefetch=2, batch=4):
        return dict(name=name, batch=batch, workers=workers, pin=pin,
                    cpu_threads=threads, prefetch=prefetch if workers else None,
                    persistent_workers=False, exploratory_batch=(batch != 4))
    cases = [case("baseline", threads=default_threads), case("main_threads1"),
             case("threads1_pin", pin=True), case("workers2_pin", workers=2, pin=True),
             case("workers2_prefetch4", workers=2, pin=True, prefetch=4),
             case("workers4_pin", workers=4, pin=True),
             case("workers4_prefetch4", workers=4, pin=True, prefetch=4),
             case("workers6_pin", workers=6, pin=True), case("workers8_pin", workers=8, pin=True)]
    if args.quick:
        cases = cases[:2]
    results = []
    hardware = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                "ram_gib": psutil.virtual_memory().total / 1024**3,
                "physical_cores": psutil.cpu_count(logical=False), "logical_cores": psutil.cpu_count(),
                "checkpoint_epoch": state["epoch"], "warmup_batches": args.warmup,
                "measured_batches": args.steps,
                "method": "Actual engine.fit, full two-view losses/backward/AMP/clip/AdamW/EMA and scalar logging; no validation or scientific checkpoint writes"}
    result_path = audit / ("quick_benchmark.json" if args.quick else "benchmark.json")
    def run(candidate):
        results.append(run_case(candidate, cfg, state, model, full, log_pi, audit, args.warmup, args.steps))
        atomic_json(result_path, {"hardware": hardware, "results": results})
    for candidate in cases:
        run(candidate)
    if not args.quick:
        safe = sorted((r for r in results if r["status"] == "ok" and not r["exploratory_batch"]),
                      key=lambda r: -r["images_per_second"])
        winner = {k: safe[0][k] for k in cases[0]}
        for rep in range(2):
            run({**cases[0], "name": f"baseline_repeat{rep+1}"})
            run({**winner, "name": f"winner_repeat{rep+1}"})
        run({**winner, "name": "batch8_exploratory", "batch": 8, "exploratory_batch": True})
        run({**winner, "name": "batch16_exploratory", "batch": 16, "exploratory_batch": True})
    print(f"Benchmark report -> {result_path}", flush=True)


if __name__ == "__main__":
    main()
