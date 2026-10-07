"""Measure both complete models with identical CUDA inputs and batch size."""

import argparse
import gc
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.engine import load_model
from dinov3_global.inference import preprocess
from dinov3_global.evaluation import write_json, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=REPO / "runs/v6_axial_pilot")
    args = parser.parse_args()
    torch.set_num_threads(1)
    paths = np.load(REPO / "runs/v5/city_cv/fold0/val_predictions.npz")["paths"][:4]
    x = torch.stack([preprocess(str(p), 0, (720, 1280))[0] for p in paths]).cuda().float().div_(255)
    models = {}
    for name, path in (
        ("v5_fold0", REPO / "runs/v5/city_cv/fold0/best.pt"),
        ("v6_axial_fold0", args.pilot / "city_cv/fold0/best.pt"),
    ):
        model, _, _ = load_model(str(path), str(REPO), torch.device("cuda"))
        with torch.inference_mode(), torch.autocast("cuda"):
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            elapsed = []
            for _ in range(30):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(x)
                end.record()
                torch.cuda.synchronize()
                elapsed.append(start.elapsed_time(end))
        latency = float(np.median(elapsed))
        models[name] = dict(head_parameters=sum(p.numel() for p in model.head.parameters()),
                            batch_latency_ms=latency, images_per_second=4000 / latency,
                            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                            checkpoint_sha256=sha256_file(path))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.pilot / "benchmark.json", dict(
        status="complete", batch=4, target_hw=[720, 1280], warmup_batches=5, measured_batches=30,
        latency="median CUDA-event time; complete frozen encoder and head; preprocessing excluded",
        autocast_dtype=str(torch.get_autocast_dtype("cuda")), gpu=torch.cuda.get_device_name(0),
        models=models, completed_at_unix=time.time(),
    ))
    print(models)


if __name__ == "__main__":
    main()
