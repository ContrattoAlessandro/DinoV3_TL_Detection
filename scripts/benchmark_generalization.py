"""Gate the completed RGB-to-probabilities pipeline before expensive pilots."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.config import load_config
from dinov3_global.engine import _build_model
from dinov3_global.evaluation import write_json, sha256_file
from dinov3_global.preprocessing import preprocess_for_config


def benchmark(out, iterations=30):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dev = torch.device("cuda")
    files = sorted((REPO / "datasets/DTLD_jpg/train").glob("*.jpg"))[:16]
    if len(files) < 4:
        raise ValueError("Native DTLD RGB inputs missing")
    results = {}
    configs = {
        "v5": torch.load(REPO / "runs/v5/city_cv/fold0/best.pt", map_location="cpu", weights_only=True)["cfg"]
    }
    configs.update({run: load_config(REPO / f"configs/generalization_{run.lower()}.yaml") for run in "ACD"})
    for name, cfg in configs.items():
        model = _build_model(cfg, dev, str(REPO)).eval()
        result = dict(
            target_hw=cfg["data"]["target_hw"],
            head_parameters=sum(p.numel() for p in model.head.parameters()),
        )
        for batch_size in (1, 4):
            times = []
            for i in range(iterations + 5):
                torch.cuda.synchronize()
                started = time.perf_counter()
                prepared = [
                    preprocess_for_config(
                        files[(i * batch_size + j) % len(files)],
                        cfg,
                        cfg["data"].get("crop_sides", 0) if name == "v5" else 0,
                    )
                    for j in range(batch_size)
                ]
                metadata = {key: torch.stack([p[2][key] for p in prepared]).to(dev) for key in prepared[0][2]}
                images = torch.stack([p[0] for p in prepared]).to(dev).float().div_(255)
                with torch.inference_mode(), torch.autocast("cuda"):
                    model(images, **metadata)["logits"].softmax(-1).float().cpu().numpy()
                torch.cuda.synchronize()
                if i >= 5:
                    times.append((time.perf_counter() - started) * 1000)
            result[f"batch{batch_size}"] = dict(
                median_ms=float(np.median(times)), p95_ms=float(np.quantile(times, 0.95)), samples=times
            )
        results[name] = result
        del model
        torch.cuda.empty_cache()
    for name, result in results.items():
        result["latency_ratios"] = {
            f"batch{b}": result[f"batch{b}"]["median_ms"] / results["v5"][f"batch{b}"]["median_ms"]
            for b in (1, 4)
        }
        result["eligible"] = max(result["latency_ratios"].values()) <= 2
    record = dict(
        status="complete",
        gpu=torch.cuda.get_device_name(),
        scope="JPEG decode, resize/letterbox, metadata, host/device transfers, complete forward, softmax and CPU result",
        iterations=iterations,
        budget_ratio=2.0,
        input_sha256={p.name: sha256_file(p) for p in files},
        models=results,
    )
    write_json(out, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise ValueError("Benchmark already exists")
    print(json.dumps(benchmark(args.out, args.iterations), indent=2))
