"""Measure validation loading/compute and check raw-logit equivalence."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dinov3_global.dinov3_global import engine
from dinov3_global.dinov3_global.data_global import DTLDGlobalDataset, collate_global
from dinov3_global.dinov3_global.eval_global import fit_temperature
from dinov3_global.scripts.crossval_city import atomic_json
from dinov3_global.scripts.runtime_loader import loader_kwargs


class ValidationTiming:
    def __init__(self, loader):
        self.loader = loader
        self.total_start = None
        self.steady_start = None
    def __iter__(self):
        torch.cuda.synchronize()
        self.total_start = time.perf_counter()
        for i, batch in enumerate(self.loader):
            if i == 4:
                torch.cuda.synchronize()
                self.steady_start = time.perf_counter()
            yield batch
        torch.cuda.synchronize()
        self.end = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()
    root = args.run.resolve(); audit = root / "throughput_audit"
    state = torch.load(audit / "preserved_last.pt", map_location="cpu", weights_only=True)
    cfg = state["cfg"]; data = cfg["data"]
    full = DTLDGlobalDataset(str(REPO / data["label_dir"]), str(REPO / data["img_root"]), "train",
              crop_sides=data["crop_sides"], label_crop_sides=data["label_crop_sides"],
              target_hw=tuple(data["target_hw"]), label_policy=data["label_policy"], policy_weight=data["policy_weight"])
    cities = json.loads((root / "fold_manifest.json").read_text())["folds"][0]["val_cities"]
    full.items = [it for it in full.items if it["city"] in cities][:1024]
    torch.set_num_threads(8)
    model = engine._build_model(cfg, torch.device("cuda"), str(REPO))
    model.decoder.load_state_dict(state["ema"])
    selected = json.loads((audit / "selected_runtime.json").read_text())
    cases = [("original", None, 8), ("selected_workers2", selected, 1),
             ("workers6", {**selected, "num_workers": 6, "prefetch_factor": 2}, 1)]
    results = []; original_logits = None
    for name, runtime, threads in cases:
        torch.set_num_threads(threads)
        loader = DataLoader(full, batch_size=4, collate_fn=collate_global, **loader_kwargs(cfg, runtime))
        timer = ValidationTiming(loader)
        ys, logits, _ = engine.predict_split(model, timer, torch.device("cuda"))
        if original_logits is None:
            original_logits = logits.copy()
        start = time.perf_counter(); temperature = fit_temperature(logits, ys)
        row = {"name": name, "frames": len(ys), "wall_seconds": timer.end - timer.total_start,
               "startup_plus_4_batches_seconds": timer.steady_start - timer.total_start,
               "steady_images_per_second": (len(ys) - 16) / (timer.end - timer.steady_start),
               "raw_logits_bit_identical": bool(np.array_equal(original_logits, logits)),
               "max_raw_logit_difference": float(np.abs(original_logits - logits).max()),
               "temperature": temperature, "calibration_seconds": time.perf_counter() - start}
        results.append(row); atomic_json(audit / "validation_benchmark.json", {"results": results})
        print(row, flush=True)
        del loader, timer
        gc.collect()


if __name__ == "__main__":
    main()
