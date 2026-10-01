"""Compare real GPU updates under original versus optimized loader execution."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dinov3_global.dinov3_global import engine
from dinov3_global.dinov3_global.data_global import DTLDGlobalDataset, class_frequencies
from dinov3_global.scripts.benchmark_throughput import run_case
from dinov3_global.scripts.crossval_city import atomic_json
from dinov3_global.scripts.runtime_loader import validate_runtime


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--runtime", type=Path, required=True)
    args = ap.parse_args()
    root = args.run.resolve()
    runtime = validate_runtime(json.loads(args.runtime.read_text()))
    audit = root / "throughput_audit"
    saved = torch.load(audit / "preserved_last.pt", map_location="cpu", weights_only=True)
    data = saved["cfg"]["data"]
    ds = DTLDGlobalDataset(str(REPO / data["label_dir"]), str(REPO / data["img_root"]), "train",
              crop_sides=data["crop_sides"], label_crop_sides=data["label_crop_sides"],
              target_hw=tuple(data["target_hw"]), label_policy=data["label_policy"], policy_weight=data["policy_weight"])
    cities = json.loads((root / "fold_manifest.json").read_text())["folds"][0]["val_cities"]
    ds.items = [it for it in ds.items if it["city"] not in cities]
    log_pi = torch.log(torch.tensor(class_frequencies(ds), dtype=torch.float32) + 1e-8)
    variants = {}
    for name in ("v4_safe", "v5"):
        torch.set_num_threads(8)
        cfg = saved["cfg"] if name == "v4_safe" else json.loads((root / "v5/effective_config.json").read_text())
        engine.set_seed(0)
        model = engine._build_model(cfg, torch.device("cuda"), str(REPO))
        if name == "v4_safe":
            state = saved
        else:
            engine.set_seed(0)
            head = copy.deepcopy(model.decoder.state_dict())
            optimizer = torch.optim.AdamW(model.decoder_parameters(), lr=cfg["optim"]["lr"],
                                         weight_decay=cfg["optim"]["weight_decay"])
            state = {"head": head, "ema": head, "cfg": cfg, "epoch": 0,
                     "opt": optimizer.state_dict(), "scaler": torch.amp.GradScaler("cuda").state_dict(),
                     "best_score": -1., "stale_epochs": 0, "rng_state": engine.capture_rng_state()}
            del optimizer
        baseline = {"name": name + "_original", "workers": 0, "pin": False, "cpu_threads": 8,
                    "prefetch": None, "persistent_workers": False, "batch": 4, "exploratory_batch": False}
        tuned = {**baseline, "name": name + "_optimized", "workers": runtime["num_workers"],
                 "pin": runtime["pin_memory"], "cpu_threads": runtime["cpu_threads"],
                 "prefetch": runtime["prefetch_factor"]}
        original = run_case(baseline, cfg, state, model, ds, log_pi, audit, 4, 28)
        original_head = {k: v.detach().cpu().clone() for k, v in model.decoder.state_dict().items()}
        optimized = run_case(tuned, cfg, state, model, ds, log_pi, audit, 4, 28)
        max_delta, identical, close = 0., True, True
        for key, tensor in model.decoder.state_dict().items():
            a, b = original_head[key], tensor.detach().cpu()
            identical = identical and torch.equal(a, b)
            close = close and torch.allclose(a, b, atol=1e-6, rtol=1e-5)
            max_delta = max(max_delta, float((a.float() - b.float()).abs().max()))
        item = {"sample_order_identical": original["sampled_paths_sha256"] == optimized["sampled_paths_sha256"],
                "all_rng_states_identical": original["final_rng_sha256"] == optimized["final_rng_sha256"],
                "weights_bit_identical": identical, "weights_close": close, "max_abs_weight_difference": max_delta,
                "microbatches": 32, "optimizer_updates": 8,
                "original_head_sha256": original["final_head_sha256"],
                "optimized_head_sha256": optimized["final_head_sha256"]}
        item["passed"] = item["sample_order_identical"] and item["all_rng_states_identical"] and close
        variants[name] = item
        atomic_json(audit / "gpu_equivalence.json", {"variants": variants,
                    "passed": len(variants) == 2 and all(v["passed"] for v in variants.values()),
                    "runtime_loader": runtime})
        print(f"Equivalence {name}: {item}", flush=True)
        del model, original_head
        torch.cuda.empty_cache()
    if not all(v["passed"] for v in variants.values()):
        raise RuntimeError("Execution equivalence failed; do not apply runtime changes")


if __name__ == "__main__":
    main()
