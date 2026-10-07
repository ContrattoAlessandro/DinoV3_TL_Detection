"""Real CUDA two-view training and exact epoch-boundary resume verification."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.config import load_config
from dinov3_global.data import DTLDGlobalDataset, collate_global, class_frequencies
from dinov3_global.engine import _build_model, fit, load_model, set_seed, dataset_options
from dinov3_global.evaluation import write_json


class SmokePause(Exception):
    pass


class OneEpoch:
    def __init__(self, loader):
        self.loader, self.dataset, self.calls = loader, loader.dataset, 0
        self.batch_size, self.pin_memory = loader.batch_size, loader.pin_memory

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self.calls += 1
        if self.calls > 1:
            raise SmokePause()
        return iter(self.loader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs/generalization_study/smoke")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Smoke output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cfg = copy.deepcopy(load_config(REPO / "configs/generalization_c.yaml"))
    cfg["optim"]["epochs"] = 2
    cfg["optim"]["clean_probe_size"] = 4
    cfg["optim"]["selection_baseline"] = dict(mAP=0.0, NoR_recall=0.0)
    d = cfg["data"]
    full = DTLDGlobalDataset(
        str(REPO / d["label_dir"]),
        str(REPO / d["img_root"]),
        "train",
        target_hw=tuple(d["target_hw"]),
        crop_sides=d["crop_sides"],
        label_crop_sides=d["label_crop_sides"],
        label_policy=d["label_policy"],
        **dataset_options(cfg),
    )
    val_cities = {"Dortmund", "Kassel", "Fulda"}

    def subset(validation, all_items):
        items = [it for it in all_items if (it["city"] in val_cities) == validation]
        selected = [next(it for it in items if it["y"] == y) for y in (0, 1, 2)]
        selected.append(next(it for it in items if it["y"] == 0 and it is not selected[0]))
        return DTLDGlobalDataset(
            str(REPO / d["label_dir"]),
            str(REPO / d["img_root"]),
            "train",
            target_hw=tuple(d["target_hw"]),
            crop_sides=d["crop_sides"],
            label_crop_sides=d["label_crop_sides"],
            label_policy=d["label_policy"],
            train=not validation,
            items=selected,
            **dataset_options(cfg),
        )

    train, val = subset(False, full.items), subset(True, full.items)
    del full
    tl = DataLoader(train, batch_size=4, shuffle=True, collate_fn=collate_global)
    vl = DataLoader(val, batch_size=4, shuffle=False, collate_fn=collate_global)
    log_pi = torch.log(torch.tensor(class_frequencies(train), dtype=torch.float32) + 1e-8)
    dev = torch.device("cuda")
    initial_threads = torch.get_num_threads()

    def fresh():
        torch.set_num_threads(initial_threads)
        set_seed(0)
        model = _build_model(cfg, dev, str(REPO))
        torch.set_num_threads(1)
        set_seed(0)
        return model

    model = fresh()
    calls = []
    handle = model.head.register_forward_pre_hook(
        lambda module, inputs: calls.append(1) if module.training else None
    )
    fit(cfg, model, tl, vl, str(args.out / "uninterrupted"), dev, log_pi)
    handle.remove()
    assert len(calls) == 4, "Two training views must be evaluated in each of two epochs"
    assert all(p.grad is None and not p.requires_grad for p in model.backbone.parameters())
    reference = torch.load(args.out / "uninterrupted/last.pt", weights_only=True, map_location="cpu")
    del model
    torch.cuda.empty_cache()
    model = fresh()
    try:
        fit(cfg, model, OneEpoch(tl), vl, str(args.out / "resumed"), dev, log_pi)
    except SmokePause:
        pass
    state = torch.load(args.out / "resumed/last.pt", weights_only=True, map_location="cpu")
    assert state["epoch"] == 1
    model.head.load_state_dict(state["head"], strict=True)
    fit(
        cfg,
        model,
        tl,
        vl,
        str(args.out / "resumed"),
        dev,
        log_pi,
        start_ep=state["epoch"],
        opt_state=state["opt"],
        scaler_state=state["scaler"],
        ema_state=state["ema"],
        best_init=state["best_score"],
        stale_init=state["stale_epochs"],
        rng_state=state["rng_state"],
        best_selection_key=state["best_selection_key"],
        best_diagnostic_map=state["best_diagnostic_map"],
    )
    resumed = torch.load(args.out / "resumed/last.pt", weights_only=True, map_location="cpu")
    for section in ("head", "ema"):
        assert all(torch.isfinite(t).all() for t in resumed[section].values())
        assert all(torch.equal(t, resumed[section][k]) for k, t in reference[section].items()), section
    model.eval()
    batch = next(iter(vl))
    x = batch["image"].cuda().float().div_(255)
    metadata = {k: batch[k].cuda() for k in ("content_mask", "geometry")}
    with torch.inference_mode(), torch.autocast("cuda"):
        expected = model(x, **metadata)["logits"].cpu()
    del model
    torch.cuda.empty_cache()
    loaded, _, _ = load_model(str(args.out / "resumed/last.pt"), str(REPO), dev)
    # load_model selects EMA, so explicitly load the live head for this roundtrip.
    loaded.head.load_state_dict(resumed["head"], strict=True)
    with torch.inference_mode(), torch.autocast("cuda"):
        assert torch.equal(expected, loaded(x, **metadata)["logits"].cpu())
    history = [
        json.loads(line) for line in (args.out / "uninterrupted/history.jsonl").read_text().splitlines()
    ]
    write_json(
        args.out / "report.json",
        dict(
            status="complete",
            real_frozen_backbone=True,
            target_hw=[720, 1280],
            batch=4,
            training_images=4,
            validation_images=4,
            city_disjoint=True,
            epochs=2,
            both_training_views=True,
            optimizer_and_ema_finite=True,
            checkpoint_safe_load=True,
            resumed_head_and_ema_bit_exact=True,
            peak_gpu_allocated_bytes=max(h["peak_gpu_allocated_bytes"] for h in history),
            elapsed_seconds=time.monotonic() - started,
        ),
    )
    print("Full-resolution two-view CUDA smoke and exact resume passed", flush=True)


if __name__ == "__main__":
    main()
