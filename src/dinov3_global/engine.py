"""Training, EMA validation, checkpoint loading, and reproducible resume for the evidence MIL architecture."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .augment import apply_lamp_erasure, train_view
from .data import (
    DTLDGlobalDataset,
    build_train_val_datasets,
    class_frequencies,
    collate_global,
    state_histogram,
)
from .metrics import _safe_ap, fit_temperature, report, softmax_np, to_jsonable
from .losses import consistency_kl, global_loss, token_losses
from .model import DinoGlobal
from .config import validate_config
from .preprocessing import batch_model_kwargs


def dataset_options(cfg):
    return dict(
        preprocessing=cfg["data"].get("preprocessing", "letterbox"),
        zoom_out_prob=cfg.get("augment", {}).get("zoom_out_prob", 0),
        zoom_out_min=cfg.get("augment", {}).get("zoom_out_min", 0.9),
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def capture_rng_state():
    """Keep epoch-boundary resumes on the same sampling/augmentation stream."""
    ns = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    ns = state["numpy"]
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _loader_args(cfg: dict):
    nw = int(cfg.get("loader", {}).get("num_workers", 0))
    bs = int(cfg["optim"].get("batch", 2))
    pin = bool(cfg.get("loader", {}).get("pin_memory", False)) and nw > 0
    return nw, bs, pin


def _build_model(cfg, dev, repo_root="."):
    validate_config(cfg)
    backbone, decoder = cfg["backbone"], cfg["decoder"]
    local = backbone.get("local_ckpt")
    if local and not os.path.isabs(local):
        local = os.path.join(repo_root, local)
    h, w = cfg["data"]["target_hw"]
    keys = (
        "proj_dim",
        "dropout",
        "topk",
        "logit_scale",
        "context_dropout",
        "context_bound",
    )
    return DinoGlobal(
        hf_id=backbone["hf_id"],
        local_ckpt=local,
        attn_implementation=backbone.get("attn_implementation", "sdpa"),
        dtype=backbone.get("dtype", "float16") if dev.type == "cuda" else "float32",
        grid_hw=(h // 16, w // 16),
        mid_layer=6,
        head_kind=decoder["head"],
        **{k: decoder[k] for k in keys},
    ).to(dev)


def _lr_of(ep: int, warmup: int, epochs: int, base: float) -> float:
    """Cosine schedule with linear warm-up. ``ep`` is 0-based."""
    if ep < warmup:
        return base * (ep + 1) / max(warmup, 1)
    t = (ep - warmup) / max(epochs - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def _make_sampler(ds: DTLDGlobalDataset, mode: str) -> Optional[WeightedRandomSampler]:
    """Sample with replacement using per-example weight frequency**(-p).

    The default protocol disables replacement sampling (mode="none").
    Optional power sampling assigns class mass proportional to frequency**(1-p).
    """
    if mode in (None, "none", ""):
        return None
    ys = np.array([it["y"] for it in ds.items], dtype=int)
    freq = np.bincount(ys, minlength=3).astype(float)
    inv = 1.0 / np.maximum(freq, 1e-6)
    if mode == "sqrt":
        inv = np.sqrt(inv)
    elif mode.startswith("pow"):
        inv = inv ** (float(mode[3:] or 0.0) / 100.0)  # "pow075" -> exponent 0.75
    w = inv[ys] / inv[ys].sum()
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=len(ds), replacement=True)


def build_loaders(cfg: dict, repo_root: str = "."):
    """Train/val loaders. Val is carved session-disjoint from official train.

    The official test split is NEVER loaded here, so per-epoch best.pt selection
    on val mAP cannot optimistically bias the final test report.
    """
    d = cfg["data"]
    ld = os.path.join(repo_root, d["label_dir"])
    ir = os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    tr, va, info = build_train_val_datasets(
        ld,
        ir,
        crop_sides=d.get("crop_sides", 0),
        label_crop_sides=d.get("label_crop_sides", 0),
        target_hw=(th, tw),
        label_policy=d.get("label_policy", "map_to_nor"),
        policy_weight=float(d.get("policy_weight", 1.0)),
        val_frac=float(d.get("val_frac", 0.15)),
        val_seed=int(d.get("val_seed", 0)),
        **dataset_options(cfg),
    )
    print(
        f"train/val split: {info['n_train']} train / {info['n_val']} val "
        f"({info['n_train_seq']}/{info['n_val_seq']} seqs) "
        f"train_hist={info['train_hist']} val_hist={info['val_hist']} "
        f"cities={info['cities']}"
    )
    nw, bs, pin = _loader_args(cfg)
    sampler = _make_sampler(tr, d.get("sampler", "none"))
    common = dict(
        collate_fn=collate_global,
        num_workers=nw,
        pin_memory=pin,
        **({"prefetch_factor": 2, "persistent_workers": True} if nw > 0 else {}),
    )
    tl = DataLoader(tr, shuffle=(sampler is None), sampler=sampler, batch_size=bs, **common)
    vl = DataLoader(va, shuffle=False, batch_size=bs, **common)
    return tl, vl, tr, va, info


def build_test_loader(cfg: dict, repo_root: str = "."):
    """Held-out official test loader. Use only in scripts/evaluate.py, once."""
    d = cfg["data"]
    ld = os.path.join(repo_root, d["label_dir"])
    ir = os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    te = DTLDGlobalDataset(
        ld,
        ir,
        "test",
        train=False,
        crop_sides=d.get("crop_sides", 0),
        label_crop_sides=d.get("label_crop_sides", 0),
        target_hw=(th, tw),
        label_policy=d.get("label_policy", "map_to_nor"),
        **dataset_options(cfg),
    )
    nw, bs, pin = _loader_args(cfg)
    common = dict(
        collate_fn=collate_global,
        num_workers=nw,
        pin_memory=pin,
        **({"prefetch_factor": 2, "persistent_workers": True} if nw > 0 else {}),
    )
    tel = DataLoader(te, shuffle=False, batch_size=bs, **common)
    return tel, te


@torch.no_grad()
def predict_split(model: DinoGlobal, loader: DataLoader, device: torch.device):
    """Return labels, logits, and city, lamp-size, and image-identity metadata."""
    model.eval()
    ys, lgs, cities, sizes, paths, target_sizes = [], [], [], [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(img, **batch_model_kwargs(batch, device))
        ys.append(batch["label"].numpy())
        lgs.append(out["logits"].float().cpu().numpy())
        cities.extend(batch["city"])
        sizes.append(batch["max_lamp_h"].numpy())
        target_sizes.append(
            batch.get("target_lamp_h", torch.full((len(batch["label"]),), float("nan"))).numpy()
        )
        paths.extend(batch["path"])
    return (
        np.concatenate(ys),
        np.concatenate(lgs),
        {
            "cities": cities,
            "max_lamp_h": np.concatenate(sizes),
            "target_lamp_h": np.concatenate(target_sizes),
            "paths": paths,
        },
    )


def _to_device(batch: dict, dev: torch.device) -> torch.Tensor:
    return batch["image"].to(dev, non_blocking=True).float().div_(255.0)


def load_model(ckpt_path, repo_root, dev):
    """Load an DinoGlobal-MIL checkpoint strictly; use EMA weights for inference."""
    state = torch.load(ckpt_path, map_location=dev, weights_only=True)
    cfg = state["cfg"]
    validate_config(cfg)
    model = _build_model(cfg, dev, repo_root)
    model.head.load_state_dict(state["ema"] if "ema" in state else state["head"], strict=True)
    model.eval()
    return model, cfg, state


def _pseudo_flags(ds) -> np.ndarray:
    """pseudo-NoR flag per val item, in loader order (val is never shuffled).
    Used for the live clean-vs-pseudo AP_NoR diagnostic in fit()."""
    items = ds.items if hasattr(ds, "items") else [ds.dataset.items[i] for i in ds.indices]
    return np.array([int(it.get("pseudo_nor", 0)) for it in items], dtype=int)


def _items_of(ds) -> list:
    if hasattr(ds, "items"):
        return ds.items
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):  # torch Subset (smoke path)
        return [ds.dataset.items[i] for i in ds.indices]
    return []


def state_balance_weights(tl, mode, n_state: int = 6):
    """Per-class weights for the state CE. ``sqrt`` = inverse-sqrt frequency
    normalised to mean 1 (the off/unknown-vs-red confusion manufactures false
    evidence on pseudo-NoR frames; green/red dominate the unweighted CE).
    Counts come from the train split's cached frames. None disables."""
    if not mode or mode in ("none", ""):
        return None
    counts = state_histogram(_items_of(tl.dataset), n_state)
    if counts.sum() <= 0:
        return None
    freq = counts / counts.sum()
    w = np.zeros(n_state, dtype=np.float64)
    nz = counts > 0
    if mode == "sqrt":
        w[nz] = 1.0 / np.sqrt(freq[nz])
    elif mode == "inv":
        w[nz] = 1.0 / np.maximum(freq[nz], 1e-6)
    else:
        raise ValueError(f"unknown state_balance mode {mode!r}")
    # normalise over classes that actually occur; a zero-count class (subset
    # smoke runs) must not distort the scale - its weight never applies.
    w[nz] = w[nz] / w[nz].mean()
    w[~nz] = 1.0
    print(f"state_balance[{mode}]: counts={counts.astype(int).tolist()} weights={np.round(w, 2).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def global_balance_weights(log_pi, mode):
    if mode == "none":
        return None
    if mode != "sqrt":
        raise ValueError("Unknown global class weighting")
    pi = log_pi.exp()
    weights = pi.clamp_min(1e-8).rsqrt()
    return weights / (pi * weights).sum()


def pictogram_balance_weights(tl):
    counts = np.zeros(10)
    for item in _items_of(tl.dataset):
        counts += np.bincount(item["frame"].pictogram, minlength=10)
    weights = np.ones(10)
    present = counts > 0
    if present.any():
        weights[present] = 1 / np.sqrt(counts[present] / counts.sum())
        weights[present] /= weights[present].mean()
        weights = np.minimum(weights, 3)
    return torch.tensor(weights, dtype=torch.float32)


def clean_probe_loader(tl, size, seed, out_dir):
    """Frozen class-stratified probe, preferring distinct sessions per class."""
    from .dtld import _seq_of
    from .validation import atomic_json, digest

    if not size or not isinstance(tl.dataset, DTLDGlobalDataset):
        return None
    items = tl.dataset.items
    rng = np.random.default_rng(seed)
    hist = np.bincount([it["y"] for it in items], minlength=3)
    quotas = np.floor(hist / hist.sum() * min(size, len(items))).astype(int)
    for i in np.argsort(-(hist / hist.sum() * min(size, len(items)) - quotas))[
        : min(size, len(items)) - quotas.sum()
    ]:
        quotas[i] += 1
    chosen = []
    for label, quota in enumerate(quotas):
        indices = np.flatnonzero(np.array([it["y"] for it in items]) == label)
        indices = rng.permutation(indices).tolist()
        seen, preferred, remainder = set(), [], []
        for i in indices:
            session = _seq_of(items[i]["entry"]["image_path"])
            if session in seen:
                remainder.append(i)
            else:
                preferred.append(i)
                seen.add(session)
        chosen.extend((preferred + remainder)[:quota])
    chosen.sort()
    ds = copy.copy(tl.dataset)
    ds.items, ds.train = [items[i] for i in chosen], False
    membership = [it["entry"]["image_path"] for it in ds.items]
    probe_record = dict(
        seed=seed,
        paths=membership,
        membership_sha256=digest(membership),
        purpose="Clean training probe, never checkpoint selection",
    )
    path = os.path.join(out_dir, "clean_probe.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as stream:
            if json.load(stream) != probe_record:
                raise ValueError("Clean probe membership changed")
    atomic_json(path, probe_record)
    return DataLoader(
        ds,
        batch_size=tl.batch_size,
        collate_fn=collate_global,
        shuffle=False,
        num_workers=0,
        pin_memory=tl.pin_memory,
    )


def train(
    cfg: dict,
    out_dir: str,
    repo_root: str = ".",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_train: Optional[int] = None,
    max_val: Optional[int] = None,
    resume: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    if not resume and any(
        os.path.exists(os.path.join(out_dir, name)) for name in ("best.pt", "last.pt", "history.jsonl")
    ):
        raise ValueError("Run directory already contains training artifacts; use --resume or a new --out")
    set_seed(int(cfg["optim"].get("seed", 0)))
    dev = torch.device(device)
    tl, vl, tr_ds, va_ds, info = build_loaders(cfg, repo_root)
    freq = class_frequencies(tr_ds)  # from FULL train subset (before truncation)
    log_pi = torch.log(torch.tensor(freq, dtype=torch.float32) + 1e-8)

    if max_train is not None or max_val is not None:  # smoke path
        from torch.utils.data import Subset

        if max_train is not None and max_train < len(tr_ds):
            tr_ds = Subset(tr_ds, list(range(max_train)))
            tl = DataLoader(
                tr_ds, batch_size=tl.batch_size, collate_fn=collate_global, shuffle=True, num_workers=0
            )
        if max_val is not None and max_val < len(va_ds):
            va_ds = Subset(va_ds, list(range(max_val)))
            vl = DataLoader(
                va_ds, batch_size=vl.batch_size, collate_fn=collate_global, shuffle=False, num_workers=0
            )

    model = _build_model(cfg, dev, repo_root)
    start_ep = 0
    opt_state = scaler_state = ema_state = None
    best_init = -1.0
    stale_init = 0
    if resume:
        sd = torch.load(resume, map_location=dev, weights_only=True)
        model.head.load_state_dict(sd["head"])
        ema_state = sd.get("ema")
        opt_state, scaler_state = sd.get("opt"), sd.get("scaler")
        start_ep = int(sd.get("epoch", 0))
        best_init = float(sd.get("best_score", sd.get("metrics", {}).get("mAP", -1.0)))
        stale_init = int(sd.get("stale_epochs", 0))
        # Resuming with a changed architecture/data/loss is a new experiment,
        # not a continuation. Avoid silently mixing incompatible histories.
        for section in ("backbone", "decoder", "data", "loss", "augment"):
            if sd["cfg"].get(section, {}) != cfg.get(section, {}):
                raise ValueError(f"resume config differs in {section}; use a new run")
        print(f"resumed from {resume} at epoch {start_ep} (best so far mAP={best_init:.4f})")
    return fit(
        cfg,
        model,
        tl,
        vl,
        out_dir,
        dev,
        log_pi,
        start_ep=start_ep,
        opt_state=opt_state,
        scaler_state=scaler_state,
        ema_state=ema_state,
        best_init=best_init,
        stale_init=stale_init,
        pseudo_flags=_pseudo_flags(va_ds),
        rng_state=sd.get("rng_state") if resume else None,
    )


def fit(
    cfg: dict,
    model: DinoGlobal,
    tl,
    vl,
    out_dir: str,
    dev: torch.device,
    log_pi: torch.Tensor,
    start_ep: int = 0,
    opt_state=None,
    scaler_state=None,
    ema_state=None,
    best_init: float = -1.0,
    pseudo_flags=None,
    stale_init: int = 0,
    rng_state=None,
):
    """Training loop shared by ordinary training and city cross-validation."""
    os.makedirs(out_dir, exist_ok=True)
    o = cfg["optim"]
    opt = torch.optim.AdamW(
        model.head_parameters(), lr=float(o.get("lr", 1e-4)), weight_decay=float(o.get("weight_decay", 0.05))
    )
    epochs = int(o.get("epochs", 30))
    warmup = int(o.get("warmup_epochs", 3))
    accum = int(o.get("accum", 8))
    use_amp = bool(o.get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    ema = copy.deepcopy(model.head)
    if ema_state is not None:
        ema.load_state_dict(ema_state)
    ema_decay = float(o.get("ema_decay", 0.999))
    grad_clip = float(o.get("grad_clip", 1.0))

    lcfg = cfg.get("loss", {})
    tau_la = float(lcfg.get("tau_la", 1.0))
    smoothing = float(lcfg.get("label_smoothing", 0.05))
    w_lamp = float(lcfg.get("token_lamp", 0.3))
    w_state = float(lcfg.get("token_state", 0.2))
    w_rel = float(lcfg.get("token_rel", 0.5))
    w_dir = float(lcfg.get("token_dir", 0.0))
    w_consist = float(lcfg.get("consist", 0.0))
    rel_pw = float(lcfg.get("rel_pos_weight", 100.0))
    rel_gamma = float(lcfg.get("rel_focal_gamma", 0.0))
    state_boost = float(lcfg.get("state_rel_boost", 0.0))
    state_w = state_balance_weights(tl, lcfg.get("state_balance", "none"))
    if state_w is not None:
        state_w = state_w.to(dev)
    class_w = global_balance_weights(log_pi, lcfg.get("class_balance", "none"))
    class_w = class_w.to(dev) if class_w is not None else None
    w_pictogram = float(lcfg.get("token_pictogram", 0))
    pic_w = pictogram_balance_weights(tl).to(dev) if w_pictogram else None
    acfg = cfg.get("augment", {})
    p_erase = float(acfg.get("p_erase", 0.0))
    view_kw = dict(
        strength=float(acfg.get("strength", 1.0)),
        crop=False,
        photometric_enabled=bool(acfg.get("photometric", True)),
        degradations_enabled=bool(acfg.get("blur", True)),
        per_image=bool(acfg.get("per_image", True)),
        clean_prob=float(acfg.get("clean_prob", 0.0)),
        blend_min=float(acfg.get("blend_min", 0.0)),
    )
    branch_supervision = bool(lcfg.get("per_branch_supervision", False))
    supervise_view2 = bool(lcfg.get("supervise_view2", False))
    consist_warmup = int(lcfg.get("consist_warmup_epochs", 0))
    use_token = True

    def _lr(ep: int) -> float:
        return _lr_of(ep, warmup, epochs, float(o.get("lr", 1e-4)))

    def _optim_step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.head_parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():  # EMA tracks EVERY optimizer step (incl. the
            for pe, pm in zip(ema.parameters(), model.head.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)

    patience = int(o.get("early_stop_patience", 0))  # 0 = disabled
    stale = stale_init
    best_map, best_state = best_init, None
    selection_metric = o.get("early_stop_metric", "mAP")
    if selection_metric != "mAP":
        raise ValueError("New DinoGlobal-MIL runs select checkpoints by validation mAP")
    probe_loader = clean_probe_loader(tl, int(o.get("clean_probe_size", 0)), int(o.get("seed", 0)), out_dir)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if rng_state is not None:
        restore_rng_state(rng_state)

    for ep in range(start_ep, epochs):
        epoch_started = time.monotonic()
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
        for g in opt.param_groups:
            g["lr"] = _lr(ep)
        model.train()
        tot = {"total": 0.0, "global": 0.0, "token": 0.0, "consist": 0.0}
        components = {key: 0.0 for key in ("lamp", "state", "rel", "dir", "pictogram")}
        nstep, nbatch = 0, 0
        opt.zero_grad(set_to_none=True)
        nbatch_total = len(tl) if hasattr(tl, "__len__") else None
        for bi, batch in enumerate(tqdm(tl, desc=f"ep{ep + 1}/{epochs} train")):
            x = _to_device(batch, dev)
            y = batch["label"].to(dev, non_blocking=True)
            w = batch["weight"].to(dev, non_blocking=True)
            lamp_t = batch["lamp_tgt"].to(dev, non_blocking=True)
            valid_t = batch["valid"].to(dev, non_blocking=True)
            state_t = batch["state_tgt"].to(dev, non_blocking=True)
            rel_t = batch["rel_tgt"].to(dev, non_blocking=True)
            dir_t = batch.get("dir_tgt")
            dir_t = dir_t.to(dev, non_blocking=True) if dir_t is not None else None
            metadata = batch_model_kwargs(batch, dev)
            attribute_masks = {
                key: batch[key].to(dev, non_blocking=True)
                for key in ("state_valid", "rel_valid", "dir_valid", "pictogram_valid")
                if key in batch
            }
            instance_id = batch["instance_id"].to(dev) if "instance_id" in batch else None
            instance_weights = {
                key: batch[key + "_instance_weight"].to(dev)
                for key in ("state", "rel", "dir", "pictogram")
                if key + "_instance_weight" in batch
            }
            pictogram_t = batch["pictogram_tgt"].to(dev) if "pictogram_tgt" in batch else None
            if p_erase > 0:
                # manufactured NoR: erase ALL lamps of a frame -> genuine
                # "no relevant lamp" case, label and token targets follow the
                # pixels. Applied before the views so both consistency views
                # see the same (frame, label) pair.
                before_erasure = lamp_t.reshape(len(x), -1).sum(1)
                x, y, w, lamp_t, valid_t, rel_t = apply_lamp_erasure(
                    x, lamp_t, y, w, lamp_t, valid_t, rel_t, p_erase
                )
                if "content_mask" in metadata:
                    erased = (before_erasure > 0) & (lamp_t.reshape(len(x), -1).sum(1) == 0)
                    valid_t = valid_t & metadata["content_mask"]
                    for key in attribute_masks:
                        attribute_masks[key] = attribute_masks[key].clone()
                        attribute_masks[key][erased] = metadata["content_mask"][erased]
            if class_w is not None:
                w = w * class_w[y]
            # view 1 carries the token-level aux losses, so it must NOT be
            # geometrically cropped: the token targets are rasterised on the
            # uncropped frame (measured: 0.780 -> 0.005 lamp-hit-rate when cropped).
            x1 = train_view(x, **view_kw)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(x1, **metadata)
                g_loss = global_loss(out["logits"], y, log_pi, tau_la, smoothing, w)
                loss = g_loss
                t_loss = torch.zeros((), device=dev)
                if use_token:
                    # note: lamp-erasure zeroes lamp/rel targets above, and the
                    # state/dir losses mask on lamp-positive tokens, so erased
                    # frames automatically lose their state/dir supervision.
                    supervised_maps = out.get("branch_maps") if branch_supervision else None
                    supervised_maps = supervised_maps or [out["maps"]]
                    token_components = [
                        token_losses(
                            mp,
                            lamp_t,
                            valid_t,
                            state_t,
                            rel_t,
                            w_lamp,
                            w_state,
                            w_rel,
                            float(lcfg.get("lamp_pos_weight", 10.0)),
                            rel_pw,
                            rel_focal_gamma=rel_gamma,
                            w_dir=w_dir,
                            dir_tgt=dir_t,
                            state_weights=state_w,
                            state_rel_boost=state_boost,
                            rel_lamp_weight=lcfg.get("rel_lamp_weight"),
                            valid_masks=attribute_masks or None,
                            instance_id=instance_id,
                            reduction=lcfg.get("attribute_reduction", "token"),
                            pictogram_tgt=pictogram_t,
                            pictogram_weights=pic_w,
                            w_pictogram=w_pictogram,
                            instance_weights=instance_weights,
                        )
                        for mp in supervised_maps
                    ]
                    t_loss = torch.stack([part["total"] for part in token_components]).mean()
                    for key in components:
                        components[key] += float(
                            torch.stack(
                                [part.get(key, t_loss.new_zeros(())) for part in token_components]
                            ).mean()
                        )
                    loss = loss + t_loss
                c_loss = torch.zeros((), device=dev)
                if w_consist > 0 or supervise_view2:
                    # Both views retain the full frame: a crop can remove the
                    # only relevant lamp and invalidate an existential label.
                    x2 = train_view(x, **view_kw)
                    out2 = model(x2, **metadata)
                    if supervise_view2:
                        g2 = global_loss(out2["logits"], y, log_pi, tau_la, smoothing, w)
                        loss = loss - g_loss + (g_loss + g2) * 0.5
                        g_loss = (g_loss + g2) * 0.5
                    if w_consist > 0:
                        c_loss = consistency_kl(out["logits"], out2["logits"])
                        ramp = min(1.0, (ep + 1) / max(consist_warmup, 1))
                        loss = loss + w_consist * ramp * c_loss
                # average over the accumulation window; the leftover window at
                # epoch end must divide by its ACTUAL size (dividing a partial
                # window by `accum` under-weighted its gradient).
                den = accum
                if nbatch_total is not None:
                    tail = nbatch_total % accum
                    if tail and bi >= nbatch_total - tail:
                        den = tail
                loss = loss / den
            scaler.scale(loss).backward()
            tot["total"] += float(loss.detach()) * den
            tot["global"] += float(g_loss.detach())
            tot["token"] += float(t_loss.detach())
            tot["consist"] += float(c_loss.detach())
            nbatch += 1
            if (bi + 1) % accum == 0:
                _optim_step()
                nstep += 1
        if (nbatch % accum) != 0:
            _optim_step()

        # ---- val (held-out train sequences) on EMA head; test untouched ----
        live = copy.deepcopy(model.head.state_dict())
        model.head.load_state_dict(ema.state_dict())
        ys, lgs, meta = predict_split(model, vl, dev)
        probe_report = None
        if probe_loader is not None:
            py, pl, pm = predict_split(model, probe_loader, dev)
            probe_report = report(py, pl, cities=pm["cities"])
        model.head.load_state_dict(live)
        rep = report(
            ys, lgs, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"], target_lamp_h=meta["target_lamp_h"]
        )
        met = rep["metrics"]
        # slice mAP is only comparable to overall mAP when the slice contains
        # all 3 classes (mAP averages over the classes present in the slice,
        # so a 2-class city reports an inflated number). The gate uses 3-class
        # slices only; n_classes travels with each slice for the reports.
        city_m = {
            c: m["mAP"]
            for c, m in rep.get("slices", {}).get("city", {}).items()
            if m.get("n_classes", 3) >= 3
        }
        worst_city = min(city_m.values()) if city_m else float("nan")
        print(
            f"ep{ep + 1} loss={tot['total'] / max(nbatch, 1):.4f} "
            f"(g={tot['global'] / max(nbatch, 1):.4f} t={tot['token'] / max(nbatch, 1):.4f} "
            f"c={tot['consist'] / max(nbatch, 1):.4f}) "
            f"AP_RR={met['AP_RR']:.4f} AP_RG={met['AP_RG']:.4f} AP_NoR={met['AP_NoR']:.4f} "
            f"mAP={met['mAP']:.4f} acc_bal={met['acc_bal']:.4f} worst_city_mAP={worst_city:.4f}"
        )
        if city_m:
            print("  city mAP: " + " ".join(f"{c}={v:.3f}" for c, v in sorted(city_m.items())))
        # live failure-mode view: per-class operating point + clean vs pseudo NoR
        _pred = lgs.argmax(1)
        parts = []
        for ci, cn in enumerate(("RR", "RG", "NoR")):
            m = ys == ci
            rec = float((_pred[m] == ci).mean()) if m.any() else float("nan")
            hit = _pred == ci
            prec = float((ys[hit] == ci).mean()) if hit.any() else float("nan")
            parts.append(f"{cn} r{rec:.2f}/p{prec:.2f}")
        if pseudo_flags is not None:
            pf = np.asarray(pseudo_flags)[: len(ys)]
            for nm, msk in (("cleanNoR", (ys == 2) & (pf == 0)), ("pseudoNoR", (ys == 2) & (pf == 1))):
                if int(msk.sum()) >= 10:
                    idx = np.flatnonzero(msk | (ys != 2))
                    ap = _safe_ap((ys[idx] == 2).astype(int), softmax_np(lgs[idx])[:, 2])
                    parts.append(f"AP_{nm}={ap:.3f}")
        print("  classes: " + "  ".join(parts))

        # Select checkpoints using raw T=1 validation metrics. Save the fitted
        # validation temperature and calibrated report separately.
        T = fit_temperature(lgs, ys)
        rep_cal = report(ys, lgs, T=T, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
        met["temperature"] = T  # divide logits by T for calibrated probs
        met["ece_calibrated"] = rep_cal["metrics"]["ece"]

        score = met["mAP"]
        improved = score > best_map
        if not math.isfinite(score):
            raise ValueError(f"selection metric {selection_metric} is undefined on validation")
        if improved:
            best_map = score
            stale = 0
        else:
            stale += 1
        state = {
            "head": copy.deepcopy(model.head.state_dict()),
            "ema": copy.deepcopy(ema.state_dict()),
            "cfg": cfg,
            "epoch": ep + 1,
            "metrics": to_jsonable(met),
            "report": to_jsonable(rep),
            "report_calibrated": to_jsonable(rep_cal),
            "log_pi": log_pi.tolist(),
            "best_score": best_map,
            "selection_metric": selection_metric,
            "stale_epochs": stale,
        }
        if improved:
            best_state = state
            torch.save(state, os.path.join(out_dir, "best.pt"))
        last = dict(state)
        last.update(
            {"opt": opt.state_dict(), "scaler": scaler.state_dict(), "rng_state": capture_rng_state()}
        )
        torch.save(last, os.path.join(out_dir, "last.pt"))
        with open(os.path.join(out_dir, "history.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    to_jsonable(
                        {
                            "epoch": ep + 1,
                            **{k: v for k, v in met.items() if isinstance(v, (int, float))},
                            "worst_city_mAP": worst_city,
                            "selection_score": score,
                            "training_losses": {k: v / max(nbatch, 1) for k, v in tot.items()},
                            "attribute_losses": {k: v / max(nbatch, 1) for k, v in components.items()},
                            "clean_probe": to_jsonable(probe_report),
                            "validation": to_jsonable(rep),
                            "elapsed_seconds": time.monotonic() - epoch_started,
                            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(dev)
                            if dev.type == "cuda"
                            else 0,
                        }
                    )
                )
                + "\n"
            )
        if patience and stale >= patience:
            print(
                f"early stop: val {selection_metric} flat for {stale} epochs "
                f"(patience {patience}); best {best_map:.4f}"
            )
            # Preserve the deliberate early stop when cross-validation resumes.
            with open(os.path.join(out_dir, "STOPPED"), "w") as f:
                f.write(f"epoch {ep + 1} stale {stale} best {best_map:.4f}\n")
            break

    print(
        f"Selected validation EMA {selection_metric}={best_map:.4f}; checkpoint: {os.path.join(out_dir, 'best.pt')}"
    )
    return best_state
