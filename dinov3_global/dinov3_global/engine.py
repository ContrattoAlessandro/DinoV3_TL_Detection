"""Training engine for DinoGlobal (v2).

Changes vs exp1 engine:
- uint8 batches -> float on GPU (exp1 died of host-RAM OOM on float32 decode)
- GPU augmentation (augment.py) + optional two-view consistency loss
- logit-adjusted CE (balanced-error consistent) + token-level aux losses from
  DTLD boxes (decoder.head == "mil")
- EMA on the head, grad clipping, resume, temperature-scaled calibration
- val every epoch on EMA: global metrics + per-city slices (worst-city tracked
  as the generalisation gate). Official test NEVER loads here.

12GB notes: batch=2, accum 8 (eff 16) fits ViT-S+/16 @1280x720 with sdpa fp16.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .augment import apply_lamp_erasure, train_view
from .data_global import (DTLDGlobalDataset, build_train_val_datasets,
                          class_frequencies, collate_global, state_histogram)
from .eval_global import _safe_ap, fit_temperature, report, softmax_np, to_jsonable
from .losses import consistency_kl, global_loss, token_losses
from .model import DinoGlobal


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader_args(cfg: dict):
    nw = int(cfg.get("loader", {}).get("num_workers", 0))
    bs = int(cfg["optim"].get("batch", 2))
    pin = bool(cfg.get("loader", {}).get("pin_memory", False)) and nw > 0
    return nw, bs, pin


def _build_model(cfg: dict, dev: torch.device, repo_root: str = ".") -> DinoGlobal:
    m, d = cfg["backbone"], cfg["decoder"]
    _ckpt = m.get("local_ckpt", None)
    if _ckpt and not os.path.isabs(_ckpt):
        _ckpt = os.path.join(repo_root, _ckpt)
    head = d.get("head", "mil")
    patch = int(m.get("patch", 16))
    th, tw = cfg.get("data", {}).get("target_hw", [720, 1280])
    grid_hw = (int(th) // patch, int(tw) // patch)
    kw = dict(proj_dim=d.get("proj_dim", 256), dropout=d.get("dropout", 0.1))
    if head == "attn":
        kw.update(depth=d.get("depth", 2), heads=d.get("heads", 4),
                  ff_dim=d.get("ff_dim", 1024))
    else:
        kw.update(n_branches=d.get("n_branches", 2),
                  tau_init=tuple(d.get("tau_init", [0.05, 0.3])),
                  logit_scale=d.get("logit_scale", 4.0),
                  pool=d.get("pool", "lse"),
                  topk=tuple(d.get("topk", [1, 2, 4])),
                  nor_sees_evidence=bool(d.get("nor_sees_evidence", False)),
                  dir_head=bool(d.get("dir_head", False)),
                  cls_modulates_evidence=bool(d.get("cls_modulates_evidence", False)))
    return DinoGlobal(hf_id=m.get("hf_id", "facebook/dinov3-vits16plus-pretrain-lvd1689m"),
                      attn_implementation=m.get("attn_implementation", "sdpa"),
                      dtype=m.get("dtype", "float16") if dev.type == "cuda" else "float32",
                      head=head, local_ckpt=_ckpt,
                      grid_hw=grid_hw,
                      mid_layer=(int(m["mid_layer"]) if m.get("mid_layer") is not None
                                 else None),
                      **kw).to(dev)


def _lr_of(ep: int, warmup: int, epochs: int, base: float) -> float:
    """Cosine schedule with linear warm-up. ``ep`` is 0-based."""
    if ep < warmup:
        return base * (ep + 1) / max(warmup, 1)
    t = (ep - warmup) / max(epochs - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * t))


def _make_sampler(ds: DTLDGlobalDataset, mode: str) -> Optional[WeightedRandomSampler]:
    """Class-rebalancing sampler. Per-sample weight = freq**-p, so the resulting
    class share is ``freq**(1-p)`` normalised - note this depends on the class
    SIZES, not on freq alone. Measured shares on the exp2 train histogram
    (RR 8082 / RG 14863 / NoR 1202, natural = 33.5 / 61.6 / 5.0 %):
        none  -> RR 33.5  RG 61.6  NoR  4.98 %
        sqrt  -> RR 36.5  RG 49.5  NoR 14.07 %   (p = 0.50)
        pow025-> RR 36.5  RG 55.1  NoR  8.40 %   (p = 0.25, barely rebalances)
        pow075-> RR 35.3  RG 42.3  NoR 22.30 %   (p = 0.75, the v3 setting)
        pow100-> 33.3 / 33.3 / 33.3 %             (p = 1.0, full inverse)
    """
    if mode in (None, "none", ""):
        return None
    ys = np.array([it["y"] for it in ds.items], dtype=int)
    freq = np.bincount(ys, minlength=3).astype(float)
    inv = 1.0 / np.maximum(freq, 1e-6)
    if mode == "sqrt":
        inv = np.sqrt(inv)
    elif mode.startswith("pow"):
        inv = inv ** (float(mode[3:] or 0.0) / 100.0)   # "pow075" -> exponent 0.75
    w = inv[ys] / inv[ys].sum()
    return WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                 num_samples=len(ds), replacement=True)


def build_loaders(cfg: dict, repo_root: str = "."):
    """Train/val loaders. Val is carved session-disjoint from official train.

    The official test split is NEVER loaded here, so per-epoch best.pt selection
    on val mAP cannot optimistically bias the final test report.
    """
    d, o = cfg["data"], cfg["optim"]
    ld = os.path.join(repo_root, d["label_dir"])
    ir = os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    tr, va, info = build_train_val_datasets(
        ld, ir, crop_sides=d.get("crop_sides", 114),
        label_crop_sides=d.get("label_crop_sides", 114),
        target_hw=(th, tw),
        label_policy=d.get("label_policy", "map_to_nor"),
        policy_weight=float(d.get("policy_weight", 0.3)),
        val_frac=float(d.get("val_frac", 0.15)), val_seed=int(d.get("val_seed", 0)))
    print(f"train/val split: {info['n_train']} train / {info['n_val']} val "
          f"({info['n_train_seq']}/{info['n_val_seq']} seqs) "
          f"train_hist={info['train_hist']} val_hist={info['val_hist']} "
          f"cities={info['cities']}")
    nw, bs, pin = _loader_args(cfg)
    sampler = _make_sampler(tr, d.get("sampler", "none"))
    common = dict(collate_fn=collate_global, num_workers=nw,
                  pin_memory=pin, **({"prefetch_factor": 2, "persistent_workers": True}
                                     if nw > 0 else {}))
    tl = DataLoader(tr, shuffle=(sampler is None), sampler=sampler,
                    batch_size=bs, **common)
    vl = DataLoader(va, shuffle=False, batch_size=bs, **common)
    return tl, vl, tr, va, info


def build_test_loader(cfg: dict, repo_root: str = "."):
    """Held-out official test loader. Use only in scripts/evaluate.py, once."""
    d = cfg["data"]
    ld = os.path.join(repo_root, d["label_dir"])
    ir = os.path.join(repo_root, d["img_root"])
    th, tw = d.get("target_hw", [720, 1280])
    te = DTLDGlobalDataset(ld, ir, "test", train=False,
                           crop_sides=d.get("crop_sides", 114),
                           label_crop_sides=d.get("label_crop_sides", 114),
                           target_hw=(th, tw),
                           label_policy=d.get("label_policy", "map_to_nor"))
    nw, bs, pin = _loader_args(cfg)
    common = dict(collate_fn=collate_global, num_workers=nw,
                  pin_memory=pin, **({"prefetch_factor": 2, "persistent_workers": True}
                                     if nw > 0 else {}))
    tel = DataLoader(te, shuffle=False, batch_size=bs, **common)
    return tel, te


@torch.no_grad()
def predict_split(model: DinoGlobal, loader: DataLoader, device: torch.device):
    """Returns (ys, logits, meta) with meta = dict(cities, max_lamp_h, paths)."""
    model.eval()
    ys, lgs, cities, sizes, paths = [], [], [], [], []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(img)
        ys.append(batch["label"].numpy())
        lgs.append(out["logits"].float().cpu().numpy())
        cities.extend(batch["city"])
        sizes.append(batch["max_lamp_h"].numpy())
        paths.extend(batch["path"])
    return (np.concatenate(ys), np.concatenate(lgs),
            {"cities": cities, "max_lamp_h": np.concatenate(sizes), "paths": paths})


def _to_device(batch: dict, dev: torch.device) -> torch.Tensor:
    return batch["image"].to(dev, non_blocking=True).float().div_(255.0)


def load_model(ckpt_path: str, repo_root: str, dev: torch.device):
    """Load a best.pt/last.pt -> (model, cfg, state_dict). Head kind is inferred
    from the state dict so legacy exp1 attn checkpoints still load."""
    sd = torch.load(ckpt_path, map_location=dev)
    cfg = sd["cfg"]
    state = sd.get("ema", sd.get("head", sd.get("decoder", sd)))
    head = "mil" if any(k.startswith("branches") for k in state) else "attn"
    cfg.setdefault("decoder", {})["head"] = head
    model = _build_model(cfg, dev, repo_root)
    model.decoder.load_state_dict(state, strict=True)
    model.eval()
    return model, cfg, sd


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
    print(f"state_balance[{mode}]: counts={counts.astype(int).tolist()} "
          f"weights={np.round(w, 2).tolist()}")
    return torch.tensor(w, dtype=torch.float32)


def train(cfg: dict, out_dir: str, repo_root: str = ".",
          device: str = "cuda" if torch.cuda.is_available() else "cpu",
          max_train: Optional[int] = None, max_val: Optional[int] = None,
          resume: Optional[str] = None):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(int(cfg["optim"].get("seed", 0)))
    dev = torch.device(device)
    tl, vl, tr_ds, va_ds, info = build_loaders(cfg, repo_root)
    freq = class_frequencies(tr_ds)  # from FULL train subset (before truncation)
    log_pi = torch.log(torch.tensor(freq, dtype=torch.float32) + 1e-8)

    if max_train is not None or max_val is not None:  # smoke path
        from torch.utils.data import Subset
        if max_train is not None and max_train < len(tr_ds):
            tr_ds = Subset(tr_ds, list(range(max_train)))
            tl = DataLoader(tr_ds, batch_size=tl.batch_size, collate_fn=collate_global,
                            shuffle=True, num_workers=0)
        if max_val is not None and max_val < len(va_ds):
            va_ds = Subset(va_ds, list(range(max_val)))
            vl = DataLoader(va_ds, batch_size=vl.batch_size, collate_fn=collate_global,
                            shuffle=False, num_workers=0)

    model = _build_model(cfg, dev, repo_root)
    start_ep = 0
    opt_state = scaler_state = ema_state = None
    best_init = -1.0
    if resume:
        sd = torch.load(resume, map_location=dev)
        model.decoder.load_state_dict(sd["head"])
        ema_state = sd.get("ema")
        opt_state, scaler_state = sd.get("opt"), sd.get("scaler")
        start_ep = int(sd.get("epoch", 0))
        best_init = float(sd.get("metrics", {}).get("mAP", -1.0))
        print(f"resumed from {resume} at epoch {start_ep} (best so far mAP={best_init:.4f})")
    return fit(cfg, model, tl, vl, out_dir, dev, log_pi, start_ep=start_ep,
               opt_state=opt_state, scaler_state=scaler_state, ema_state=ema_state,
               best_init=best_init, pseudo_flags=_pseudo_flags(va_ds))


def fit(cfg: dict, model: DinoGlobal, tl, vl, out_dir: str,
        dev: torch.device, log_pi: torch.Tensor, start_ep: int = 0,
        opt_state=None, scaler_state=None, ema_state=None,
        best_init: float = -1.0, pseudo_flags=None):
    """Training loop (shared by train() and scripts/crossval_city.py)."""
    os.makedirs(out_dir, exist_ok=True)
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.decoder_parameters(), lr=float(o.get("lr", 1e-4)),
                            weight_decay=float(o.get("weight_decay", 0.05)))
    epochs = int(o.get("epochs", 30))
    warmup = int(o.get("warmup_epochs", 3))
    accum = int(o.get("accum", 8))
    use_amp = bool(o.get("amp", True)) and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    ema = copy.deepcopy(model.decoder)
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
    p_erase = float(cfg.get("augment", {}).get("p_erase", 0.0))
    use_token = cfg.get("decoder", {}).get("head", "mil") == "mil"

    def _lr(ep: int) -> float:
        return _lr_of(ep, warmup, epochs, float(o.get("lr", 1e-4)))

    def _optim_step():
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.decoder_parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():  # EMA tracks EVERY optimizer step (incl. the
            for pe, pm in zip(ema.parameters(), model.decoder.parameters()):
                pe.mul_(ema_decay).add_(pm, alpha=1 - ema_decay)

    patience = int(o.get("early_stop_patience", 0))  # 0 = disabled
    stale = 0
    best_map, best_state = best_init, None
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True

    for ep in range(start_ep, epochs):
        for g in opt.param_groups:
            g["lr"] = _lr(ep)
        model.train()
        tot = {"total": 0.0, "global": 0.0, "token": 0.0, "consist": 0.0}
        nstep, nbatch = 0, 0
        opt.zero_grad(set_to_none=True)
        nbatch_total = len(tl) if hasattr(tl, "__len__") else None
        for bi, batch in enumerate(tqdm(tl, desc=f"ep{ep+1}/{epochs} train")):
            x = _to_device(batch, dev)
            y = batch["label"].to(dev, non_blocking=True)
            w = batch["weight"].to(dev, non_blocking=True)
            lamp_t = batch["lamp_tgt"].to(dev, non_blocking=True)
            valid_t = batch["valid"].to(dev, non_blocking=True)
            state_t = batch["state_tgt"].to(dev, non_blocking=True)
            rel_t = batch["rel_tgt"].to(dev, non_blocking=True)
            dir_t = batch.get("dir_tgt")
            dir_t = dir_t.to(dev, non_blocking=True) if dir_t is not None else None
            if p_erase > 0:
                # manufactured NoR: erase ALL lamps of a frame -> genuine
                # "no relevant lamp" case, label and token targets follow the
                # pixels. Applied before the views so both consistency views
                # see the same (frame, label) pair.
                x, y, w, lamp_t, valid_t, rel_t = apply_lamp_erasure(
                    x, lamp_t, y, w, lamp_t, valid_t, rel_t, p_erase)
            # view 1 carries the token-level aux losses, so it must NOT be
            # geometrically cropped: the token targets are rasterised on the
            # uncropped frame (measured: 0.780 -> 0.005 lamp-hit-rate when cropped).
            x1 = train_view(x, crop=False)
            with torch.autocast("cuda", enabled=use_amp):
                out = model(x1)
                g_loss = global_loss(out["logits"], y, log_pi, tau_la, smoothing, w)
                loss = g_loss
                t_loss = torch.zeros((), device=dev)
                if use_token:
                    # note: lamp-erasure zeroes lamp/rel targets above, and the
                    # state/dir losses mask on lamp-positive tokens, so erased
                    # frames automatically lose their state/dir supervision.
                    t_loss = token_losses(out["maps"],
                                          lamp_t, valid_t, state_t, rel_t,
                                          w_lamp, w_state, w_rel,
                                          float(lcfg.get("lamp_pos_weight", 10.0)),
                                          rel_pw,
                                          rel_focal_gamma=rel_gamma,
                                          w_dir=w_dir, dir_tgt=dir_t,
                                          state_weights=state_w,
                                          state_rel_boost=state_boost)["total"]
                    loss = loss + t_loss
                c_loss = torch.zeros((), device=dev)
                if w_consist > 0:
                    # view 2 has no token loss -> free to include the framing crop
                    x2 = train_view(x)
                    out2 = model(x2)
                    c_loss = consistency_kl(out["logits"], out2["logits"])
                    loss = loss + w_consist * c_loss
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
        live = copy.deepcopy(model.decoder.state_dict())
        model.decoder.load_state_dict(ema.state_dict())
        ys, lgs, meta = predict_split(model, vl, dev)
        model.decoder.load_state_dict(live)
        rep = report(ys, lgs, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
        met = rep["metrics"]
        # slice mAP is only comparable to overall mAP when the slice contains
        # all 3 classes (mAP averages over the classes present in the slice,
        # so a 2-class city reports an inflated number). The gate uses 3-class
        # slices only; n_classes travels with each slice for the reports.
        city_m = {c: m["mAP"] for c, m in rep.get("slices", {}).get("city", {}).items()
                  if m.get("n_classes", 3) >= 3}
        worst_city = min(city_m.values()) if city_m else float("nan")
        print(f"ep{ep+1} loss={tot['total']/max(nbatch,1):.4f} "
              f"(g={tot['global']/max(nbatch,1):.4f} t={tot['token']/max(nbatch,1):.4f} "
              f"c={tot['consist']/max(nbatch,1):.4f}) "
              f"AP_RR={met['AP_RR']:.4f} AP_RG={met['AP_RG']:.4f} AP_NoR={met['AP_NoR']:.4f} "
              f"mAP={met['mAP']:.4f} acc_bal={met['acc_bal']:.4f} worst_city_mAP={worst_city:.4f}")
        if city_m:
            print("  city mAP: " + " ".join(f"{c}={v:.3f}" for c, v in sorted(city_m.items())))
        # live failure-mode view: per-class operating point + clean vs pseudo NoR
        # (this is what was invisible for 9 epochs of exp2 and only found
        # afterwards by scripts/diagnose.py: AP_NoR 0.115 clean vs 0.042 pseudo)
        _pred = lgs.argmax(1)
        parts = []
        for ci, cn in enumerate(("RR", "RG", "NoR")):
            m = ys == ci
            rec = float((_pred[m] == ci).mean()) if m.any() else float("nan")
            hit = _pred == ci
            prec = float((ys[hit] == ci).mean()) if hit.any() else float("nan")
            parts.append(f"{cn} r{rec:.2f}/p{prec:.2f}")
        if pseudo_flags is not None:
            pf = np.asarray(pseudo_flags)[:len(ys)]
            for nm, msk in (("cleanNoR", (ys == 2) & (pf == 0)),
                            ("pseudoNoR", (ys == 2) & (pf == 1))):
                if int(msk.sum()) >= 10:
                    idx = np.flatnonzero(msk | (ys != 2))
                    ap = _safe_ap((ys[idx] == 2).astype(int), softmax_np(lgs[idx])[:, 2])
                    parts.append(f"AP_{nm}={ap:.3f}")
        print("  classes: " + "  ".join(parts))

        # temperature scaling on val logits (calibrated probs for inspection).
        # v4 fix: metrics above are computed at T = 1 - best.pt selection and
        # history stay comparable with exp2/exp3. The old code overwrote
        # met["temperature"] with the fitted T on *uncalibrated* metrics, so
        # state["metrics"] mixed the two; the calibrated report is now stored
        # separately (and evaluate.py fits T on val, not from the checkpoint).
        T = fit_temperature(lgs, ys)
        rep_cal = report(ys, lgs, T=T, cities=meta["cities"], max_lamp_h=meta["max_lamp_h"])
        met["temperature"] = T                     # divide logits by T for calibrated probs
        met["ece_calibrated"] = rep_cal["metrics"]["ece"]

        state = {"head": copy.deepcopy(model.decoder.state_dict()),
                 "ema": copy.deepcopy(ema.state_dict()), "cfg": cfg,
                 "epoch": ep + 1, "metrics": met, "report": rep,
                 "report_calibrated": rep_cal,
                 "log_pi": log_pi.tolist()}
        if met["mAP"] > best_map:
            best_map = met["mAP"]
            best_state = state
            stale = 0
            torch.save(state, os.path.join(out_dir, "best.pt"))
        else:
            stale += 1
        last = dict(state)
        last.update({"opt": opt.state_dict(), "scaler": scaler.state_dict()})
        torch.save(last, os.path.join(out_dir, "last.pt"))
        with open(os.path.join(out_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(to_jsonable(
                {"epoch": ep + 1, **{k: v for k, v in met.items()
                                     if isinstance(v, (int, float))},
                 "worst_city_mAP": worst_city})) + "\n")
        if patience and stale >= patience:
            print(f"early stop: val mAP flat for {stale} epochs "
                  f"(patience {patience}); best {best_map:.4f}")
            # marker so the auto-resume watchdog does not relaunch past a
            # deliberate early stop (it only counts history lines otherwise)
            with open(os.path.join(out_dir, "STOPPED"), "w") as f:
                f.write(f"epoch {ep+1} stale {stale} best {best_map:.4f}\n")
            break

    print(f"best val mAP={best_map:.4f} (test untouched; run scripts/evaluate.py "
          f"once for the unbiased report)")
    return best_state
