"""Architecture diagnosis for a trained DinoGlobal-MIL checkpoint (VAL ONLY).

Answers, with numbers, the questions that matter for the next model version:

1. What did the head actually learn? (learned taus, evidence gain/bias)
2. Does the MIL evidence path separate the classes at all? (score separation)
3. Is the NoR logit a good ranker, and would the pooled EVIDENCE have been a
   better one? (oracle complement AP -> justifies feeding S into the NoR head)
4. Is AP_NoR lost on "clean" NoR frames or on pseudo-NoR frames (relevant
   off/unknown lamps) -> is `label_policy: downweight` misaligned with the metric?
5. Does the training-only token supervision actually localise lamps, and how
   much is it corrupted by random_framing (token targets are computed on the
   UNCROPPED image)?
6. Class/pseudo-label statistics of the split.

Read-only: loads val only, never the official test split.

  python dinov3_global/scripts/diagnose.py --ckpt dinov3_global/runs/exp2/best.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.engine import (build_loaders, load_model)  # noqa: E402
from dinov3_global.dinov3_global.eval_global import softmax_np  # noqa: E402

CLS = ["RR", "RG", "NoR"]


@torch.no_grad()
def forward_full(model, x, dev):
    """Re-runs the trained head branch-by-branch so we can see every stage."""
    dec = model.decoder
    feats = model.backbone(x)
    patches, cls = feats["patches"].float(), feats["cls"].float()
    mid = feats.get("patches_mid")
    if mid is not None:  # v4 mid-layer fusion (shared projection -> both)
        patches = torch.cat([patches, mid.float()], dim=-1)
        cls = torch.cat([cls, feats["cls_mid"].float()], dim=-1)
    h = dec.proj_norm(dec.proj(patches)) + dec.pos.to(dev).unsqueeze(0)
    per_branch, maps = [], None
    for br in dec.branches:
        s, mp = dec._branch(br, h)
        per_branch.append((s, mp))
        maps = mp if maps is None else maps
    cm = dec.proj_norm(dec.proj(cls))
    s_mean = torch.stack([s for s, _ in per_branch]).mean(0)   # [B,2]
    # keep this in sync with TokenMILHead.forward (nor_sees_evidence + cls mod)
    if getattr(dec, "nor_sees_evidence", False):
        nor_in = torch.cat([cm, s_mean], dim=-1)
    else:
        nor_in = cm
    nor = dec.nor_head(nor_in).squeeze(-1)
    mod = (dec.cls_mod(cm) if getattr(dec, "cls_modulates_evidence", False)
           else torch.zeros_like(s_mean))
    return per_branch, maps, nor, mod


def _ap(y_bin, score):
    if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
        return float("nan")
    return float(average_precision_score(y_bin, score))


def deterministic_crop(x, s=0.9, fy=0.25, fx=0.25):
    """Replay of random_framing with FIXED parameters (as a training step would)."""
    B, _, H, W = x.shape
    ch, cw = int(H * s), int(W * s)
    top, left = int(H * fy), int(W * fx)
    x = x[:, :, top:top + ch, left:left + cw]
    return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-val", type=int, default=0, help="0 = all val frames")
    ap.add_argument("--n-crop", type=int, default=800, help="frames for the framing test")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, sd = load_model(args.ckpt, repo_root, dev)
    dec = model.decoder
    print(f"ckpt {args.ckpt} | head={cfg['decoder']['head']} | "
          f"epoch={sd.get('epoch')} | val mAP={sd.get('metrics', {}).get('mAP')}")
    n_par = sum(p.numel() for p in dec.parameters())
    print(f"trainable head params: {n_par/1e6:.2f}M (backbone frozen: "
          f"{sum(p.numel() for p in model.backbone.parameters())/1e6:.1f}M)")

    # ---------------------------------------------------------------- data stats
    tl, vl, tr_ds, va_ds, info = build_loaders(cfg, repo_root)
    def _stats(ds, name):
        y = np.array([it["y"] for it in ds.items])
        pn = np.array([it["pseudo_nor"] for it in ds.items])
        nl = np.array([it["n_lamps"] for it in ds.items])
        h = np.bincount(y, minlength=3)
        print(f"\n[{name}] n={len(y)} hist(RR,RG,NoR)={h.tolist()} "
              f"prevalence={np.round(h/len(y), 4).tolist()}")
        m = y == 2
        print(f"  NoR frames: {int(m.sum())} | pseudo-NoR (relevant off/unknown lamp "
              f"present) {int(pn.sum())} ({100*pn.sum()/max(m.sum(),1):.0f}% of NoR) | "
              f"clean NoR {int(m.sum()-pn.sum())}")
        if m.sum():
            print(f"  NoR frames with n_lamps=0: {int((nl[m]==0).sum())} "
                  f"| median n_lamps in NoR {float(np.median(nl[m])):.0f}")
        return pn
    pn_tr = _stats(tr_ds, "train")
    pn_va = _stats(va_ds, "val")

    # ------------------------------------------------------------- head internals
    try:
        tau = [float(b.tau.detach()) for b in dec.branches]
        print(f"\n[learned pooling taus] {[round(t,4) for t in tau]} (init {[0.05,0.3]})")
    except RuntimeError:  # topk branches have no learnable tau
        tau = []
        print(f"\n[pooling] topk mode ks={[b.k for b in dec.branches]} (no learnable tau)")
    print(f"[learned evidence scale a] {dec.scale.detach().cpu().numpy().round(4).tolist()}"
          f"  [bias b] {dec.bias.detach().cpu().numpy().round(4).tolist()}")

    # ------------------------------------------------------------- forward on val
    y_all, lg_all, nor_all = [], [], []
    br_scores = [[] for _ in dec.branches]
    emax_all, nl_all, rl_all = [], [], []
    lamp_prob_chunks, lamp_tgt_chunks, valid_chunks = [], [], []
    n_seen = 0
    for batch in vl:
        x = batch["image"].to(dev).float().div_(255.0)
        with torch.autocast("cuda", enabled=dev.type == "cuda"):
            pb, maps, nor, mod = forward_full(model, x, dev)
            s_avg = torch.stack([pb[bi][0] for bi in range(len(pb))]).mean(0)
            logits = torch.stack([s_avg[:, 0] * dec.scale[0] + dec.bias[0] + mod[:, 0],
                                  s_avg[:, 1] * dec.scale[1] + dec.bias[1] + mod[:, 1],
                                  nor], dim=1).detach()
        y_all.append(batch["label"].numpy())
        lg_all.append(logits.float().cpu().numpy())
        nor_all.append(nor.float().cpu().numpy())
        for bi, (s, _m) in enumerate(pb):
            br_scores[bi].append(s.detach().float().cpu().numpy())
        emax = torch.stack([maps["rr"].amax((1, 2)), maps["rg"].amax((1, 2))], dim=1)
        emax_all.append(emax.float().cpu().numpy())
        lamp_prob_chunks.append(maps["lamp"].float().cpu().numpy().reshape(len(x), -1))
        lamp_tgt_chunks.append(batch["lamp_tgt"].numpy().reshape(len(x), -1))
        valid_chunks.append(batch["valid"].numpy().reshape(len(x), -1))
        rl_all.append(batch["rel_tgt"].numpy().reshape(len(x), -1))
        nl_all.append(np.array(batch["n_lamps"], dtype=float))
        n_seen += len(x)
        if args.n_val and n_seen >= args.n_val:
            break
    y = np.concatenate(y_all); logits = np.concatenate(lg_all)
    nor_logit = np.concatenate(nor_all)
    lp = np.concatenate(lamp_prob_chunks)
    lt = np.concatenate(lamp_tgt_chunks)
    vld = np.concatenate(valid_chunks)
    rl = np.concatenate(rl_all)
    S = np.stack([np.concatenate(x) for x in br_scores], axis=1)  # [N, nbr, 2]
    emax = np.concatenate(emax_all)
    prob = softmax_np(logits)
    nbr = S.shape[1]

    # ------------------------------------------------------- score separation
    print("\n[class separation of the pooled evidence]  (mean +- sd per GT class)")
    for ci, c in enumerate(CLS):
        m = y == ci
        row = f"  {c:>3} (n={int(m.sum()):5d}): "
        for bi in range(nbr):
            row += (f"S[br{bi}]_rr={S[m, bi, 0].mean():+.4f}+-{S[m, bi, 0].std():.4f} "
                    f"S[br{bi}]_rg={S[m, bi, 1].mean():+.4f}  ")
        row += f" emax_rr={emax[m, 0].mean():.4f} nor_logit={nor_logit[m].mean():+.3f}"
        print(row)
    d = S[y == 0, 0, 0].mean() - S[y == 2, 0, 0].mean()
    S_avg = S.mean(1)
    d_avg = S_avg[y == 0, 0].mean() - S_avg[y == 2, 0].mean()
    print(f"  --> dynamic range of the pooled score (RR minus NoR, branch0): {d:+.4f}"
          f" | branch-averaged: {d_avg:+.4f}")
    print(f"  --> implied pre-softmax logit gap (a0 * that): "
          f"{float(dec.scale[0].detach()) * d:+.4f}   (tau*log N penalty: "
          + (f"at tau={tau[0]:.3f} is {tau[0]*np.log(3600):.3f}, at tau={tau[1]:.3f} is "
             f"{tau[1]*np.log(3600):.3f})" if len(tau) >= 2 else "n/a in topk mode)"))
    print("  --> false red-evidence: fraction of frames with max token RR evidence "
          "> 0.5 / > 0.25")
    nl = np.concatenate(nl_all)
    for ci, c in enumerate(CLS):
        m = y == ci
        print(f"      {c:>3}: emax_rr>0.5 {float((emax[m,0]>0.5).mean()):.3f}   "
              f"emax_rr>0.25 {float((emax[m,0]>0.25).mean()):.3f}   "
              f"mean lamps/frame {nl[m].mean():.1f}   "
              f"rel_lamp_tokens/frame {float((lt[m]*rl[m]).sum()/max(m.sum(),1)):.2f}")

    # ------------------------------------------------------- NoR path diagnostics
    ybin_nor = (y == 2).astype(int)
    print("\n[NoR decision path]")
    print(f"  AP_NoR with the learned NoR logit            : {_ap(ybin_nor, prob[:,2]):.4f}")
    print(f"  AP_NoR with raw evidence complement -(S0rr+S0rg): "
          f"{_ap(ybin_nor, -(S[:,0,0]+S[:,0,1])):.4f}")
    print(f"  AP_NoR with 1-max(S0rr,S0rg)                 : "
          f"{_ap(ybin_nor, 1.0-np.maximum(S[:,0,0],S[:,0,1])):.4f}")
    print(f"  AP_NoR with branch-AVERAGED complement -(Srr+Srg): "
          f"{_ap(ybin_nor, -(S_avg[:,0]+S_avg[:,1])):.4f}")
    print(f"  AP_NoR with branch-AVERAGED 1-max(Srr,Srg)      : "
          f"{_ap(ybin_nor, 1.0-np.maximum(S_avg[:,0],S_avg[:,1])):.4f}")
    print(f"  AP_NoR with -emax_rr (best token evidence)    : {_ap(ybin_nor, -emax[:,0]):.4f}")
    print(f"  AP_NoR with -emax_max(rr,rg)                 : "
          f"{_ap(ybin_nor, -emax.max(axis=1)):.4f}")
    # does adding evidence to the CLS feature help? (linear probe, val-fit only)
    cm_all = []
    with torch.no_grad():
        for batch in vl:
            x = batch["image"].to(dev).float().div_(255.0)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                feats = model.backbone(x)
                cls = feats["cls"].float()
                if feats.get("patches_mid") is not None:  # v4 fusion
                    cls = torch.cat([cls, feats["cls_mid"].float()], dim=-1)
                cm_all.append(dec.proj_norm(dec.proj(cls)).float().cpu().numpy())
            if len(np.concatenate(cm_all)) >= len(y):
                break
    cm = np.concatenate(cm_all)[:len(y)]
    X = np.concatenate([cm, S[:, 0, :], emax], axis=1)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ntr = int(0.7 * len(y))
    lr_ = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lr_.parameters(), lr=3e-3, weight_decay=1e-2)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(ybin_nor, dtype=torch.float32)
    for _ in range(400):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(lr_(Xt[:ntr]).squeeze(-1), yt[:ntr])
        loss.backward(); opt.step()
    with torch.no_grad():
        pr = lr_(Xt).squeeze(-1).numpy()
    print(f"  AP_NoR, LINEAR PROBE on [CLS-head feats + S + emax] (fit on 70% of val, "
          f"scored on all): {_ap(ybin_nor, pr):.4f}")

    # ------------------------------------------------------- clean vs pseudo NoR
    print("\n[AP_NoR by NoR sub-population]  (pseudo-NoR = frame is NoR ONLY because "
          "its relevant lamp is off/unknown)")
    for name, m in (("clean NoR only", (y == 2) & (pn_va[:len(y)] == 0)),
                    ("pseudo-NoR only", (y == 2) & (pn_va[:len(y)] == 1)),
                    ("all NoR", y == 2)):
        mm = m | (y != 2)   # one-vs-rest within this subset
        sub_prob = prob[mm][:, 2]
        sub_y = (y[mm] == 2).astype(int)
        sub_pred = prob[mm].argmax(1)
        is_nor = y[mm] == 2
        # recall = of the sub-population's NoR frames, how many argmax to NoR;
        # precision = of the subset's NoR predictions, how many are truly NoR.
        # (v4 fix: these two were swapped/mislabeled here before.)
        rec = float((sub_pred[is_nor] == 2).mean()) if is_nor.any() else float("nan")
        pre = float(is_nor[sub_pred == 2].mean()) if (sub_pred == 2).any() else float("nan")
        print(f"  {name:>15}: n_sub={int(m.sum()):4d}  AP={_ap(sub_y, sub_prob):.4f}"
              f"  recall@argmax={rec:.3f} precision@argmax={pre:.3f}")
    pred = prob.argmax(1)
    for ci, c in enumerate(CLS):
        m = y == ci
        hit = pred == ci
        rec = float((pred[m] == ci).mean()) if m.any() else float("nan")
        pre = float((y[hit] == ci).mean()) if hit.any() else float("nan")
        print(f"  recall {c:>3}={rec:.3f} precision={pre:.3f} (n={int(m.sum())})")

    # ------------------------------------------------------- token supervision
    sel = vld.reshape(-1) > 0
    auc = roc_auc_score(lt.reshape(-1)[sel], lp.reshape(-1)[sel])
    n_lamp_tok = (lt > 0.5).sum(1)
    n_rel_tok = ((lt > 0.5) & (rl > 0.5)).sum(1)
    print(f"\n[token-level supervision] lampness AUC on valid tokens: {auc:.4f}")
    print(f"  lamp tokens/frame: mean {n_lamp_tok.mean():.2f} median "
          f"{float(np.median(n_lamp_tok)):.0f} of 3600 "
          f"({100*n_lamp_tok.mean()/3600:.3f}% of tokens) | relevant lamp "
          f"tokens/frame mean {n_rel_tok.mean():.2f}")
    print(f"  frames whose top-1 lamp token is inside a GT box: "
          f"{float((lt[np.arange(len(y)), lp.argmax(1)] > 0.5).mean()):.4f}")

    # framing misalignment: token targets are built on the UNCROPPED image while
    # random_framing (p=0.3 in training) crops+resizes the input.
    n_take = min(args.n_crop, len(y))
    hits_plain, hits_crop, taken = [], [], 0
    with torch.no_grad():
        for batch in vl:
            x = batch["image"].to(dev).float().div_(255.0)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                _, mp1, _, _ = forward_full(model, x, dev)
                _, mp2, _, _ = forward_full(model, deterministic_crop(x), dev)
            p1 = mp1["lamp"].float().cpu().numpy().reshape(len(x), -1)
            p2 = mp2["lamp"].float().cpu().numpy().reshape(len(x), -1)
            ltv = batch["lamp_tgt"].numpy().reshape(len(x), -1)
            rows = [i for i in range(len(x)) if ltv[i].sum() > 0]
            rows = rows[:max(0, n_take - taken)]
            for i in rows:
                hits_plain.append(ltv[i, p1[i].argmax()] > 0.5)
                hits_crop.append(ltv[i, p2[i].argmax()] > 0.5)
            taken += len(rows)
            if taken >= n_take:
                break
    hp = float(np.mean(hits_plain)); hc = float(np.mean(hits_crop))
    print(f"  top-1 lamp token inside a GT box over {taken} frames, clean input : {hp:.4f}")
    print(f"  same frames after ONE random_framing-style crop (s=0.9, offset 25%) : "
          f"{hc:.4f}")
    print("  -> token targets are computed on the UNCROPPED frame, so 30% of training "
          "steps (random_framing p=0.3) supervise the wrong tokens")

    print("\n[mAP / APs on this val pass, EMA weights]")
    for ci, c in enumerate(CLS):
        print(f"  AP_{c}={_ap((y==ci).astype(int), prob[:, ci]):.4f}")
    print(f"  mAP={np.mean([_ap((y==ci).astype(int), prob[:, ci]) for ci in range(3)]):.4f}")


if __name__ == "__main__":
    main()
