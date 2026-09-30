"""Live training monitor: learning / plateau / overfit diagnosis.

  python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp2
  python dinov3_global/scripts/monitor.py --run dinov3_global/runs/exp2 --plot

Reads history.jsonl (written every epoch) + the console log, prints per-epoch
metrics with trend deltas, and a verdict:
  LEARNING   - val mAP still improving over the last 3 epochs
  PLATEAU    - no improvement for N epochs but train loss still healthy
  OVERFIT    - train loss falling while val mAP flat/down, best.pt is old
  NOISY      - not enough epochs yet to judge
Use --plot to save run.png (loss + val curves) when matplotlib is available.
"""
import argparse
import json
import os
import re
from typing import List, Optional

LOSS_RE = re.compile(
    r"ep(\d+) loss=([\d.]+) \(g=([\d.]+) t=([\d.]+) c=([\d.]+)\)")


def _f(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


def read_text_smart(path: str) -> str:
    """Console logs may be UTF-16 (PowerShell *> redirect) or UTF-8."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="ignore")
    return raw.decode("utf-8", errors="ignore")


def load_history(run: str) -> List[dict]:
    path = os.path.join(run, "history.jsonl")
    rows = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    # strict-JSON history may carry null for undefined metrics
                    # (e.g. worst_city_mAP on sub-sample runs) -> NaN here
                    rows.append({k: (float("nan") if v is None else v)
                                 for k, v in r.items()})
    return rows


def load_losses(run: str) -> List[dict]:
    """Epoch train losses from the console log (any *.log in run/ or its parent)."""
    rows = []
    cands = [os.path.join(run, "train_console.log")]
    parent = os.path.dirname(os.path.abspath(run))
    cands += [os.path.join(parent, n) for n in os.listdir(parent)
              if n.endswith(".log") and os.path.basename(run) in n]
    for p in cands:
        if not os.path.isfile(p):
            continue
        for line in read_text_smart(p).splitlines():
            m = LOSS_RE.search(line.replace("\x08", ""))
            if m:
                rows.append({"epoch": int(m.group(1)), "train_loss": _f(m.group(2)),
                             "g": _f(m.group(3)), "t": _f(m.group(4)), "c": _f(m.group(5))})
    return rows


def verdict(rows: List[dict], losses: List[dict], patience: int = 5) -> str:
    if len(rows) < 3:
        return "NOISY - too few epochs to judge (need >= 3)"
    m = [r.get("mAP", float("nan")) for r in rows]
    best_ep = max(range(len(m)), key=lambda i: m[i]) + 1
    last3 = m[-3:]
    improving = (last3[-1] > last3[0]) or (max(last3) >= max(m) - 1e-9 and max(m) == last3[-1])
    stale = len(m) - best_ep

    tl = [x["train_loss"] for x in losses]
    loss_falling = len(tl) >= 2 and (tl[-1] < tl[-2] < tl[-3] if len(tl) >= 3 else tl[-1] < tl[-2])

    if stale >= patience and loss_falling:
        return (f"OVERFIT - best val mAP {max(m):.4f} was epoch {best_ep} "
                f"({stale} epochs stale) while train loss keeps falling")
    if improving:
        return f"LEARNING - val mAP rising (last3: {' -> '.join(f'{v:.4f}' for v in last3)})"
    if stale >= patience:
        return f"PLATEAU - no val mAP gain in {stale} epochs (best {max(m):.4f} @ ep{best_ep})"
    return f"MIXED - best {max(m):.4f} @ ep{best_ep}; watch for {patience}-epoch staleness"


def live_progress(run: str) -> Optional[str]:
    """Current tqdm line from the console log: 'ep2/30 train: 43% 2600/6037 [...]'."""
    parent = os.path.dirname(os.path.abspath(run))
    base = os.path.basename(os.path.abspath(run))
    cands = [p for p in [os.path.join(parent, f"{base}_console.log"),
                         os.path.join(run, "train_console.log")]
             if os.path.isfile(p)]
    for p in cands:
        tail = [ln.replace("\x08", "").strip()
                for ln in read_text_smart(p).splitlines()
                if "train:" in ln and "/" in ln]
        if tail:
            return tail[-1]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    rows = load_history(args.run)
    losses = load_losses(args.run)
    live = live_progress(args.run)
    if live:
        print("live:", live[-120:])
    if not rows and not losses:
        print(f"no epoch finished yet in {args.run} - check the 'live' line above")
        return

    lmap = {x["epoch"]: x for x in losses}
    print(f"{'ep':>3} {'tr_loss':>8} {'g':>7} {'t':>7} {'c':>7} {'AP_RR':>7} {'AP_RG':>7} "
          f"{'AP_NoR':>7} {'mAP':>7} {'d_mAP':>7} {'acc_bal':>7} {'worst_c':>7} {'ece':>6}")
    for i, r in enumerate(rows, 1):
        l = lmap.get(i, {})
        d = r.get("mAP", float("nan")) - rows[i - 2].get("mAP", float("nan")) if i > 1 else float("nan")
        print(f"{i:>3} {l.get('train_loss', float('nan')):8.4f} {l.get('g', float('nan')):7.4f} "
              f"{l.get('t', float('nan')):7.4f} {l.get('c', float('nan')):7.4f} "
              f"{r.get('AP_RR', float('nan')):7.4f} {r.get('AP_RG', float('nan')):7.4f} "
              f"{r.get('AP_NoR', float('nan')):7.4f} {r.get('mAP', float('nan')):7.4f} "
              f"{d:7.4f} {r.get('acc_bal', float('nan')):7.4f} "
              f"{r.get('worst_city_mAP', float('nan')):7.4f} {r.get('ece', float('nan')):6.3f}")
    print("\nverdict:", verdict(rows, losses))

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed - skipping --plot")
            return
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if losses:
            ax[0].plot([x["epoch"] for x in losses], [x["train_loss"] for x in losses],
                       label="train loss")
            ax[0].plot([x["epoch"] for x in losses], [x["g"] for x in losses], "--", label="global")
            ax[0].plot([x["epoch"] for x in losses], [x["t"] for x in losses], "--", label="token")
            ax[0].set_xlabel("epoch"); ax[0].set_title("train loss"); ax[0].legend()
        eps = [r.get("epoch", i) for i, r in enumerate(rows, 1)]
        for k in ("mAP", "acc_bal", "AP_RR", "AP_RG", "AP_NoR", "worst_city_mAP"):
            ax[1].plot(eps, [r.get(k, float("nan")) for r in rows], label=k)
        ax[1].set_xlabel("epoch"); ax[1].set_title("val metrics"); ax[1].legend()
        out = os.path.join(args.run, "monitor.png")
        fig.tight_layout(); fig.savefig(out, dpi=110)
        print(f"plot -> {out}")


if __name__ == "__main__":
    main()
