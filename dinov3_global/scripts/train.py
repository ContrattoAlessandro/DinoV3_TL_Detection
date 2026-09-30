"""Train entry-point.

  python dinov3_global/scripts/train.py --config dinov3_global/configs/base.yaml \
      --out dinov3_global/runs/exp2
  python dinov3_global/scripts/train.py ... --resume dinov3_global/runs/exp2/last.pt
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.engine import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov3_global/configs/base.yaml")
    ap.add_argument("--out", default="dinov3_global/runs/exp2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["optim"]["seed"] = args.seed
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    train(cfg, out, repo_root=repo_root, device=device,
          max_train=args.max_train, max_val=args.max_val, resume=args.resume)


if __name__ == "__main__":
    main()
