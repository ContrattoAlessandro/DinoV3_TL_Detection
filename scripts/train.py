"""Train the default Experiment C architecture on session-disjoint DTLD training data."""

import argparse
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from dinov3_global.engine import train
from dinov3_global.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/train")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--resume", default=None, help="path to last.pt to resume from")
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)
    cfg = load_config(cfg_path)
    if args.epochs is not None:
        cfg["optim"]["epochs"] = args.epochs
    if args.seed is not None:
        cfg["optim"]["seed"] = args.seed
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out if os.path.isabs(args.out) else os.path.join(repo_root, args.out)
    train(
        cfg,
        out,
        repo_root=repo_root,
        device=device,
        max_train=args.max_train,
        max_val=args.max_val,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
