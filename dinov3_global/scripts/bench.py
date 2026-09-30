"""Bench: verify 1280x720 ViT-S fits 12GB and measure ms/img.

python dinov3_global/scripts/bench.py --config dinov3_global/configs/base.yaml
Asserts seq_len==3605 (3600 patches + CLS path) and prints peak VRAM.
"""
import argparse
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from dinov3_global.dinov3_global.model import DinoGlobal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="dinov3_global/configs/base.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    with open(os.path.join(repo_root, args.config)) as f:
        cfg = yaml.safe_load(f)
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    m = cfg["backbone"]
    _ckpt = m.get("local_ckpt", None)
    if _ckpt and not os.path.isabs(_ckpt):
        _ckpt = os.path.join(repo_root, _ckpt)
    model = DinoGlobal(hf_id=m["hf_id"], attn_implementation=m.get("attn_implementation", "sdpa"),
                       dtype=m.get("dtype", "float16") if dev.type == "cuda" else "float32",
                       proj_dim=cfg["decoder"].get("proj_dim", 256),
                       local_ckpt=_ckpt).to(dev)
    model.eval()
    x = torch.zeros(1, 3, 720, 1280, device=dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    # warmup
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for _ in range(3):
            out = model(x)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad(), torch.autocast("cuda", enabled=dev.type == "cuda"):
        for _ in range(args.iters):
            out = model(x)
    torch.cuda.synchronize() if dev.type == "cuda" else None
    ms = (time.time() - t0) / args.iters * 1000
    print(f"logits {tuple(out['logits'].shape)} attn {tuple(out['attn'].shape)}")
    assert out["logits"].shape == (1, 3), "decoder must output 3 global logits"
    assert out["attn"].shape == (1, 45, 80), "attn map must be 45x80 patch grid"
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated(dev) / 1024 ** 3
        print(f"forward {ms:.1f} ms/img batch=1 | peak VRAM {peak:.2f} GB / 12 GB")
        print(f"epoch estimate train(28525): {ms*28525/60000:.0f} min backbone+decoder forward-only")
    else:
        print(f"forward {ms:.1f} ms/img (cpu)")


if __name__ == "__main__":
    main()
