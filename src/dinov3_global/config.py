"""Load the single supported architecture and reject obsolete model settings."""

from pathlib import Path
import yaml


def validate_config(cfg):
    fixed = dict(
        head="mil",
        pool="local",
        nor_sees_evidence=True,
        dir_head=True,
        cls_modulates_evidence=True,
        positional_mode="relevance",
        relevance_context=True,
    )
    for key, expected in fixed.items():
        if cfg["decoder"].get(key) != expected:
            raise ValueError(f"v5 requires decoder.{key}={expected!r}")
    if cfg["backbone"].get("mid_layer") != 6 or cfg["backbone"].get("frozen") is not True:
        raise ValueError("v5 requires a frozen backbone and layer-6/final fusion")
    if cfg["backbone"].get("patch") != 16 or cfg["backbone"].get("dim") != 384:
        raise ValueError("v5 uses the 384-dimensional ViT-S+/16 backbone")
    if cfg["backbone"].get("use_cls") is not True:
        raise ValueError("v5 requires the backbone CLS token")
    if cfg["backbone"].get("hf_id") != "facebook/dinov3-vits16plus-pretrain-lvd1689m":
        raise ValueError("v5 requires the DINOv3 ViT-S+/16 pretrained model")
    if cfg["data"].get("global_classes") != ["RR", "RG", "NoR"]:
        raise ValueError("Class order must be RR, RG, NoR")
    if cfg["data"].get("label_policy") != "map_to_nor":
        raise ValueError("The documented v5 label policy is map_to_nor")
    h, w = cfg["data"]["target_hw"]
    if min(h, w) < 16 or h % 16 or w % 16:
        raise ValueError("Image dimensions must be positive multiples of 16")
    return cfg


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))
