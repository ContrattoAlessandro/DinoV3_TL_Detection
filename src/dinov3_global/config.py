"""Load v5 or its optional relevance-only axial spatial variant."""

from pathlib import Path
import yaml
from .head import spatial_settings


def architecture_name(cfg):
    if cfg["decoder"].get("head") == "v7_evidence":
        return "v7_evidence"
    return "v6_axial" if cfg["decoder"].get("spatial_context", {}).get("mode", "none") == "axial" else "v5"


def validate_config(cfg):
    head = cfg["decoder"].get("head")
    if head not in ("mil", "v7_evidence"):
        raise ValueError("Unsupported decoder.head")
    spatial_settings(cfg["decoder"].get("spatial_context"), cfg["decoder"]["proj_dim"])
    fixed = dict(
        pool="local",
        nor_sees_evidence=True,
        dir_head=True,
        cls_modulates_evidence=True,
        positional_mode="relevance",
        relevance_context=True,
    )
    if head == "v7_evidence":
        fixed.update(cls_modulates_evidence=False, positional_mode="none")
        if cfg["decoder"].get("spatial_context", {}).get("mode", "none") != "none":
            raise ValueError("v7_evidence does not use axial attention")
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
    mode = cfg["data"].get("preprocessing", "legacy")
    if mode not in ("legacy", "letterbox"):
        raise ValueError("Unknown data.preprocessing")
    if mode == "letterbox" and (
        cfg["data"].get("crop_sides") != 0 or cfg["data"].get("label_crop_sides") != 0
    ):
        raise ValueError("Full-frame letterboxing requires zero input and label side crop")
    if cfg.get("loss", {}).get("class_balance", "none") not in ("none", "sqrt"):
        raise ValueError("Unknown loss.class_balance")
    if cfg.get("loss", {}).get("attribute_reduction", "token") not in ("token", "instance"):
        raise ValueError("Unknown loss.attribute_reduction")
    return cfg


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))
