"""Configuration contract for the single supported Experiment C architecture."""

from pathlib import Path
import yaml


def architecture_name(cfg):
    validate_config(cfg)
    return "dinoglobal_mil"


def validate_config(cfg):
    decoder, backbone, data = cfg["decoder"], cfg["backbone"], cfg["data"]
    # Keep C's serialized identifier so the original EMA checkpoint loads strictly.
    if decoder.get("head") != "v7_evidence":
        raise ValueError("Only the Experiment C evidence head is supported")
    for key, expected in dict(pool="local", relevance_context=True).items():
        if decoder.get(key) != expected:
            raise ValueError(f"Experiment C requires decoder.{key}={expected!r}")
    # Original C checkpoints contain fixed, unused configuration fields from v5.
    for key, expected in dict(
        nor_sees_evidence=True, dir_head=True, cls_modulates_evidence=False, positional_mode="none"
    ).items():
        if key in decoder and decoder[key] != expected:
            raise ValueError(f"Incompatible decoder.{key}")
    if decoder.get("spatial_context", {}).get("mode", "none") != "none":
        raise ValueError("Experiment C does not use an additional spatial module")
    if backbone.get("mid_layer") != 6 or backbone.get("frozen") is not True:
        raise ValueError("A frozen backbone with layer-6/final fusion is required")
    if backbone.get("patch") != 16 or backbone.get("dim") != 384:
        raise ValueError("The backbone must be the 384-dimensional ViT-S+/16")
    if backbone.get("use_cls") is not True:
        raise ValueError("The backbone CLS token is required")
    if backbone.get("hf_id") != "facebook/dinov3-vits16plus-pretrain-lvd1689m":
        raise ValueError("The pinned DINOv3 ViT-S+/16 pretrained model is required")
    if data.get("global_classes") != ["RR", "RG", "NoR"]:
        raise ValueError("Class order must be RR, RG, NoR")
    if data.get("label_policy") != "map_to_nor":
        raise ValueError("The label policy must be map_to_nor")
    h, w = data["target_hw"]
    if min(h, w) < 16 or h % 16 or w % 16:
        raise ValueError("Image dimensions must be positive multiples of 16")
    if data.get("preprocessing") != "letterbox" or any(
        data.get(key) != 0 for key in ("crop_sides", "label_crop_sides")
    ):
        raise ValueError("Experiment C requires full-frame letterboxing with zero side crop")
    if cfg.get("loss", {}).get("class_balance", "none") not in ("none", "sqrt"):
        raise ValueError("Unknown loss.class_balance")
    if cfg.get("loss", {}).get("attribute_reduction", "token") not in ("token", "instance"):
        raise ValueError("Unknown loss.attribute_reduction")
    return cfg


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))
