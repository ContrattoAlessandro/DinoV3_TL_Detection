"""Configuration contract for the single supported DinoGlobal-MIL architecture."""

from pathlib import Path
import yaml
from .backbone import backbone_spec


def architecture_name(cfg):
    validate_config(cfg)
    return "dinoglobal_mil"


def validate_config(cfg):
    decoder, backbone, data = cfg["decoder"], cfg["backbone"], cfg["data"]
    # Preserve the released checkpoint's serialized head identifier.
    if decoder.get("head") != "v7_evidence":
        raise ValueError("Only the DinoGlobal-MIL evidence head is supported")
    for key, expected in dict(pool="local", relevance_context=True).items():
        if decoder.get(key) != expected:
            raise ValueError(f"DinoGlobal-MIL requires decoder.{key}={expected!r}")
    # The released checkpoint contains these fixed compatibility fields.
    for key, expected in dict(
        nor_sees_evidence=True, dir_head=True, cls_modulates_evidence=False, positional_mode="none"
    ).items():
        if key in decoder and decoder[key] != expected:
            raise ValueError(f"Incompatible decoder.{key}")
    if decoder.get("spatial_context", {}).get("mode", "none") != "none":
        raise ValueError("DinoGlobal-MIL does not use an additional spatial module")
    if any(
        key in decoder for key in ("context_source", "detach_relevance_attributes")
    ) or "supervise_attributes_view2" in cfg.get("loss", {}):
        raise ValueError("Only the released context and attribute-supervision paths are supported")
    if backbone.get("mid_layer") != 6 or backbone.get("frozen") is not True:
        raise ValueError("A frozen backbone with layer-6/final fusion is required")
    spec = backbone_spec(backbone.get("hf_id"))
    if backbone.get("patch") != spec["patch"] or backbone.get("dim") != spec["dim"]:
        raise ValueError("backbone.patch and backbone.dim must match the selected DINOv3 model")
    if backbone.get("use_cls") is not True:
        raise ValueError("The backbone CLS token is required")
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
        raise ValueError("DinoGlobal-MIL requires full-frame letterboxing with zero side crop")
    if cfg.get("loss", {}).get("class_balance", "none") not in ("none", "sqrt"):
        raise ValueError("Unknown loss.class_balance")
    if cfg.get("loss", {}).get("attribute_reduction", "token") not in ("token", "instance"):
        raise ValueError("Unknown loss.attribute_reduction")
    if any(key in cfg.get("loss", {}) for key in ("lamp_positive_fraction", "token_evidence")):
        raise ValueError("Experimental RGB/evidence-supervision configurations are unsupported")
    return cfg


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return validate_config(yaml.safe_load(stream))
