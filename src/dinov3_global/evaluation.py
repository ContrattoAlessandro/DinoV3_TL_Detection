"""Audited DTLD test membership and shared-encoder evaluation of fixed heads."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from .data import global_label_from_states
from .dtld import _seq_of, image_file_for, load_split, parse_frame
from .preprocessing import batch_model_kwargs

CLASSES = ("RR", "RG", "NoR")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Atomic progress/provenance writes remain readable during long runs."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def portable_path(path, repo_root):
    path = Path(path).resolve()
    try:
        return path.relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return path.name


def audit_test_split(cfg, dataset, repo_root):
    """Fail on missing/aliased images; account for every native annotation.

    Exact-file overlap is a diagnostic of the published partition, never a
    reason to silently change test membership. Only frames without any valid
    lamp annotation follow the existing dataset parser's exclusion rule.
    """
    root = Path(repo_root).resolve()
    label_dir = root / cfg["data"]["label_dir"]
    image_root = root / cfg["data"]["img_root"]
    splits, summaries = {}, {}
    for split in ("train", "test"):
        entries = load_split(str(label_dir), split)
        rows, skipped = [], []
        seen_native, seen_files = set(), set()
        for entry in tqdm(entries, desc=f"Audit DTLD {split}", mininterval=5):
            native = entry["image_path"].replace("\\", "/").lstrip("./")
            path = Path(image_file_for(str(image_root), split, native))
            if native in seen_native or path.name in seen_files:
                raise ValueError(f"Duplicate or aliased {split} annotation: {native}")
            seen_native.add(native)
            seen_files.add(path.name)
            if not path.is_file():
                raise FileNotFoundError(f"Missing official {split} image: {path}")
            frame = parse_frame(entry)
            if frame is None:
                skipped.append(native)
                continue
            y, debug = global_label_from_states(frame.relevance, frame.state)
            if split == "test":
                with Image.open(path) as image:
                    expected_size = (
                        (2048, 1024)
                        if cfg["data"].get("preprocessing") == "letterbox"
                        else tuple(reversed(cfg["data"]["target_hw"]))
                    )
                    if image.size != expected_size or image.mode != "RGB":
                        raise ValueError(f"Test image is not a prepared RGB frame: {path}")
                    image.verify()
            rows.append(
                dict(
                    native_path=native,
                    image=portable_path(path, root),
                    image_sha256=sha256_file(path),
                    label=CLASSES[y],
                    pseudo_nor=bool(y == 2 and debug["n_rel_off_unknown"] > 0),
                    city=_seq_of(native).split("/")[0],
                    session=_seq_of(native),
                )
            )
        membership = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        counts = Counter(row["label"] for row in rows)
        summaries[split] = dict(
            annotation=portable_path(label_dir / f"DTLD_{split}.json", root),
            annotation_sha256=sha256_file(label_dir / f"DTLD_{split}.json"),
            n_native=len(entries),
            n_evaluated=len(rows),
            excluded_no_valid_lamp=len(skipped),
            excluded_native_paths=skipped,
            class_counts={name: counts[name] for name in CLASSES},
            pseudo_nor=sum(row["pseudo_nor"] for row in rows),
            n_sessions=len({row["session"] for row in rows}),
            city_counts=dict(sorted(Counter(row["city"] for row in rows).items())),
            membership_sha256=hashlib.sha256(membership).hexdigest(),
        )
        splits[split] = rows
    expected = [row["native_path"] for row in splits["test"]]
    actual = [item["entry"]["image_path"].replace("\\", "/").lstrip("./") for item in dataset.items]
    labels = [CLASSES[item["y"]] for item in dataset.items]
    if actual != expected or labels != [row["label"] for row in splits["test"]]:
        raise ValueError("Test loader membership/labels disagree with the native annotation audit")
    by_hash = defaultdict(lambda: {"train": [], "test": []})
    for split, rows in splits.items():
        for row in rows:
            by_hash[row["image_sha256"]][split].append(row)
    cross = [group for group in by_hash.values() if group["train"] and group["test"]]
    overlap = dict(
        native_images=len({r["native_path"] for r in splits["train"]} & set(expected)),
        basenames=len(
            {Path(r["image"]).name for r in splits["train"]} & {Path(r["image"]).name for r in splits["test"]}
        ),
        sessions=len({r["session"] for r in splits["train"]} & {r["session"] for r in splits["test"]}),
        exact_file_cross_split_pairs=sum(len(g["train"]) * len(g["test"]) for g in cross),
        exact_file_cross_split_groups=cross,
        exact_file_within_split_pairs={
            split: sum(len(g[split]) * (len(g[split]) - 1) // 2 for g in by_hash.values())
            for split in ("train", "test")
        },
        exact_file_conflicting_label_groups=sum(
            len({r["label"] for rows in g.values() for r in rows}) > 1
            for g in by_hash.values()
            if sum(map(len, g.values())) > 1
        ),
    )
    scope = (
        "native full-frame RGB JPEG bytes"
        if cfg["data"].get("preprocessing") == "letterbox"
        else "prepared JPEG bytes"
    )
    return dict(splits=summaries, overlap=overlap, image_hash_scope=scope), splits


@torch.no_grad()
def predict_heads(model, heads, loader, device, repo_root, progress=None):
    """Reuse the identical frozen encoder, preserving each head's raw logits.

    The first batch is also evaluated through separate complete-model forwards
    for every head. Bitwise equality is required before proceeding.
    """
    model.eval()
    for head in heads:
        head.eval()
    ys, logits, cities, sizes, paths = [], [[] for _ in heads], [], [], []
    verification, original_head = [], model.head
    started = time.monotonic()
    processed, last_update = 0, 0.0
    for batch_index, batch in enumerate(tqdm(loader, desc="DTLD official test", mininterval=5)):
        image = batch["image"].to(device, non_blocking=True).float().div_(255.0)
        metadata = batch_model_kwargs(batch, device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            features = {key: value.float() for key, value in model.backbone(image).items()}
            outputs = [
                head(
                    features["patches"],
                    features["cls"],
                    patches_mid=features["patches_mid"],
                    cls_mid=features["cls_mid"],
                    **metadata,
                )[0]
                for head in heads
            ]
            if batch_index == 0:
                try:
                    for index, head in enumerate(heads):
                        model.head = head
                        reference = model(image, **metadata)["logits"]
                        torch.testing.assert_close(outputs[index], reference, rtol=0, atol=0)
                        verification.append(dict(member=index, max_abs_logit_difference=0.0))
                finally:
                    model.head = original_head
        ys.append(batch["label"].numpy())
        for collected, output in zip(logits, outputs):
            collected.append(output.float().cpu().numpy())
        cities.extend(batch["city"])
        sizes.append(batch["max_lamp_h"].numpy())
        paths.extend(portable_path(path, repo_root) for path in batch["path"])
        processed += len(batch["label"])
        elapsed = time.monotonic() - started
        if progress is not None and (
            batch_index == 0 or elapsed - last_update >= 30 or processed == len(loader.dataset)
        ):
            progress(
                dict(
                    status="evaluating",
                    images_done=processed,
                    images_total=len(loader.dataset),
                    elapsed_seconds=elapsed,
                    estimated_remaining_seconds=elapsed * (len(loader.dataset) - processed) / processed,
                )
            )
            last_update = elapsed
    return (
        np.concatenate(ys),
        [np.concatenate(member) for member in logits],
        dict(cities=cities, max_lamp_h=np.concatenate(sizes), paths=paths),
        dict(first_batch_images=len(ys[0]), members=verification, elapsed_seconds=time.monotonic() - started),
    )
