"""Render verified test predictions as labeled images and offline review galleries."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
DIAGNOSTICS = REPO / "runs/generalization_study/diagnostic_tests"
CLASSES = ("RR", "RG", "NoR")
MODELS = ("v5_fold0", "B", "C")
WIDTH = 1600
BG, PANEL, TEXT, MUTED = "#111827", "#1f2937", "#f9fafb", "#b7c4d6"
CYAN, GRAY, ORANGE = "#67e8f9", "#c4cbd5", "#fbbf24"
CLASS_COLORS = ("#fb7185", "#4ade80", "#a5b4fc")
MEANINGS = {
    "RR": "Relevant red / yellow / red-yellow",
    "RG": "Relevant green (with no relevant RR signal)",
    "NoR": "No relevant RR/RG signal; off/unknown maps here",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value, domain):
    value = str(value).replace("\\", "/")
    return Path(value).stem if domain == "DTLD" else value


def predictions(path, domain):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            paths, labels, logits = data["paths"].tolist(), data["labels"].copy(), data["logits"].copy()
    else:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        paths = [row["file"] for row in rows]
        labels = np.array([CLASSES.index(row["label"]) for row in rows])
        logits = np.array([[float(row["logit_" + name]) for name in CLASSES] for row in rows])
    ids = [canonical(path, domain) for path in paths]
    if len(ids) != len(set(ids)) or logits.shape != (len(ids), 3) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid prediction cache: {path}")
    if labels.shape != (len(ids),) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError(f"Invalid labels: {path}")
    return {key: (int(label), logit) for key, label, logit in zip(ids, labels, logits)}


def softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = np.exp(values - values.max())
    return values / values.sum()


def lamp_label(lamps):
    relevant = [lamp["state"] for lamp in lamps if lamp["relevant"]]
    if any(state in ("red", "yellow", "red_yellow") for state in relevant):
        return "RR"
    return "RG" if "green" in relevant else "NoR"


def select_dtld(entries, seed):
    """Balance labels and cycle through cities/sessions, independent of predictions."""
    rng = random.Random(seed)
    selected = []
    for label, count in zip(CLASSES, (34, 33, 33)):
        groups = defaultdict(lambda: defaultdict(list))
        for entry in entries:
            if entry["label"] == label:
                groups[entry["city"]][entry["session"]].append(entry)
        queues = {}
        for city in sorted(groups):
            sessions = sorted(groups[city])
            rng.shuffle(sessions)
            for session in sessions:
                rng.shuffle(groups[city][session])
            queue = []
            while sessions:
                for session in sessions:
                    if groups[city][session]:
                        queue.append(groups[city][session].pop())
                sessions = [session for session in sessions if groups[city][session]]
            queues[city] = queue
        cities = sorted(queues)
        rng.shuffle(cities)
        current = []
        while len(current) < count:
            for city in cities:
                if queues[city] and len(current) < count:
                    current.append(queues[city].pop(0))
            if not any(queues.values()) and len(current) < count:
                raise ValueError(f"Insufficient DTLD examples for {label}")
        selected.extend(current)
    return sorted(selected, key=lambda entry: entry["native_path"])


def prepare(seed):
    frozen = read_json(DIAGNOSTICS / "plan.json")
    baseline = read_json(REPO / "runs/v5/evaluations/dtld_test/plan.json")["checkpoints"][0]
    model_info = {
        "v5_fold0": dict(
            checkpoint=baseline["checkpoint"],
            sha256=baseline["checkpoint_sha256"],
            epoch=baseline["epoch"],
            architecture="v5",
            weights="EMA",
        ),
    }
    for job in frozen["jobs"]:
        if job["name"] in MODELS:
            model_info[job["name"]] = dict(
                checkpoint=Path(job["checkpoint"]).relative_to(REPO).as_posix(),
                sha256=job["sha256"],
                epoch=job["epoch"],
                architecture=job["architecture"],
                weights="EMA",
            )
    for name, info in model_info.items():
        if sha256(REPO / info["checkpoint"]) != info["sha256"]:
            raise ValueError(f"Checkpoint changed: {name}")

    inputs = [
        DIAGNOSTICS / "dtld_membership.json",
        DIAGNOSTICS / "vzc_test_labels.json",
        REPO / "datasets/DTLD/v2.0/DTLD_test.json",
        REPO / "datasets/VZC_TLD/labels/test_v1.json",
        REPO / "metadata/atlas_labels.json",
    ]
    for path in inputs[2:]:
        expected = frozen["input_sha256"][path.relative_to(REPO).as_posix()]
        if sha256(path) != expected:
            raise ValueError(f"Annotations changed after evaluation: {path}")

    native = {Path(entry["image_path"]).stem: entry for entry in read_json(inputs[2])["images"]}
    dtld = []
    for entry in select_dtld(read_json(inputs[0])["test"], seed):
        identity = Path(entry["native_path"]).stem
        lamps = []
        for box in native[identity].get("labels", []):
            attrs = box.get("attributes", {})
            if attrs.get("relevance") not in ("relevant", "not_relevant") or box["w"] < 1 or box["h"] < 1:
                continue
            xyxy = [
                max(0, box["x"]),
                max(0, box["y"]),
                min(2047, box["x"] + box["w"]),
                min(1023, box["y"] + box["h"]),
            ]
            if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
                continue
            lamps.append(
                dict(
                    box=xyxy,
                    relevant=attrs["relevance"] == "relevant",
                    state=attrs.get("state", "unknown"),
                    pictogram=attrs.get("pictogram", "unknown"),
                )
            )
        if lamp_label(lamps) != entry["label"]:
            raise ValueError(f"DTLD global label disagrees with native lamps: {identity}")
        dtld.append(
            dict(
                id=identity,
                image=entry["image"],
                image_sha256=entry["image_sha256"],
                gt=entry["label"],
                slice=entry["city"],
                session=entry["session"],
                pseudo_nor=entry["pseudo_nor"],
                uncertain=False,
                lamps=lamps,
                gt_source="Official DTLD test annotations",
            )
        )

    atlas = []
    labels = read_json(inputs[4])
    for identity, entry in labels["images"].items():
        if entry["label"] not in CLASSES:
            raise ValueError(f"Missing ATLAS label: {identity}")
        atlas.append(
            dict(
                id=identity,
                image=(Path(labels["image_root"]) / identity).as_posix(),
                image_sha256=entry["sha256"],
                gt=entry["label"],
                slice=Path(identity).parts[-3],
                session="",
                pseudo_nor=False,
                uncertain=bool(entry.get("uncertain")),
                lamps=[],
                gt_source="Manual ATLAS benchmark label",
                notes=entry.get("notes", ""),
            )
        )

    coco = read_json(inputs[3])
    categories = {item["id"]: item["name"] for item in coco["categories"]}
    annotations = defaultdict(list)
    for entry in coco["annotations"]:
        name = categories[entry["category_id"]]
        x, y, w, h = entry["bbox"]
        annotations[entry["image_id"]].append(
            dict(
                box=[x, y, x + w, y + h],
                relevant=name.startswith("relevant_"),
                state=name.rsplit("_", 1)[-1],
                pictogram="",
            )
        )
    vzc = []
    for entry in read_json(inputs[1])["images"]:
        if entry["split"] != "test":
            raise ValueError("Unexpected VZC training image in test snapshot")
        lamps = annotations[entry["image_id"]]
        if lamp_label(lamps) != entry["class_name"]:
            raise ValueError(f"VZC label disagrees with COCO lamps: {entry['file']}")
        vzc.append(
            dict(
                id=entry["file"],
                image="datasets/VZC_TLD/" + entry["file"],
                image_sha256=entry["sha256"],
                gt=entry["class_name"],
                slice="published test",
                session="",
                pseudo_nor=entry["pseudo_nor"],
                uncertain=False,
                lamps=lamps,
                gt_source="Published VZC-TLD test annotations",
            )
        )
    domains = dict(DTLD=dtld, ATLAS=atlas, VZC_TLD=vzc)
    if [len(entries) for entries in domains.values()] != [100, 528, 598]:
        raise ValueError("Review membership must be 100 DTLD, 528 ATLAS, and 598 VZC-TLD")
    prediction_sources = {}
    for domain, entries in domains.items():
        for model in MODELS:
            if model == "v5_fold0":
                relative = dict(
                    DTLD="dtld_test/member0/predictions.npz",
                    ATLAS="atlas/v5_fold0/predictions.csv",
                    VZC_TLD="vzc/v5_fold0/predictions.npz",
                )[domain]
                source = REPO / "runs/v5/evaluations" / relative
                expected_hash = frozen["input_sha256"][source.relative_to(REPO).as_posix()]
            else:
                source = DIAGNOSTICS / domain / model / "predictions.npz"
                report = read_json(source.parent / "report.json")
                if report["checkpoint_sha256"] != model_info[model]["sha256"] or report["temperature"] != 1:
                    raise ValueError("Prediction cache checkpoint or temperature differs")
                expected_hash = report["predictions_sha256"]
            if sha256(source) != expected_hash:
                raise ValueError(f"Prediction cache changed: {source}")
            cached = predictions(source, domain)
            expected_ids = {entry["id"] for entry in entries}
            if domain != "DTLD" and model != "v5_fold0" and set(cached) != expected_ids:
                raise ValueError("Incomplete external prediction membership")
            prediction_sources[domain + "/" + model] = dict(
                file=source.relative_to(REPO).as_posix(),
                sha256=expected_hash,
            )
            for entry in entries:
                label, logits = cached[entry["id"]]
                if CLASSES[label] != entry["gt"]:
                    raise ValueError(f"Prediction/ground truth mismatch: {entry['id']}")
                probs = softmax(logits)
                pred = CLASSES[int(probs.argmax())]
                entry.setdefault("models", {})[model] = dict(
                    pred=pred,
                    correct=pred == entry["gt"],
                    confidence=float(probs.max()),
                    probabilities=probs.tolist(),
                    logits=logits.tolist(),
                )
        for index, entry in enumerate(entries, 1):
            entry["dataset"] = domain
            entry["index"] = index
            entry["disagreement"] = len({row["pred"] for row in entry["models"].values()}) > 1
            entry["any_error"] = any(not row["correct"] for row in entry["models"].values())
            entry["short_id"] = hashlib.sha256(entry["id"].encode()).hexdigest()[:10]
            for model, row in entry["models"].items():
                filename = f"{index:04d}_GT-{entry['gt']}_PRED-{row['pred']}_{entry['short_id']}.jpg"
                row["image"] = f"{model}/{domain}/images/{filename}"
                row["thumbnail"] = f"{model}/{domain}/thumbnails/{filename}"
    plan = dict(
        seed=seed,
        dtld_selection="34 RR, 33 RG, 33 NoR; cycle cities and sessions; independent of predictions",
        temperature=1.0,
        prediction_rule="argmax of raw softmax",
        models=model_info,
        inference="Exact cached outputs from completed frozen-checkpoint evaluations; no new model fitting",
        prediction_sources=prediction_sources,
        input_sha256={path.relative_to(REPO).as_posix(): sha256(path) for path in inputs},
        images={
            domain: [
                {key: entry[key] for key in ("id", "image", "image_sha256", "gt", "slice", "session")}
                for entry in entries
            ]
            for domain, entries in domains.items()
        },
    )
    return domains, plan


@lru_cache(maxsize=32)
def font(size, bold=False):
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    candidates = (
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def text(draw, xy, value, size=22, fill=TEXT, bold=False):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def fit(draw, value, max_width, size=20):
    while draw.textlength(value, font=font(size)) > max_width:
        value = value[:-5] + "..." if len(value) > 5 else value[:1]
    return value


def render(entry, model, info, photo):
    result = entry["models"][model]
    top, footer = 324, 70
    photo_h = round(photo.height * WIDTH / photo.width)
    # Show all annotated lamps in the scene; enlarge up to 12, prioritizing relevant lamps.
    ranked = sorted(enumerate(entry["lamps"], 1), key=lambda pair: (not pair[1]["relevant"], pair[0]))[:12]
    rows = (len(ranked) + 5) // 6
    crop_h = 256 if rows else 0
    canvas = Image.new("RGB", (WIDTH, top + photo_h + (50 + rows * crop_h if rows else 0) + footer), BG)
    draw = ImageDraw.Draw(canvas)
    text(draw, (28, 16), f"{model}  |  {entry['dataset']}  |  Image {entry['index']:04d}", 30, bold=True)
    verdict = "CORRECT" if result["correct"] else "INCORRECT"
    color = "#4ade80" if result["correct"] else "#fb7185"
    draw.rounded_rectangle(
        (1330, 12, 1572, 61), radius=12, fill="#16382d" if result["correct"] else "#4a2330"
    )
    text(draw, (1360, 19), verdict, 26, color, True)
    text(draw, (28, 64), fit(draw, entry["id"], 1540, 19), 19, MUTED)
    draw.rounded_rectangle((24, 102, 790, 222), radius=12, fill=PANEL)
    draw.rounded_rectangle((810, 102, 1576, 222), radius=12, fill=PANEL)
    text(draw, (44, 109), "GROUND TRUTH", 19, CYAN, True)
    text(draw, (44, 132), entry["gt"], 43, CLASS_COLORS[CLASSES.index(entry["gt"])], True)
    text(draw, (183, 148), MEANINGS[entry["gt"]], 21)
    note = " | MARKED UNCERTAIN" if entry["uncertain"] else ""
    if entry["pseudo_nor"]:
        note += " | Relevant off/unknown-only"
    text(draw, (44, 191), entry["gt_source"] + note, 17, ORANGE if note else MUTED)
    text(draw, (831, 109), "MODEL PREDICTION", 19, CYAN, True)
    text(draw, (831, 132), result["pred"], 43, CLASS_COLORS[CLASSES.index(result["pred"])], True)
    text(draw, (970, 145), f"{100 * result['confidence']:.1f}% probability", 25, bold=True)
    text(
        draw, (831, 191), f"{info['architecture']} | epoch {info['epoch']} | EMA | raw softmax T=1", 18, MUTED
    )
    for index, (label, probability) in enumerate(zip(CLASSES, result["probabilities"])):
        x = 28 + index * 522
        text(draw, (x, 233), f"P({label})  {100 * probability:.1f}%", 21, CLASS_COLORS[index], True)
        draw.rounded_rectangle((x, 267, x + 475, 277), radius=5, fill="#374151")
        if probability > 0.002:
            draw.rounded_rectangle(
                (x, 267, x + max(5, 475 * probability), 277), radius=5, fill=CLASS_COLORS[index]
            )
    legend = (
        "GT lamp boxes: CYAN = relevant, GRAY = not relevant. Numbers match enlarged crops."
        if entry["lamps"]
        else "ATLAS has image-level manual labels; no relevance box annotations are available."
    )
    text(draw, (28, 289), legend, 19, MUTED)
    visible = photo.resize((WIDTH, photo_h), Image.Resampling.LANCZOS)
    if model == "v5_fold0" and entry["dataset"] == "DTLD":
        # The cached v5 prediction used the legacy side-cropped JPEG, not the full photo shown here.
        tint = Image.new("RGBA", visible.size, (0, 0, 0, 0))
        overlay = ImageDraw.Draw(tint)
        left = round(114 * WIDTH / 2048)
        right = WIDTH - left
        overlay.rectangle((0, 0, left, photo_h), fill=(15, 23, 42, 170))
        overlay.rectangle((right, 0, WIDTH, photo_h), fill=(15, 23, 42, 170))
        visible = Image.alpha_composite(visible.convert("RGBA"), tint).convert("RGB")
        canvas.paste(visible, (0, top))
        for x in (left, right):
            draw.line((x, top, x, top + photo_h), fill=ORANGE, width=3)
    else:
        canvas.paste(visible, (0, top))
    sx, sy = WIDTH / photo.width, photo_h / photo.height
    for number, lamp in enumerate(entry["lamps"], 1):
        x1, y1, x2, y2 = lamp["box"]
        box = [
            max(0, x1 * sx),
            top + max(0, y1 * sy),
            min(WIDTH - 1, x2 * sx),
            top + min(photo_h - 1, y2 * sy),
        ]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        color = CYAN if lamp["relevant"] else GRAY
        draw.rectangle(box, outline=color, width=3 if lamp["relevant"] else 1)
        lx, ly = min(WIDTH - 43, box[0]), max(top, box[1] - 23)
        draw.rectangle((lx, ly, lx + 40, ly + 22), fill=BG)
        text(draw, (lx + 3, ly - 1), str(number), 16, color, True)
    if rows:
        strip_y = top + photo_h
        text(
            draw,
            (28, strip_y + 11),
            f"GROUND-TRUTH LAMP CROPS  |  {len(ranked)} of {len(entry['lamps'])} shown; all boxes marked above",
            21,
            bold=True,
        )
        for position, (number, lamp) in enumerate(ranked):
            x, y = 24 + (position % 6) * 260, strip_y + 48 + (position // 6) * crop_h
            color = CYAN if lamp["relevant"] else GRAY
            draw.rounded_rectangle((x, y, x + 250, y + 241), radius=8, fill=PANEL, outline=color, width=2)
            x1, y1, x2, y2 = lamp["box"]
            margin = max(12, (x2 - x1) * 0.8, (y2 - y1) * 0.4)
            bounds = (
                max(0, int(x1 - margin)),
                max(0, int(y1 - margin)),
                min(photo.width, int(x2 + margin + 1)),
                min(photo.height, int(y2 + margin + 1)),
            )
            crop = photo.crop(bounds)
            scale = min(228 / crop.width, 160 / crop.height)
            enlarged = crop.resize(
                (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                Image.Resampling.BICUBIC,
            )
            px, py = x + (250 - enlarged.width) // 2, y + 9 + (160 - enlarged.height) // 2
            canvas.paste(enlarged, (px, py))
            draw.rectangle(
                (
                    px + (x1 - bounds[0]) * scale,
                    py + (y1 - bounds[1]) * scale,
                    px + (x2 - bounds[0]) * scale,
                    py + (y2 - bounds[1]) * scale,
                ),
                outline=color,
                width=2,
            )
            text(
                draw,
                (x + 12, y + 176),
                f"#{number:02d}  {'RELEVANT' if lamp['relevant'] else 'NOT RELEVANT'}",
                17,
                color,
                True,
            )
            text(
                draw,
                (x + 12, y + 201),
                fit(draw, f"{lamp['state']}  |  {lamp['pictogram'] or 'pictogram unavailable'}", 227, 15),
                15,
            )
    foot_y = canvas.height - footer
    if model == "v5_fold0" and entry["dataset"] == "DTLD":
        detail = "v5 input: side-cropped 720x1280 JPEG. Amber boundaries mark its retained field of view."
    elif model in ("B", "C"):
        detail = "Model input: full-frame letterbox at 720x1280. The photo above is shown without padding."
    else:
        detail = "Model input: full frame resized to 720x1280. The photo above preserves its original aspect ratio."
    text(
        draw,
        (28, foot_y + 5),
        detail,
        19,
        ORANGE if model == "v5_fold0" and entry["dataset"] == "DTLD" else MUTED,
    )
    text(
        draw,
        (28, foot_y + 35),
        f"Slice: {entry['slice']}  |  Boxes and crop attributes are ground truth; prediction is image-level.",
        18,
        MUTED,
    )
    return canvas


GALLERY = r"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#101722;color:#eef3fa;font:16px system-ui,sans-serif}
header{padding:26px 32px 18px;border-bottom:1px solid #334155;background:#152131}h1{margin:0 0 8px;font-size:30px}
p{line-height:1.5;margin:6px 0;color:#bac8da}a{color:#87dcff}nav{display:flex;gap:18px;margin-top:16px;flex-wrap:wrap}
.filters{display:flex;gap:12px;flex-wrap:wrap;padding:18px 32px;background:#182638;position:sticky;top:0;z-index:2}
label{font-size:13px;color:#c0ccdd;display:flex;flex-direction:column;gap:5px}select,input,button{font:inherit;background:#0e1828;color:#f1f5f9;border:1px solid #53627a;border-radius:7px;padding:8px}
input{width:240px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}.count{padding:12px 32px;color:#a7bad0}
main{padding:0 32px 32px}.item{margin-bottom:24px;background:#182334;border:1px solid #34435a;border-radius:12px;overflow:hidden}
.meta{padding:14px 18px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}.id{overflow-wrap:anywhere;color:#c1cee0;font-size:13px}
.models{display:grid;grid-template-columns:repeat(__COLUMNS__,minmax(0,1fr));gap:1px;background:#34435a}.model{background:#152031;padding:12px;min-width:0}
.model h3{margin:0 0 8px;font-size:17px;display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap}.ok{color:#6ee7ad}.bad{color:#fda4af}
.model img{width:100%;height:auto;display:block;border-radius:7px;cursor:zoom-in}.probs{font-size:13px;line-height:1.7;margin-top:8px;color:#c6d2e3}
.badge{border:1px solid #596b87;padding:4px 8px;border-radius:5px;font-weight:700;font-size:13px}.pager{display:flex;gap:14px;align-items:center;justify-content:center;padding:10px 24px 30px}
dialog{width:min(96vw,1800px);max-height:96vh;padding:0;border:1px solid #64748b;background:#101722;color:white;border-radius:10px}dialog::backdrop{background:#000c}
.dialogbar{position:sticky;top:0;padding:12px;display:flex;gap:10px;align-items:center;background:#182334}.dialogbar span{flex:1}dialog img{display:block;width:100%;height:auto}
@media(max-width:850px){.models{grid-template-columns:1fr}.filters{position:static}main,header{padding-left:14px;padding-right:14px}}
</style>
<header><h1>__TITLE__</h1><p>Same images, fixed checkpoints. Ground truth and prediction are printed on every exported JPEG.</p>
<p>DTLD: 100 class-balanced test frames (34 RR, 33 RG, 33 NoR). ATLAS: all 528 manual benchmark images. VZC-TLD: all 598 published test images.</p>
<p>RR = relevant red/yellow/red-yellow; RG = relevant green with no relevant RR; NoR = no relevant RR/RG evidence, including off/unknown-only cases. Probabilities use raw softmax at T=1.</p>
<p>Cyan boxes mark ground-truth relevant lamps; gray boxes mark other annotated lamps. Enlarged crops also show ground-truth attributes. These are image-classification models.</p>
<nav>__NAV__</nav></header>
<div class="filters">
<label>Dataset<select id="dataset"><option value="">All datasets</option><option>DTLD</option><option>ATLAS</option><option>VZC_TLD</option></select></label>
<label>Ground truth<select id="gt"><option value="">Any class</option><option>RR</option><option>RG</option><option>NoR</option></select></label>
<label>Prediction<select id="pred"><option value="">Any class</option><option>RR</option><option>RG</option><option>NoR</option></select></label>
<label>Show<select id="mode"><option value="all">All images</option><option value="error">Errors__ERROR_LABEL__</option><option value="correct">Correct__CORRECT_LABEL__</option><option value="disagree">Models disagree</option><option value="off">Relevant off/unknown-only</option><option value="uncertain">Uncertain manual label</option></select></label>
<label>City / camera<select id="slice"><option value="">All slices</option></select></label>
<label>Sort<select id="sort"><option value="original">Dataset / image order</option><option value="confidence">Highest prediction probability first</option></select></label>
<label>Image name / session<input id="search" placeholder="Search"></label>
</div><div class="count" id="count"></div><main id="items"></main>
<div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div>
<dialog id="zoom"><div class="dialogbar"><button id="zoomprev">Previous image</button><span id="zoomtitle"></span><button id="zoomnext">Next image</button><a id="original" target="_blank">Open JPEG</a><button id="close">Close</button></div><img id="zoomimage" alt="Ground truth and model prediction"></dialog>
<script id="records" type="application/json">__DATA__</script>
<script>
const records=JSON.parse(document.getElementById('records').textContent), models=__MODELS__, prefix=__PREFIX__;
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pageSize=models.length===1?24:12;let page=0, filtered=[],zoomIndex=0,zoomModel=models[0];
for(const slice of [...new Set(records.map(r=>r.slice))].sort()){let o=document.createElement('option');o.value=slice;o.textContent=slice;$('slice').append(o)}
function filter(){page=0;const search=$('search').value.toLowerCase();filtered=records.filter(r=>{
let visible=models.map(m=>r.models[m]);return (!$('dataset').value||r.dataset===$('dataset').value)&&(!$('gt').value||r.gt===$('gt').value)&&
(!$('pred').value||visible.some(m=>m.pred===$('pred').value))&&(!$('slice').value||r.slice===$('slice').value)&&
(!search||(r.id+' '+r.session).toLowerCase().includes(search))&&
($('mode').value==='all'||($('mode').value==='error'&&visible.some(m=>!m.correct))||($('mode').value==='correct'&&visible.every(m=>m.correct))||
($('mode').value==='disagree'&&r.disagreement)||($('mode').value==='off'&&r.pseudo_nor)||($('mode').value==='uncertain'&&r.uncertain))});
if($('sort').value==='confidence')filtered.sort((a,b)=>Math.max(...models.map(m=>b.models[m].confidence))-Math.max(...models.map(m=>a.models[m].confidence)));draw()}
function draw(){const pages=Math.max(1,Math.ceil(filtered.length/pageSize));page=Math.min(page,pages-1);
$('count').textContent=filtered.length+' of '+records.length+' images | '+models.join(' / ')+' | Click an image to inspect the full-size JPEG.';
$('items').innerHTML=filtered.slice(page*pageSize,(page+1)*pageSize).map((r,i)=>`<article class="item"><div class="meta"><strong>${esc(r.dataset)} #${String(r.index).padStart(4,'0')}</strong><span class="badge">GROUND TRUTH: ${esc(r.gt)}</span><span>${esc(r.slice)}</span>${r.disagreement?'<span class="badge">Models disagree</span>':''}${r.uncertain?'<span class="badge">UNCERTAIN LABEL</span>':''}${r.pseudo_nor?'<span class="badge">Relevant off/unknown-only</span>':''}<span class="id">${esc(r.id)}</span></div><div class="models">${models.map(m=>{let p=r.models[m];return `<div class="model"><h3><span>${esc(m)} → ${esc(p.pred)} (${(p.confidence*100).toFixed(1)}%)</span><span class="${p.correct?'ok':'bad'}">${p.correct?'CORRECT':'INCORRECT'}</span></h3><img loading="lazy" src="${esc(prefix+p.thumbnail)}" data-index="${page*pageSize+i}" data-model="${esc(m)}" alt="${esc(m)}: ground truth ${r.gt}, prediction ${p.pred}"><div class="probs">${['RR','RG','NoR'].map((c,j)=>c+': '+(100*p.probabilities[j]).toFixed(1)+'%').join(' &nbsp; | &nbsp; ')}</div></div>`}).join('')}</div></article>`).join('');
$('page').textContent='Page '+(page+1)+' / '+pages;$('prev').disabled=page===0;$('next').disabled=page>=pages-1}
function showZoom(index,model){zoomIndex=index;zoomModel=model;let r=filtered[index],p=r.models[model];$('zoomtitle').textContent=model+' | '+r.dataset+' #'+r.index+' | GT '+r.gt+' → '+p.pred;$('zoomimage').src=prefix+p.image;$('original').href=prefix+p.image;$('zoomprev').disabled=index===0;$('zoomnext').disabled=index===filtered.length-1;if(!$('zoom').open)$('zoom').showModal()}
$('items').addEventListener('click',e=>{if(e.target.dataset.index!==undefined)showZoom(Number(e.target.dataset.index),e.target.dataset.model)});
for(const id of ['dataset','gt','pred','mode','slice','sort','search'])$(id).addEventListener(id==='search'?'input':'change',filter);
$('prev').onclick=()=>{page--;draw();window.scrollTo(0,0)};$('next').onclick=()=>{page++;draw();window.scrollTo(0,0)};
$('zoomprev').onclick=()=>{if(zoomIndex>0)showZoom(zoomIndex-1,zoomModel)};$('zoomnext').onclick=()=>{if(zoomIndex<filtered.length-1)showZoom(zoomIndex+1,zoomModel)};$('close').onclick=()=>$('zoom').close();
document.addEventListener('keydown',e=>{if($('zoom').open){if(e.key==='ArrowLeft')$('zoomprev').click();if(e.key==='ArrowRight')$('zoomnext').click()}});filter();
</script></html>"""


def gallery(path, entries, models, prefix):
    title = "Model comparison: v5_fold0 / B / C" if len(models) > 1 else models[0] + " — visual review"
    nav = '<a href="../index.html">Compare all three</a>' if len(models) == 1 else ""
    nav += "".join(f'<a href="{prefix}{name}/index.html">{name} gallery</a>' for name in MODELS)
    # Embed the data so the gallery also works when opened directly from File Explorer.
    payload = (
        json.dumps(entries, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    replacements = dict(
        TITLE=html.escape(title),
        COLUMNS=str(len(models)),
        NAV=nav,
        DATA=payload,
        MODELS=json.dumps(models),
        PREFIX=json.dumps(prefix),
        ERROR_LABEL=" (any model)" if len(models) > 1 else "",
        CORRECT_LABEL=" (all models)" if len(models) > 1 else "",
    )
    value = GALLERY
    for key, replacement in replacements.items():
        value = value.replace("__" + key + "__", replacement)
    path.write_text(value, encoding="utf-8")


def export_one(entry, out, info):
    source = REPO / entry["image"]
    if sha256(source) != entry["image_sha256"]:
        raise ValueError(f"Source image changed: {source}")
    with Image.open(source) as original:
        photo = original.convert("RGB")
    for model in MODELS:
        result = entry["models"][model]
        canvas = render(entry, model, info[model], photo)
        canvas.save(out / result["image"], quality=90, subsampling=0)
        thumbnail = canvas.copy()
        thumbnail.thumbnail((560, 900), Image.Resampling.LANCZOS)
        thumbnail.save(out / result["thumbnail"], quality=82)
    return entry["dataset"]


def exports(out, domains, plan, workers):
    entries = [entry for group in domains.values() for entry in group]
    for model in MODELS:
        for domain in domains:
            for folder in ("images", "thumbnails"):
                (out / model / domain / folder).mkdir(parents=True, exist_ok=True)
    gallery_records = []
    for entry in entries:
        gallery_records.append(
            {key: value for key, value in entry.items() if key not in ("lamps", "image_sha256")}
        )
    gallery(out / "index.html", gallery_records, list(MODELS), "")
    summaries = {}
    for model in MODELS:
        gallery(out / model / "index.html", gallery_records, [model], "../")
        rows = []
        summaries[model] = {}
        for domain, group in domains.items():
            counts = Counter(entry["gt"] for entry in group)
            correct = sum(entry["models"][model]["correct"] for entry in group)
            summaries[model][domain] = dict(
                n=len(group),
                correct=correct,
                incorrect=len(group) - correct,
                classes=dict(counts),
                disagreements=sum(e["disagreement"] for e in group),
            )
            for entry in group:
                result = entry["models"][model]
                rows.append(
                    dict(
                        dataset=domain,
                        index=entry["index"],
                        id=entry["id"],
                        source_image=entry["image"],
                        ground_truth=entry["gt"],
                        prediction=result["pred"],
                        correct=result["correct"],
                        **{f"P_{c}": result["probabilities"][i] for i, c in enumerate(CLASSES)},
                        **{f"logit_{c}": result["logits"][i] for i, c in enumerate(CLASSES)},
                        slice=entry["slice"],
                        session=entry["session"],
                        uncertain=entry["uncertain"],
                        pseudo_nor=entry["pseudo_nor"],
                        models_disagree=entry["disagreement"],
                        review_image=str(Path(result["image"]).relative_to(model)).replace("\\", "/"),
                    )
                )
        with (out / model / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(
            out / model / "manifest.json",
            dict(model=model, **plan["models"][model], datasets=summaries[model], temperature=1, images=rows),
        )
    write_json(out / "selection.json", plan)
    write_json(out / "summary.json", summaries)
    readme = [
        "# Manual model review",
        "",
        "Open **index.html** for a side-by-side comparison, or a model's index.html for its gallery.",
        "Each model has DTLD/images, ATLAS/images and VZC_TLD/images folders with standalone annotated JPEGs.",
        "",
        "The same 1,226 images are used for each model: 100 DTLD official test frames, all 528 manually labeled ATLAS benchmark images, and all 598 published VZC-TLD test images.",
        "DTLD selection uses seed 0 by default, balances classes (34 RR / 33 RG / 33 NoR), and cycles cities/sessions independently of model predictions. It is an illustrative review set, not an estimate of overall test accuracy.",
        "",
        "Ground-truth and predicted image classes, correctness, and all three raw softmax probabilities are printed on every image. Cyan/gray boxes and enlarged lamp crops are ground-truth annotations, not detections by these classification models. ATLAS has image labels only. The one uncertain ATLAS label remains included and is flagged.",
        "NoR includes relevant off/unknown-only scenes under the existing label policy. RR has priority over RG when both occur.",
        "",
        "v5 DTLD predictions use the existing cropped/resized JPEGs; amber boundaries indicate their field of view on the displayed original RGB frame. B/C use full-frame letterboxing. External v5 images were resized without side cropping.",
        "Predictions are the exact cached outputs from the completed fixed-checkpoint evaluations (EMA, T=1, argmax), not newly fitted or recalibrated predictions. selection.json records source prediction, annotation, image and checkpoint hashes. B/C remain diagnostic checkpoints that failed the original validation eligibility gate.",
        "",
        "| Model | DTLD correct / 100 | ATLAS correct / 528 | VZC-TLD correct / 598 |",
        "|---|---:|---:|---:|",
    ]
    for model in MODELS:
        readme.append(
            "| "
            + model
            + " | "
            + " | ".join(str(summaries[model][domain]["correct"]) for domain in domains)
            + " |"
        )
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    started = time.monotonic()
    status = dict(
        state="rendering",
        total_scenes=len(entries),
        total_images=len(entries) * len(MODELS),
        completed_scenes=0,
    )
    write_json(out / "status.json", status)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(lambda entry: export_one(entry, out, plan["models"]), entries), 1):
            if index % 50 == 0 or index == len(entries):
                status.update(completed_scenes=index, elapsed_seconds=round(time.monotonic() - started, 1))
                write_json(out / "status.json", status)
                print(
                    f"Rendered {index}/{len(entries)} scenes ({index * len(MODELS)} annotated images)",
                    flush=True,
                )
    # Verify membership, decode every JPEG, and confirm probabilities and labels in the final manifests.
    for model in MODELS:
        for domain, group in domains.items():
            for folder, key in (("images", "image"), ("thumbnails", "thumbnail")):
                files = {path.name for path in (out / model / domain / folder).glob("*.jpg")}
                if files != {Path(entry["models"][model][key]).name for entry in group}:
                    raise ValueError("Rendered file membership differs from the manifest")
                for entry in group:
                    with Image.open(out / entry["models"][model][key]) as image:
                        image.verify()
            for entry in group:
                result = entry["models"][model]
                if result["pred"] != CLASSES[int(np.argmax(result["logits"]))] or not np.isclose(
                    sum(result["probabilities"]), 1
                ):
                    raise ValueError("Rendered prediction differs from raw logits")
    status.update(
        state="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=round(time.monotonic() - started, 1),
        verified_jpegs=len(entries) * len(MODELS) * 2,
    )
    write_json(out / "status.json", status)
    print(f"Complete: {out / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "runs/manual_review")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--preview", action="store_true", help="render one image per dataset/model for visual inspection"
    )
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(REPO / "runs") or args.workers < 1:
        raise ValueError("Use an output directory inside runs and a positive worker count")
    if (out / "selection.json").exists():
        raise ValueError("Review already exists; choose a different output directory")
    domains, plan = prepare(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if args.preview:
        preview = out / "preview"
        preview.mkdir(exist_ok=True)
        for domain, entries in domains.items():
            # Prefer a mismatch to verify the error display, without changing final membership.
            entry = next((entry for entry in entries if entry["any_error"]), entries[0])
            with Image.open(REPO / entry["image"]) as original:
                photo = original.convert("RGB")
            for model in MODELS:
                render(entry, model, plan["models"][model], photo).save(
                    preview / f"{domain}_{model}.jpg", quality=90
                )
        print(f"Preview: {preview}")
    else:
        exports(out, domains, plan, args.workers)


if __name__ == "__main__":
    main()
