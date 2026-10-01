"""Local, blind RR/RG/NoR annotation. Images are read-only; labels live in runs.

python dinov3_global/scripts/label_atlas.py --images C:/path/to/ATLAS
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
import secrets
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image

CLASSES = ("RR", "RG", "NoR")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_IMAGES = Path.home() / "Desktop/CARLA_traffic_lights/ATLAS"
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_predictions(path):
    if not path:
        return {}
    result = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = r.get("file", "").replace("\\", "/")
            result[key] = {"pred": r.get("pred", r.get("pred_global")),
                           "prob": [float(r.get("P_" + c, r.get("p_" + c, 0)))
                                    for c in CLASSES]}
    return result


def atlas_boxes(path):
    # Existing ATLAS boxes aid inspection only; state does not define relevance.
    path = Path(path)
    if path.parent.name != "images":
        return []
    labels = path.parent.parent / "labels" / (path.stem + ".txt")
    if not labels.exists():
        return []
    names = ["circle_green", "circle_red", "off", "circle_red_yellow",
             "arrow_left_green", "circle_yellow", "arrow_right_red",
             "arrow_left_red", "arrow_straight_red", "arrow_left_red_yellow",
             "arrow_left_yellow", "arrow_straight_yellow", "arrow_right_red_yellow",
             "arrow_right_green", "arrow_right_yellow", "arrow_straight_green",
             "arrow_straight_left_green", "arrow_straight_red_yellow",
             "arrow_straight_left_red", "arrow_straight_left_yellow",
             "arrow_straight_left_red_yellow", "arrow_straight_right_red",
             "arrow_straight_right_red_yellow", "arrow_straight_right_yellow",
             "arrow_straight_right_green"]
    boxes = []
    for line in labels.read_text().splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        c = int(values[0]); x, y, w, h = map(float, values[1:5])
        boxes.append({"xywh": [x - w / 2, y - h / 2, w, h],
                      "state": names[c] if 0 <= c < len(names) else str(c)})
    return boxes


class AnnotationStore:
    def __init__(self, images, labels, predictions=None):
        self.root = Path(images).resolve()
        self.path = Path(labels).resolve()
        self.files = sorted(p.resolve() for p in self.root.rglob("*")
                            if p.suffix.lower() in EXTS and p.is_file()
                            and p.resolve().is_relative_to(self.root))
        if not self.files:
            raise ValueError(f"No images under {self.root}")
        if self.path.is_relative_to(self.root):
            raise ValueError("Save labels outside the read-only source image tree")
        self.keys = [p.relative_to(self.root).as_posix() for p in self.files]
        self.hashes = {k: sha256(p) for k, p in zip(self.keys, self.files)}
        self.lock = threading.Lock()
        raw_predictions = load_predictions(predictions)
        self.predictions = {}
        for k in self.keys:
            matches = [r for source, r in raw_predictions.items()
                       if source == k or k.endswith("/" + source)]
            if len(matches) == 1:
                self.predictions[k] = matches[0]
        self.data = {"schema_version": 1, "task": "ego_relevance_RR_RG_NoR",
                     "purpose": "evaluation", "image_root": str(self.root),
                     "class_names": list(CLASSES), "revision": 0,
                     "images": {k: {"label": None, "uncertain": False,
                                    "notes": "", "sha256": self.hashes[k]}
                                for k in self.keys}}
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding="utf-8"))
            if saved.get("schema_version") != 1 or saved.get("class_names") != list(CLASSES):
                raise ValueError("Unsupported annotation schema")
            if set(saved.get("images", {})) != set(self.keys):
                raise ValueError("Image manifest differs from saved annotations")
            for k in self.keys:
                if saved["images"][k].get("sha256") != self.hashes[k]:
                    raise ValueError(f"Image changed since labeling: {k}")
            self.data = saved
            self.data["image_root"] = str(self.root)
        else:
            self._write()

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        payload = json.dumps(self.data, indent=2, ensure_ascii=False, allow_nan=False)
        with tmp.open("w", encoding="utf-8") as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        if self.path.exists():
            backup = self.path.with_suffix(self.path.suffix + ".bak")
            backup_tmp = backup.with_suffix(backup.suffix + ".tmp")
            backup_tmp.write_bytes(self.path.read_bytes())
            os.replace(backup_tmp, backup)
        os.replace(tmp, self.path)

    def manifest(self):
        with self.lock:
            items = []
            for i, (k, p) in enumerate(zip(self.keys, self.files)):
                with Image.open(p) as im:
                    width, height = im.size
                items.append({"id": i, "file": k, "width": width, "height": height,
                              "boxes": atlas_boxes(p), **self.data["images"][k]})
            return {"revision": self.data["revision"], "items": items,
                    "labels_path": str(self.path)}

    def update(self, payload):
        i = payload.get("id")
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < len(self.files):
            raise ValueError("invalid image id")
        label, notes, uncertain = payload.get("label"), payload.get("notes", ""), payload.get("uncertain", False)
        if label is not None and label not in CLASSES:
            raise ValueError("label must be RR, RG, NoR, or null")
        if not isinstance(notes, str) or len(notes) > 4000 or not isinstance(uncertain, bool):
            raise ValueError("invalid notes or uncertainty")
        with self.lock:
            if payload.get("revision") != self.data["revision"]:
                raise RuntimeError("Labels changed in another tab; reload before saving")
            previous = copy.deepcopy(self.data)
            row = self.data["images"][self.keys[i]]
            row.update(label=label, notes=notes, uncertain=uncertain,
                       updated_at=datetime.now(timezone.utc).isoformat())
            self.data["revision"] += 1
            try:
                self._write()
            except OSError:
                self.data = previous
                raise
            return {"revision": self.data["revision"], "record": row}

    def export_csv(self):
        stream = io.StringIO(newline="")
        writer = csv.writer(stream)
        writer.writerow(["file", "label", "uncertain", "notes", "sha256"])
        with self.lock:
            for k in self.keys:
                r = self.data["images"][k]
                # Quote formula-like free text so spreadsheet import is inert.
                note = r["notes"]
                if note.startswith(("=", "+", "-", "@")):
                    note = "'" + note
                writer.writerow([k, r["label"] or "", r["uncertain"], note, r["sha256"]])
        return stream.getvalue().encode("utf-8-sig")


def make_handler(store, port):
    token = secrets.token_urlsafe(32)
    origin = f"http://127.0.0.1:{port}"

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if args and str(args[0]).startswith("POST"):
                super().log_message(fmt, *args)

        def send(self, code, payload, content_type="application/json", download=None):
            if content_type == "application/json":
                payload = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if download:
                self.send_header("Content-Disposition", f'attachment; filename="{download}"')
            self.end_headers(); self.wfile.write(payload)

        def trusted_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.trusted_host():
                return self.send(403, {"error": "invalid host"})
            url = urlparse(self.path)
            if url.path == "/":
                return self.send(200, Path(__file__).with_name("atlas_labeler.html").read_bytes(), "text/html; charset=utf-8")
            if url.path == "/api/items":
                return self.send(200, {**store.manifest(), "token": token})
            if url.path == "/api/export":
                return self.send(200, store.export_csv(), "text/csv; charset=utf-8", "atlas_labels.csv")
            if url.path == "/api/predictions":
                # Explicitly requested by the user via Reveal predictions.
                return self.send(200, store.predictions)
            if url.path == "/image":
                try:
                    i = int(parse_qs(url.query)["id"][0])
                    if not 0 <= i < len(store.files):
                        raise ValueError()
                    p = store.files[i]
                    if p.suffix.lower() in {".tif", ".tiff", ".bmp"}:
                        stream = io.BytesIO()
                        with Image.open(p) as im:
                            im.convert("RGB").save(stream, format="PNG")
                        return self.send(200, stream.getvalue(), "image/png")
                    return self.send(200, p.read_bytes(), mimetypes.guess_type(p)[0] or "image/jpeg")
                except (KeyError, ValueError):
                    return self.send(400, {"error": "invalid image id"})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            if (not self.trusted_host() or self.path != "/api/label"
                    or self.headers.get("X-Annotation-Token") != token
                    or self.headers.get("Origin", origin) not in (origin, f"http://localhost:{port}")):
                return self.send(403, {"error": "invalid local request"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= 16384:
                    raise ValueError("invalid request length")
                result = store.update(json.loads(self.rfile.read(n)))
                self.send(200, result)
            except RuntimeError as e:
                self.send(409, {"error": str(e)})
            except OSError:
                self.send(500, {"error": "Could not save labels to disk; retry after checking the output folder"})
            except (ValueError, TypeError, AttributeError) as e:
                self.send(400, {"error": str(e)})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--labels", type=Path, default=REPO / "dinov3_global/runs/atlas_labels.json")
    ap.add_argument("--predictions", type=Path, help="optional predictions CSV; hidden until revealed")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    store = AnnotationStore(args.images, args.labels, args.predictions)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store, args.port))
    print(f"Label {len(store.files)} images at http://127.0.0.1:{args.port}", flush=True)
    print(f"Autosaved labels: {store.path}\nCtrl+C stops the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
