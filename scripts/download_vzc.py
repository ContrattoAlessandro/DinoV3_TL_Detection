"""Download and verify the pinned VZC-TLD snapshot with existing Hugging Face access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
REPO_ID = "vzc-research-chapter/vzc-traffic-light-dataset"
REVISION = "083bc5d626a2fe660171ba8a84c848222a661f02"


def checked_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Invalid snapshot path: {name}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "datasets/VZC_TLD")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true", help="verify an existing manifest offline")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "download_manifest.json"
    if args.verify_only:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["repo_id"] != REPO_ID or manifest["revision"] != REVISION:
            raise ValueError("The manifest does not describe the pinned VZC snapshot")
        files = manifest["files"]
    else:
        info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise ValueError("Unexpected dataset revision")
        files = [
            {"name": file.rfilename, "size": file.size, "lfs_sha256": file.lfs.sha256 if file.lfs else None}
            for file in info.siblings
        ]
        for file in files:
            checked_path(root, file["name"])
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    hf_hub_download,
                    REPO_ID,
                    file["name"],
                    repo_type="dataset",
                    revision=REVISION,
                    local_dir=root,
                )
                for file in files
            ]
            for future in tqdm(as_completed(futures), total=len(files), desc="Download"):
                future.result()
    for file in tqdm(files, desc="Verify"):
        path = checked_path(root, file["name"])
        if path.stat().st_size != file["size"]:
            raise ValueError(f"File size mismatch: {file['name']}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        expected = file.get("sha256") or file.get("lfs_sha256")
        if expected and digest != expected:
            raise ValueError(f"Checksum mismatch: {file['name']}")
        file["sha256"] = digest
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                file["width"], file["height"] = image.size
                image.verify()
    if args.verify_only:
        print(f"Verified {len(files)} existing files: {manifest_path}")
        return
    manifest = dict(
        repo_id=REPO_ID,
        revision=REVISION,
        files=files,
        image_root=str(root),
        image_integrity_verified=True,
        verified_at=datetime.now(timezone.utc).isoformat(),
    )
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(manifest_path)
    print(f"Verified {len(files)} files: {manifest_path}")


if __name__ == "__main__":
    main()
