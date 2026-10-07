"""Preserve completed epoch records independently of the live history viewer."""

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1] / "runs/v6_axial_pilot"


def main():
    archive = ROOT / "history_recovery/epochs"
    archive.mkdir(parents=True, exist_ok=True)
    while True:
        history = ROOT / "city_cv/fold0/history.jsonl"
        if history.exists():
            for line in history.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                path = archive / f"epoch{row['epoch']:02d}.json"
                if not path.exists():
                    temp = path.with_suffix(".tmp")
                    temp.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
                    temp.replace(path)
        status = json.loads((ROOT / "status.json").read_text(encoding="utf-8"))
        if "training" in status["completed_stages"]:
            return
        if status["status"] == "failed":
            return
        time.sleep(15)


if __name__ == "__main__":
    main()
