"""Paired, membership-checked comparisons and the frozen promotion decision."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from dinov3_global.evaluation import sha256_file, write_json
from dinov3_global.metrics import report, to_jsonable
from dinov3_global.study import promotion_checks, paired_session_bootstrap, recall_score
from scripts.compare_axial_pilot import read_predictions, vzc_test
from scripts.evaluate_ood import read_labels, evaluate_labels

CLASSES = ("RR", "RG", "NoR")


def aligned(baseline, candidate, basenames=False):
    bp, by, bl = baseline
    cp, cy, cl = candidate
    if basenames:
        bp, cp = [Path(p).stem for p in bp], [Path(p).stem for p in cp]
    if len(set(bp)) != len(bp) or len(set(cp)) != len(cp) or set(bp) != set(cp):
        raise ValueError("Comparison membership differs or is aliased")
    lookup = {p: i for i, p in enumerate(cp)}
    order = [lookup[p] for p in bp]
    if not np.array_equal(by, cy[order]):
        raise ValueError("Comparison labels differ")
    return bp, by, bl, cl[order]


def mean_predictions(members):
    first = members[0]
    logits = [first[2]]
    for member in members[1:]:
        _, _, _, current = aligned(first, member)
        logits.append(current)
    return first[0], first[1], np.mean(logits, axis=0)


def off_unknown_slices(y, logits, pseudo):
    return {
        "excluding_relevant_off_unknown_only": report(y[~pseudo], logits[~pseudo]),
        "relevant_off_unknown_only": report(y[pseudo], logits[pseudo]) if pseudo.any() else None,
        "clean_NoR_recall": float((logits[(y == 2) & ~pseudo].argmax(1) == 2).mean())
        if ((y == 2) & ~pseudo).any()
        else None,
    }


def compare(study, dtld, repeats=2000):
    baseline = REPO / "runs/v5"
    frozen = json.loads((study / "frozen_candidate.json").read_text())
    winner = frozen["winner"]
    for row in frozen["checkpoints"]:
        if sha256_file(row["path"]) != row["sha256"]:
            raise ValueError("Frozen head changed")
    for directory, key in ((dtld, "status"), (study / "evaluations/vzc", "state")):
        if json.loads((directory / "status.json").read_text())[key] != "complete":
            raise ValueError("Evaluation is incomplete")
    dtld_plan = json.loads((dtld / "plan.json").read_text())
    if [r["checkpoint_sha256"] for r in dtld_plan["checkpoints"]] != [
        r["sha256"] for r in frozen["checkpoints"]
    ]:
        raise ValueError("DTLD evaluation used different heads")
    membership = json.loads((dtld / "membership.json").read_text())
    entries_by_name = {Path(r["native_path"]).stem: r for r in membership["test"]}
    # Relevant-lamp sizes in native coordinates, common to both input resolutions.
    from dinov3_global.dtld import load_split, parse_frame

    annotations = load_split(str(REPO / "datasets/DTLD/v2.0"), "test")
    relevant_sizes = {}
    for entry in annotations:
        frame = parse_frame(entry)
        if frame is None:
            continue
        relevant = frame.relevance == 1
        relevant_sizes[Path(entry["image_path"]).stem] = (
            float((frame.boxes[relevant, 3] - frame.boxes[relevant, 1]).max()) if relevant.any() else 0.0
        )
    domains, checks, metric_rows, class_rows, confusion_rows = {}, {}, [], [], []

    def add(domain, scope, old, new, basenames=False):
        paths, y, bl, cl = aligned(old, new, basenames)
        kwargs = {}
        if domain == "DTLD":
            rows = [entries_by_name[p] for p in paths]
            kwargs["cities"] = [r["city"] for r in rows]
        reps = dict(v5=report(y, bl, **kwargs), candidate=report(y, cl, **kwargs))
        if domain == "DTLD":
            pseudo = np.array([r["pseudo_nor"] for r in rows], bool)
            sizes = np.array([relevant_sizes[p] for p in paths])
            for name, logits in (("v5", bl), ("candidate", cl)):
                reps[name]["off_unknown_sensitivity"] = off_unknown_slices(y, logits, pseudo)
                reps[name]["relevant_lamp_size"] = dict(
                    coordinate_system="native pixels",
                    slices={
                        bucket: report(y[mask], logits[mask])
                        for bucket, mask in {
                            "no_relevant_lamp": sizes == 0,
                            "small_below_24px": (sizes > 0) & (sizes < 24),
                            "medium_24_to_72px": (sizes >= 24) & (sizes < 72),
                            "large_at_least_72px": sizes >= 72,
                        }.items()
                        if mask.any()
                    },
                )
                camera = [p.rsplit("_", 1)[-1] for p in paths]
                reps[name]["camera"] = {
                    c: report(y[np.array(camera) == c], logits[np.array(camera) == c])
                    for c in sorted(set(camera))
                }
        result = dict(reports=reps, acceptance=promotion_checks(reps["v5"], reps["candidate"], domain))
        if domain == "DTLD" and scope == "ensemble4":
            result["uncertainty"] = paired_session_bootstrap(
                y, bl, cl, [r["session"] for r in rows], repeats=repeats
            )
        domains.setdefault(domain, {})[scope] = result
        for name, rep in reps.items():
            metric_rows.append(
                dict(
                    domain=domain, scope=scope, model=name, **rep["metrics"], signal_recall=recall_score(rep)
                )
            )
            class_rows.extend(
                dict(domain=domain, scope=scope, model=name, class_name=c, **rep["per_class"][c])
                for c in CLASSES
            )
            confusion_rows.extend(
                dict(
                    domain=domain,
                    scope=scope,
                    model=name,
                    true=tc,
                    predicted=pc,
                    count=rep["confusion_matrix"][i][j],
                )
                for i, tc in enumerate(CLASSES)
                for j, pc in enumerate(CLASSES)
            )
        if scope == "ensemble4":
            checks[domain] = result["acceptance"]

    for f in range(4):
        add(
            "DTLD",
            f"fold{f}",
            read_predictions(baseline / f"evaluations/dtld_test/member{f}/predictions.npz"),
            read_predictions(dtld / f"member{f}/predictions.npz"),
            True,
        )
    add(
        "DTLD",
        "ensemble4",
        read_predictions(baseline / "evaluations/dtld_test/predictions.npz"),
        read_predictions(dtld / "predictions.npz"),
        True,
    )

    atlas_old = [
        read_predictions(baseline / f"evaluations/atlas/v5_fold{f}/predictions.csv") for f in range(4)
    ]
    atlas_new = [
        read_predictions(study / f"evaluations/atlas/candidate_fold{f}/predictions.csv") for f in range(4)
    ]
    for f in range(4):
        add("ATLAS", f"fold{f}", atlas_old[f], atlas_new[f])
    old_mean, new_mean = mean_predictions(atlas_old), mean_predictions(atlas_new)
    add("ATLAS", "ensemble4", old_mean, new_mean)
    atlas_entries, excluded = read_labels(
        REPO / "metadata/atlas_labels.json", REPO / "datasets/atlas_relevance_expanded", True
    )
    for name, predictions in (("v5", old_mean), ("candidate", new_mean)):
        lookup = {p: i for i, p in enumerate(predictions[0])}
        logits = predictions[2][[lookup[p] for p, _, _ in atlas_entries]]
        domains["ATLAS"]["ensemble4"]["reports"][name]["camera_diagnostics"] = evaluate_labels(
            atlas_entries, logits, excluded
        )["per_camera"]

    vzc_old = [vzc_test(baseline / "evaluations/vzc", f"v5_fold{f}") for f in range(4)]
    vzc_new = [vzc_test(study / "evaluations/vzc", f"candidate_fold{f}") for f in range(4)]
    for f in range(4):
        add("VZC", f"fold{f}", vzc_old[f], vzc_new[f])
    add("VZC", "ensemble4", mean_predictions(vzc_old), mean_predictions(vzc_new))
    # Preserve published off/unknown and size diagnostics for VZC.
    ensemble_paths = list((study / "evaluations/vzc").glob("*_ensemble4/report.json"))
    if len(ensemble_paths) != 1:
        raise ValueError("Expected one equivalent four-head VZC ensemble")
    domains["VZC"]["ensemble4"]["additional_candidate_diagnostics"] = json.loads(
        ensemble_paths[0].read_text()
    )["test"]

    # Exactly one held-out head per training image; never average predictions from trained-on folds.
    oof_old, oof_new, seen, fold_deltas = [], [], set(), []
    manifest = json.loads((study / winner / "fold_manifest.json").read_text())
    for f in range(4):
        old = read_predictions(baseline / f"city_cv/fold{f}/val_predictions.npz")
        new = read_predictions(study / winner / f"fold{f}/val_predictions.npz")
        paths, y, bl, cl = aligned(old, new, True)
        if seen.intersection(paths) or len(paths) != manifest["folds"][f]["n_val"]:
            raise ValueError("OOF membership overlap or incomplete fold")
        seen.update(paths)
        oof_old.append((y, bl))
        oof_new.append((y, cl))
        fold_deltas.append(promotion_checks(report(y, bl), report(y, cl), "DTLD")["delta"])
    if len(seen) != manifest["n_items"]:
        raise ValueError("OOF predictions omit training frames")
    oof = {
        name: report(np.concatenate([p[0] for p in members]), np.concatenate([p[1] for p in members]))
        for name, members in (("v5", oof_old), ("candidate", oof_new))
    }
    passed = all(c["passed"] for c in checks.values())
    failures = [
        f"{domain}: {key}" for domain, c in checks.items() for key, ok in c["checks"].items() if not ok
    ]
    decision = dict(
        promote=passed,
        retain=None if passed else "v5",
        winner=winner,
        checks=checks,
        unmet_criteria=failures,
        checkpoint_manifest=str(study / "frozen_candidate.json"),
    )
    result = dict(
        status="complete",
        frozen=frozen,
        domains=domains,
        validation_oof=oof,
        fold_deltas=fold_deltas,
        uncertainty_scope="Session bootstrap measures sampling uncertainty; folds change both seed and training cities, so their spread cannot isolate training-seed variability. No independent seed replication was added to the seven-run budget.",
        external_status="ATLAS and VZC are exploratory; they have influenced architecture development",
        decision=decision,
    )
    write_json(study / "comparison.json", to_jsonable(result))
    write_json(study / "decision.json", to_jsonable(decision))
    # Different metrics dictionaries are unioned to preserve missing-class diagnostic fields.
    for filename, rows in (
        ("metrics.csv", metric_rows),
        ("per_class.csv", class_rows),
        ("confusion_matrices.csv", confusion_rows),
    ):
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (study / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Generalization study result",
        "",
        f"Candidate: {winner}. Decision: {'eligible for promotion' if passed else 'retain v5'}.",
        "",
        "| Domain | Δ mAP (pp) | Δ RR/RG recall (pp) | Passed |",
        "|---|---:|---:|---|",
    ]
    lines += [
        f"| {domain} | {c['delta']['mAP'] * 100:+.2f} | {c['delta']['signal_recall'] * 100:+.2f} | {c['passed']} |"
        for domain, c in checks.items()
    ]
    lines += [
        "",
        "Unmet criteria: " + (", ".join(failures) if failures else "none"),
        "",
        result["uncertainty_scope"],
        "",
        result["external_status"],
        "",
        "See comparison.json for precision/recall, confusion matrices, raw calibration, city/camera slices, relevant-lamp sizes and off/unknown sensitivity.",
    ]
    (study / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--dtld", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    print(json.dumps(compare(args.study, args.dtld, args.bootstrap_repeats)["decision"], indent=2))
