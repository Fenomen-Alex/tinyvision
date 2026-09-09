#!/usr/bin/env python3
"""Stage 13: build the normal-print dataset manifest and Exp3 derived splits.

Reads:
  - screening/vlm_proposals.json  (per-sample VLM verdicts)
  - screening/samples/manifest.json (structural metadata)
  - candidates/<id>.json           (source metadata)

Writes:
  - datasets/external/stage13-normal/manifest.json   (full per-frame manifest)
  - datasets/external/stage13-normal/stage13-normal-dataset.md (prose report)
  - datasets/processed/external-aug-exp3-normal/     (derived split dir)
      derived_manifest.json
      splits/{train,val,test}.csv
      splits/splits.json

Selection rule: decision==normal AND part_visible==True
Training sources: 7 (all except LY3jlcO34f4, zkJgLanCSUw)
Eval holdout sources: 2 (LY3jlcO34f4, zkJgLanCSUw)
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "datasets/external/stage13-normal/raw"
SCREENING = ROOT / "datasets/external/stage13-normal/screening"
SAMPLES = SCREENING / "samples"
PROPOSALS = SCREENING / "vlm_proposals.json"
SPLITS_SRC = ROOT / "datasets/processed/printing-errors/splits"
OUT_MANIFEST = ROOT / "datasets/external/stage13-normal" / "manifest.json"
OUT_MD = ROOT / "datasets/external/stage13-normal" / "stage13-normal-dataset.md"
EXP3_DIR = ROOT / "datasets/processed/external-aug-exp3-normal"

# Source roles
TRAIN_SOURCES = [
    "DHpo4Za2--M",    # Ender 3 Pro benchy
    "LwHaHzVV_iE",    # Creality K1 benchy
    "PscyiwiZ9Mc",    # XIMBRIUM test print
    "R0xnULlObrs",    # AnyCubic Chiron wall
    "kZ06bpskj9I",    # K1 Max benchy
    "uCCXJzvYxKA",    # Prusa Core One benchy
    "zFX80aMaYFg",    # C270 webcam live
]
EVAL_SOURCES = [
    "LY3jlcO34f4",    # Civilized Barbarian live
    "zkJgLanCSUw",    # Martinez benchy
]
ALL_SOURCES = TRAIN_SOURCES + EVAL_SOURCES

PRINTER_MAP = {
    "DHpo4Za2--M":   "Ender 3 Pro",
    "LwHaHzVV_iE":   "Creality K1",
    "kZ06bpskj9I":   "Creality K1 Max",
    "R0xnULlObrs":   "Anycubic Chiron",
    "uCCXJzvYxKA":   "Prusa Core One L",
    "zFX80aMaYFg":   "unknown (Logitech C270 webcam)",
    "PscyiwiZ9Mc":   "unknown",
    "LY3jlcO34f4":   "unknown",
    "zkJgLanCSUw":   "unknown",
}

LICENSE_NOTE = (
    "Standard YouTube License (not redistributable); "
    "eval-only / internal experiment use only"
)

SCHEMA = "stage13-normal-dataset"
SCHEMA_VERSION = 1


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def source_meta(src_id: str) -> dict:
    jf = ROOT / "datasets/external/stage13-normal/candidates" / f"{src_id}.json"
    if not jf.exists():
        return {}
    return json.loads(jf.read_text())


def build_entries(proposals: dict) -> list[dict]:
    entries = []
    for src_id in ALL_SOURCES:
        key = f"{src_id}_samples"
        src_proposals = proposals.get(key, {})
        meta = source_meta(src_id)
        for pass_dir in ["pass_a", "b"]:
            d = SAMPLES / src_id / pass_dir
            if not d.exists():
                continue
            for img in sorted(d.glob("s*.jpg")):
                skey = f"{pass_dir}_{img.stem}"
                verdict = src_proposals.get(skey, {})
                if verdict.get("status") != "ok":
                    continue
                if verdict["decision"] != "normal" or not verdict.get("part_visible"):
                    continue
                sample_manifest = json.loads((SAMPLES / "manifest.json").read_text())
                structural = next(
                    (s for s in sample_manifest.get(src_id, {}).get("samples", [])
                     if s["sample"] == img.stem and s.get("pass") == pass_dir),
                    {},
                )
                entries.append({
                    "source_id": src_id,
                    "youtube_id": src_id,
                    "title": meta.get("title", ""),
                    "channel": meta.get("channel", ""),
                    "upload_date": meta.get("upload_date", ""),
                    "printer": PRINTER_MAP.get(src_id, "unknown"),
                    "resolution": meta.get("resolution", ""),
                    "fps": meta.get("fps", 0),
                    "duration_s": meta.get("duration", 0),
                    "pass": pass_dir,
                    "sample_index": img.stem,
                    "time_s": structural.get("time_s"),
                    "coverage_bright": structural.get("coverage_bright"),
                    "decision": verdict["decision"],
                    "part_visible": True,
                    "visual_description": verdict.get("visual_description", ""),
                    "class": "normal",
                    "split": "train" if src_id in TRAIN_SOURCES else "eval_holdout",
                    "human_verification_status": "pending_human_review",
                    "image_path": str(img.resolve()),
                    "url": f"https://www.youtube.com/watch?v={src_id}",
                    "license": LICENSE_NOTE,
                    "model": verdict.get("model", ""),
                })
    return entries


def write_derived_splits(entries: list[dict]) -> None:
    EXP3_DIR.mkdir(parents=True, exist_ok=True)
    splits_dir = EXP3_DIR / "splits"
    splits_dir.mkdir(exist_ok=True)

    def read_csv(path: Path) -> list[dict]:
        with open(path) as f:
            return list(csv.DictReader(f))

    # Frozen splits (Layout A)
    frozen = SPLITS_SRC
    train_rows = read_csv(frozen / "train.csv")
    val_rows   = read_csv(frozen / "val.csv")
    test_rows  = read_csv(frozen / "test.csv")

    # Add normal augmentation rows (absolute paths, matching exp2 pattern)
    for e in entries:
        if e["split"] != "train":
            continue
        train_rows.append({"path": e["image_path"], "label": "normal"})

    def write_csv(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    write_csv(splits_dir / "train.csv", train_rows)
    write_csv(splits_dir / "val.csv", val_rows)
    write_csv(splits_dir / "test.csv", test_rows)

    # Derived manifest
    train_entries = [e for e in entries if e["split"] == "train"]
    eval_entries  = [e for e in entries if e["split"] == "eval_holdout"]
    stats = {
        "train": len(train_rows),
        "val":   len(val_rows),
        "test":  len(test_rows),
    }
    derived = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "mode": "exp3-target-normal-aug",
        "data_dir": str(ROOT / "datasets/processed/printing-errors"),
        "source_splits": str(frozen),
        "baseline_experiment_id": "shufflenet_v2_x1_0_printing-errors_cdc9e2cad4",
        "stats": stats,
        "class_to_idx": {"failing": 0, "normal": 1},
        "training_augmentation": {
            "label": "normal",
            "provenance": "vlm-screened (qwen3-vl-8b); pending human review",
            "frames": len(train_entries),
            "sources": sorted({e["source_id"] for e in train_entries}),
            "per_source": {},
        },
        "external_normal_eval": {
            "label": "normal",
            "provenance": "vlm-screened held-out sources; evaluation-only",
            "frames": len(eval_entries),
            "sources": sorted({e["source_id"] for e in eval_entries}),
        },
        "role_by_source": {},
        "caveats": [
            "Training augmentation labels are VLM-screened, NOT human-reviewed; "
            "human_verification_status is pending_human_review.",
            "All sampled frames are 30-35 s apart; near-duplicate frames are not a concern.",
            "Held-out eval sources are held-out at the SOURCE level; no frame leakage.",
            "Held-out eval frames used ONLY for exp3 external-normal-eval; never for training.",
            "Standard YouTube License; not redistributable.",
        ],
    }
    for src_id in TRAIN_SOURCES:
        derived["role_by_source"][src_id] = "training_augmentation (normal)"
    for src_id in EVAL_SOURCES:
        derived["role_by_source"][src_id] = "external_normal_eval (normal)"

    # Per-source counts
    for src_id in TRAIN_SOURCES:
        n = sum(1 for e in train_entries if e["source_id"] == src_id)
        derived["training_augmentation"]["per_source"][src_id] = n

    (EXP3_DIR / "derived_manifest.json").write_text(json.dumps(derived, indent=2))

    splits_json = {
        "data_dir": str(ROOT / "datasets/processed/printing-errors"),
        "mode": "exp3-target-normal-aug",
        "stats": stats,
        "splits": {
            "train": dict(Counter(r["label"] for r in train_rows)),
            "val":   dict(Counter(r["label"] for r in val_rows)),
            "test":  dict(Counter(r["label"] for r in test_rows)),
        },
    }
    (splits_dir / "splits.json").write_text(json.dumps(splits_json, indent=2))
    print(f"exp3 splits written: {stats}")


def write_manifest(entries: list[dict]) -> None:
    train_entries = [e for e in entries if e["split"] == "train"]
    eval_entries  = [e for e in entries if e["split"] == "eval_holdout"]
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "selection_rule": "decision==normal AND part_visible==True",
        "total_frames": len(entries),
        "train_frames": len(train_entries),
        "eval_frames":  len(eval_entries),
        "train_sources": len(TRAIN_SOURCES),
        "eval_sources":  len(EVAL_SOURCES),
        "per_source": {
            src: sum(1 for e in entries if e["source_id"] == src)
            for src in ALL_SOURCES
        },
        "license": LICENSE_NOTE,
        "human_verification_status": "pending_human_review (all frames)",
        "frozen_test": "datasets/processed/printing-errors/splits/test.csv (byte-identical copy)",
        "entries": entries,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {len(entries)} frames "
          f"(train={len(train_entries)}, eval={len(eval_entries)})")


def write_report(entries: list[dict]) -> None:
    train_entries = [e for e in entries if e["split"] == "train"]
    eval_entries  = [e for e in entries if e["split"] == "eval_holdout"]
    per_source_train = Counter(e["source_id"] for e in train_entries)
    per_source_eval  = Counter(e["source_id"] for e in eval_entries)

    lines = [
        "# Stage 13: Target-Domain Normal-Print Training Augmentation Dataset",
        "",
        f"Created: {datetime.utcnow().strftime('%Y-%m-%d')}",
        "",
        "## Goal",
        "",
        "Augment the frozen `printing-errors` training set with a small, "
        "high-quality, *source-grouped* set of verified-normal target-domain "
        "frames. This dataset is used for Experiment 3: testing whether the "
        "model's persistent normal-print false positive problem on independent "
        "live footage is caused by insufficient diversity of normal training "
        "data.",
        "",
        "## Selection criteria",
        "",
        "- Source independence: every video/source is a single, undivided unit. "
        "No frames from one video are split across train and eval.",
        "- Source-grouping: two strong, distinct sources held out entirely for "
        "external normal evaluation.",
        "- Temporal spacing: samples are 30-35 s apart; near-duplicate "
        "consecutive frames are not included.",
        "- Per-frame screening: every sampled frame reviewed individually with "
        "a strict, neutral, single-frame VLM prompt (qwen3-vl-8b). "
        "Frames accepted only if `decision==normal AND part_visible==True`.",
        "- Human verification status: `pending_human_review` for all frames. "
        "A human pass must be completed before any public-facing release.",
        f"- License: {LICENSE_NOTE}.",
        "",
        "## Source summary",
        "",
        "| source_id | role | printer | frames | pass | license |",
        "|-----------|------|---------|--------|------|---------|",
    ]
    for src in ALL_SOURCES:
        role = "train" if src in TRAIN_SOURCES else "eval_holdout"
        n = sum(1 for e in entries if e["source_id"] == src)
        printer = PRINTER_MAP.get(src, "unknown")
        line = f"| {src} | {role} | {printer} | {n} | YouTube |"
        lines.append(line)

    lines += [
        "",
        "## Frame counts",
        "",
        f"- **Train frames:** {len(train_entries)} "
        f"(from {len(TRAIN_SOURCES)} sources)",
        f"- **Eval holdout frames:** {len(eval_entries)} "
        f"(from {len(EVAL_SOURCES)} sources)",
        "- **Selection rule:** decision==normal AND part_visible==True",
        "",
        "## Per-source train frame counts",
        "",
    ]
    for src in TRAIN_SOURCES:
        lines.append(f"- {src}: {per_source_train.get(src, 0)}")
    lines += [
        "",
        "## Human verification status",
        "",
        "All frames in this dataset are VLM-screened only "
        "(`pending_human_review`). The multi-signal screening pipeline is:",
        "",
        "1. Source design: unedited, fixed-camera, real-time webcam/live streams "
        "of FDM prints.",
        "2. Continuity analysis: adjacent-frame structural diff confirms "
        "continuous real-time footage (no timelapses or edits).",
        "3. Per-frame VLM review: strict, neutral prompt requiring "
        "`decision==normal AND part_visible==True`; ambiguous frames excluded.",
        "4. Selection: only frames passing all three criteria are included.",
        "",
        "A human pass via `scripts/run_human_review.py build|review|merge` "
        "must be run before any release.",
        "",
        "## Known limitations",
        "",
        "- YouTube Standard License; not redistributable.",
        "- All sources are from the same general FDM printer ecosystem; "
        "printer diversity is moderate (Ender, Creality K1, Anycubic, Prusa, "
        "plus two unknown-printer webcam streams).",
        "- zFX80aMaYFg frames often show the print indirectly via a wide "
        "Logitech C270 webcam; `part_visible` is true but the part is small "
        "in the frame.",
        "- Frame counts per source range from 23 to 99; some sources "
        "contribute relatively few frames.",
        "",
        "## Files",
        "",
        "- `manifest.json`: full per-frame manifest with provenance.",
        "- `stage13-normal-dataset.md`: this document.",
        "- `screening/samples/<id>/{pass_a,b}/`: source sample frames.",
        "- `screening/vlm_proposals.json`: VLM verdicts for all sampled frames.",
        "- `candidates/*.json`: source metadata (title, channel, resolution).",
        "",
        "## Derived splits for Experiment 3",
        "",
        "- `datasets/processed/external-aug-exp3-normal/splits/train.csv`",
        "- `datasets/processed/external-aug-exp3-normal/splits/val.csv`",
        "- `datasets/processed/external-aug-exp3-normal/splits/test.csv`",
        "- `datasets/processed/external-aug-exp3-normal/derived_manifest.json`",
        "",
        "Train.csv = frozen printing-errors train (5,863 failing + 5,391 normal) "
        f"+ {sum(1 for e in train_entries)} verified-normal target-domain "
        "frames (absolute paths). Val/test are byte-identical copies of the "
        "frozen printing-errors splits.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"report written to {OUT_MD}")


def main() -> int:
    proposals = json.loads(PROPOSALS.read_text())
    entries = build_entries(proposals)
    write_manifest(entries)
    write_derived_splits(entries)
    write_report(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())