"""SKZ Nozzle-Cam first probe (test split, source-grouped, image-level classes).

This module prepares a small, auditable probe from the SKZ Nozzle-Cam TEST split
that was downloaded during the read-only audit (datasets/external/skz-audit/).

Rules implemented here (see report.md for the full rationale):

* An image is *failing* if it contains at least one defect-class box.
* The three probe failure classes are derived from box presence:
  Spaghetti(3) -> spaghetti, Stringing(4) -> stringing, Unterextrusion(5) -> under_extrusion.
* An image containing several defect classes is ONE failing image; it records ALL
  failure types in metadata and is assigned the highest-priority probe class
  (priority spaghetti > under_extrusion > stringing, documented in MAPPING_RULES).
* Defect classes outside the probe (Warping(6), schlechte erste Schicht(7),
  Double Print(8)) never become a probe class; images carrying only those are excluded
  and reported as `other_defect`, never silently discarded.
* **Normal is never inferred from Object/Nozzle/Purge-Line boxes.** A *candidate*
  normal image is one with NO defect-class box; the annotation semantics are recorded
  and the candidate remains a candidate.
* Sample selection is group-preserving (see select_samples / split_by_group) and only
  selects images whose grouping key is confidence-... inferred from the filename
  `Profile_<params>__<variant>` token. Ungrouped images are excluded from this probe
  and reported, not silently dropped and not counted as grouped.

Third-party licensing note: the source dataset is provided under CC BY-SA 4.0 on the
Hugging Face card and CC BY 4.0 on the Zenodo umbrella record. This module RECORDS both
observations and does not choose an interpretation.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import random
import re
import shutil
import struct
import subprocess
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

REPO = Path(__file__).resolve().parents[2]
AUDIT = REPO / "datasets" / "external" / "skz-audit"
PROBE = REPO / "datasets" / "external" / "skz-probe"

AUDIT_TEST_IMAGES = AUDIT / "test" / "images"
AUDIT_TEST_LABELS = AUDIT / "test" / "labels"

HF_DATASET_ID = "DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Nozzle_Cam"
HF_URL = "https://huggingface.co/datasets/DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Nozzle_Cam"
HF_DOI = "doi:10.57967/hf/7891"
ZENODO_RECORD = "https://zenodo.org/records/18628777"

SCHEMA = "skz-probe"
SCHEMA_VERSION = 1

CLS = {
    0: "Nozzle",
    1: "Object",
    2: "Purge Line",
    3: "Spaghetti",
    4: "Stringing",
    5: "Unterextrusion",
    6: "Warping",
    7: "schlechte erste Schicht",  # bad first layer
    8: "Double Print",
}
DEFECT = frozenset({3, 4, 5, 6, 7, 8})
PROBE_CLASS_BY_ID = {3: "spaghetti", 4: "stringing", 5: "under_extrusion"}
CLASS_PRIORITY = {"spaghetti": 2, "under_extrusion": 1, "stringing": 0}
PROBE_CLASSES = ("normal", "stringing", "spaghetti", "under_extrusion")
DEFAULT_TARGETS = {"normal": 50, "stringing": 50, "spaghetti": 50, "under_extrusion": 50}
SEED = 42
GROUP_RE = re.compile(r"^(Profile_.+?)__[0-9a-f]{8}-frame_(\d+)\.png$")

MAPPING_RULES = {
    "schema": SCHEMA,
    "version": SCHEMA_VERSION,
    "summary": (
        "Image-level classes derived from test-split YOLO box presence. "
        "Any defect-class box makes the image failing. Object/Nozzle/Purge-Line boxes "
        "never imply normal. Normal is a CANDIDATE label understood as "
        "'no defect-class box in this annotation'."
    ),
    "defect_classes": {
        str(k): v for k, v in CLS.items() if k in DEFECT
    },
    "failure_derivation": {
        "condition": "one or more defect-class boxes present",
        "image_label": "failing",
        "failure_types_recorded": "all defect-class names present, kept in metadata",
        "probe_class_for_image": "highest-priority probe class among those present",
        "priority": list(CLASS_PRIORITY),
        "priority_rationale": (
            "Higher-visibility/more-severe modes take the class label when multiple are "
            "present so a single-class audit is feasible; every mode is preserved in "
            "all_failure_types. Choice is a probe convenience, not a claim about physics."
        ),
        "normal_candidate": (
            "image with NO defect-class box. Presence of Nozzle/Object/Purge-Line boxes "
            "is not a normal signal. The candidate status is provisional until the "
            "annotation semantics are visually verified."
        ),
    },
}

# schema-name -> required key list, used by build_manifest and asserted in tests
_MANIFEST_TOP_KEYS = (
    "schema",
    "schema_version",
    "generated_at",
    "source_dataset",
    "source_loc",
    "mapping_rules",
    "selection",
    "excluded",
    "summary",
    "entries",
)
_MANIFEST_ENTRY_KEYS = (
    "probe_id",
    "derived_class",
    "source_image_id",
    "source_image_path",
    "source_label_path",
    "defect_class_ids",
    "defect_class_names",
    "all_failure_types",
    "grouping_key",
    "grouping_verified",
    "image_dimensions",
    "annotation_count",
    "multi_defect",
    "selection_rationale",
    "generation_timestamp",
    "image_sha256",
)

SOURCE_DATASET = {
    "name": "SKZ Nozzle-Cam (Errors_Additive_Manufacturing_Nozzle_Cam)",
    "hf_id": HF_DATASET_ID,
    "url": HF_URL,
    "doi": HF_DOI,
    "split": "test",
    "task": "object detection (YOLO format)",
    "license_hf_card_observed": "CC BY-SA 4.0",
    "license_zenodo_record_observed": "CC BY 4.0",
    "license_note": (
        "Discrepancy recorded without interpretation: the HF dataset card states "
        "CC BY-SA 4.0 while the umbrella Zenodo record 18628777 states CC BY 4.0. "
        "Resolve before any downstream release."
    ),
}


def parse_annotation_line(line: str) -> tuple[float, float, float, float, float] | None:
    """Parse one YOLO row into (class_id, x, y, w, h), or None for blank/invalid."""
    parts = line.split()
    if len(parts) != 5:
        return None
    try:
        vals = tuple(float(p) for p in parts)  # type: ignore[misc]
    except ValueError:
        return None
    c, x, y, w, h = vals
    if not (0 <= c <= 8 and 0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
        return None
    return c, x, y, w, h  # noqa: RET504


def read_label_classes(label_file: Path) -> tuple[frozenset[int], int]:
    """Return (classes present, annotation count) for a YOLO label file."""
    classes: set[int] = set()
    count = 0
    for line in label_file.read_text().splitlines():
        parsed = parse_annotation_line(line)
        if parsed is None:
            raise ValueError(f"invalid or unparseable annotation in {label_file}: {line!r}")
        classes.add(int(parsed[0]))
        count += 1
    return frozenset(classes), count


def defect_ids(classes: Iterable[int]) -> frozenset[int]:
    """Subset of SKZ classes that are defect classes."""
    return frozenset(c for c in classes if c in DEFECT)


def derive_class(classes: Iterable[int]) -> str | None:
    """Map the full per-image class set to a probe class (or None to exclude)."""
    defs = defect_ids(classes)
    if not defs:
        return "normal"  # candidate only (recorded separately)
    probe_found = [PROBE_CLASS_BY_ID[d] for d in defs if d in PROBE_CLASS_BY_ID]
    if not probe_found:
        return None  # only non-probe defects (warping/bad first layer/double print)
    return max(probe_found, key=lambda cls: CLASS_PRIORITY[cls])


def all_failure_types(classes: Iterable[int]) -> list[str]:
    """Original SKZ defect-class names present, ordered by class id."""
    return [CLS[c] for c in sorted(defect_ids(classes))]


def multi_defect(classes: Iterable[int]) -> bool:
    return len(defect_ids(classes)) > 1


def group_key_of(filename: str) -> str | None:
    """Group key from a test-split filename, or None when not confidence-groupable."""
    m = GROUP_RE.match(filename)
    return m.group(1) if m else None


def grouping_verified_of(filename: str) -> bool:
    """Grouping is always *inferred* from the Profile config token, never verified."""
    return False


def frame_number_of(filename: str) -> int:
    m = GROUP_RE.match(filename)
    if not m:
        raise ValueError(f"not a groupable filename: {filename}")
    return int(m.group(2))


def png_dimensions(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        head = fh.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    w, h = struct.unpack(">II", head[16:24])
    return w, h


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_all_test_records(labels_dir: Path) -> list[dict]:
    records = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        image = label_path.stem + ".png"
        classes, count = read_label_classes(label_path)
        records.append(
            {
                "image": image,
                "stem": label_path.stem,
                "label_path": str(label_path.relative_to(REPO)),
                "image_path": str((AUDIT_TEST_IMAGES / image).relative_to(REPO)),
                "classes": classes,
                "derived": derive_class(classes),
                "group": group_key_of(image),
                "frame": int(image.rsplit("-frame_", 1)[1][:-4]),
                "annotation_count": count,
            }
        )
    return records


def select_eligible(records: list[dict]) -> dict[str, list[dict]]:
    """Group by derived class, dropping ungrouped and excluded (non-probe) images.

    Ungrouped images are intentionally NOT selected and must be accounted for by the
    caller (they are reported as `ungrouped`, never silently merged into any class).
    """
    per_class: dict[str, list[dict]] = {cls: [] for cls in PROBE_CLASSES}
    for rec in records:
        if rec["derived"] is None or rec["group"] is None:
            continue
        per_class[rec["derived"]].append(rec)
    return per_class


def group_size_sorted(candidates: list[dict]) -> list[str]:
    groups: dict[str, int] = {}
    for rec in candidates:
        groups[rec["group"]] = groups.get(rec["group"], 0) + 1
    return sorted(groups, key=lambda g: (-groups[g], g))


def _median_record(records: list[dict]) -> dict:
    """Return the record with the median frame number (ties -> smaller frame)."""
    ordered = sorted(records, key=lambda r: r["frame"])
    return ordered[len(ordered) // 2]


def select_samples(
    per_class: dict[str, list[dict]], targets: dict[str, int], seed: int = SEED
) -> dict[str, list[dict]]:
    """Group-preserving sample selection.

    Pass 1 takes one (median-frame) representative per group; Pass 2 fills remaining
    quota from groups with surplus, largest surplus first (records by frame number).
    Returns a dict of class -> selected records, each augmented with `selection_rationale`
    and a stable `probe_seq`.
    """
    rng = random.Random(seed)
    for cls in PROBE_CLASSES:
        rng.shuffle(per_class[cls])
    out: dict[str, list[dict]] = {}
    for cls in PROBE_CLASSES:
        candidates = per_class[cls]
        target = targets[cls]
        by_group: dict[str, list[dict]] = {}
        for rec in candidates:
            by_group.setdefault(rec["group"], []).append(rec)
        selected: list[dict] = []
        # Pass 1: one median-frame representative per group (groups with more surplus first).
        for g in group_size_sorted(candidates):
            if len(selected) >= target:
                break
            rep = _median_record(by_group[g])
            rep["selection_rationale"] = f"pass1 group representative of {g}"
            selected.append(rep)
        # Pass 2: fill from surplus, largest surplus first.
        if len(selected) < target:
            for g in group_size_sorted(candidates):
                if len(selected) >= target:
                    break
                avail = [r for r in by_group[g] if not any(
                    s["image"] == r["image"] for s in selected)]
                for rec in sorted(avail, key=lambda r: r["frame"]):
                    if len(selected) >= target:
                        break
                    rec["selection_rationale"] = f"pass2 surplus fill (group {g})"
                    selected.append(rec)
        out[cls] = selected
    # Assign per-class sequence ids deterministically.
    for cls in PROBE_CLASSES:
        for i, rec in enumerate(out[cls]):
            rec["probe_seq"] = i
    return out


def split_by_group(group_counts: dict[str, int], train_fraction: float = 0.8, seed: int = SEED) -> dict[str, str]:
    """Source-grouped train/eval split that never splits a group.

    Pure helper for future eval use; NOT invoked by probe construction. group_counts maps
    group key -> number of images. Every image of a group lands in the same split.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    rng = random.Random(seed)
    groups = list(group_counts)
    rng.shuffle(groups)
    total = sum(group_counts.values())
    budget = train_fraction * total
    visible = 0
    split: dict[str, str] = {}
    for g in groups:
        if visible < budget:
            split[g] = "train"
            visible += group_counts[g]
        else:
            split[g] = "eval"
    return split


def _image_url(name: str) -> str:
    return urllib.parse.quote(
        f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main/images/test/{name}",
        safe="/:",
    )


def download_missing_images(selected: list[dict], out_dir: Path, max_workers: int = 12) -> int:
    """Fetch test-split PNGs not already present under out_dir. Skip existing."""
    wanted = [rec["image"] for rec in selected]
    existing = {p.name for p in out_dir.glob("*.png")} if out_dir.exists() else set()
    todo = [w for w in wanted if w not in existing]

    def fetch(name: str) -> tuple[str, bool]:
        url = _image_url(name)
        dest = out_dir / name
        if dest.exists():
            return name, False
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "90", "-f", url, "-o", str(dest)],
            capture_output=True,
        )
        if result.returncode != 0:
            dest.unlink(missing_ok=True)
            return name, False
        return name, True

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(fetch, todo))
    downloaded = sum(1 for _, ok in results if ok)
    failed = [n for n, ok in results if not ok]
    if failed:
        raise RuntimeError(
            f"failed to download {len(failed)} images: {failed[:5]!r} "
            "(remove partial files and retry)"
        )
    return downloaded


def entry_metadata(rec: dict, cls: str, build_time: str) -> dict:
    """Pure provenance metadata for a selected record (no file access)."""
    return {
        "probe_id": f"SKZNC-{cls}-{rec['probe_seq']:03d}",
        "derived_class": cls,
        "source_image_id": rec["image"],
        "source_image_path": str((AUDIT_TEST_IMAGES / rec["image"]).relative_to(REPO)),
        "source_label_path": str((AUDIT_TEST_LABELS / (rec["stem"] + ".txt")).relative_to(REPO)),
        "defect_class_ids": sorted(defect_ids(rec["classes"])),
        "defect_class_names": all_failure_types(rec["classes"]),
        "all_failure_types": all_failure_types(rec["classes"]),
        "grouping_key": rec["group"],
        "grouping_verified": grouping_verified_of(rec["image"]),
        "annotation_count": rec["annotation_count"],
        "multi_defect": multi_defect(rec["classes"]),
        "selection_rationale": rec["selection_rationale"],
        "generation_timestamp": build_time,
    }


def _entry(rec: dict, cls: str, build_time: str, images_dir: Path, labels_dir: Path) -> dict:
    src_img = AUDIT_TEST_IMAGES / rec["image"]
    w, h = png_dimensions(src_img)
    entry = entry_metadata(rec, cls, build_time)
    entry["image_dimensions"] = {"width": w, "height": h}
    entry["image_sha256"] = sha256_of(src_img)
    entry["probe_image_path"] = str((images_dir / cls / rec["image"]).relative_to(REPO))
    entry["probe_label_path"] = str((labels_dir / cls / (rec["stem"] + ".txt")).relative_to(REPO))
    return entry


def copy_selected(selected: dict[str, list[dict]], images_dir: Path, labels_dir: Path):
    for cls, recs in selected.items():
        (images_dir / cls).mkdir(parents=True, exist_ok=True)
        (labels_dir / cls).mkdir(parents=True, exist_ok=True)
        for rec in recs:
            shutil.copyfile(
                AUDIT_TEST_IMAGES / rec["image"], images_dir / cls / rec["image"]
            )
            shutil.copyfile(
                AUDIT_TEST_LABELS / (rec["stem"] + ".txt"),
                labels_dir / cls / (rec["stem"] + ".txt"),
            )


def manifest_document(
    entries: list[dict],
    counts: dict,
    excluded: dict,
    seed: int,
    targets: dict[str, int],
    build_time: str,
) -> dict:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "generated_at": build_time,
        "source_dataset": SOURCE_DATASET,
        "source_loc": {
            "local_images_dir": str(AUDIT_TEST_IMAGES.relative_to(REPO)),
            "local_labels_dir": str(AUDIT_TEST_LABELS.relative_to(REPO)),
            "remote_pattern": f"HF {HF_DATASET_ID} images/test/<name>",
        },
        "mapping_rules": MAPPING_RULES,
        "selection": {
            "seed": seed,
            "targets": targets,
            "algorithm": (
                "pass1: one median-frame representative per Profile-config group; "
                "pass2: surplus fill largest-surplus-first; deterministic"
            ),
        },
        "excluded": excluded,
        "summary": counts,
        "entries": entries,
    }


def build_manifest(
    selected: dict[str, list[dict]],
    counts: dict,
    excluded: dict,
    seed: int,
    targets: dict[str, int],
    build_time: str,
    images_dir: Path,
    labels_dir: Path,
) -> dict:
    entries = []
    for cls in PROBE_CLASSES:
        for rec in selected[cls]:
            entries.append(_entry(rec, cls, build_time, images_dir, labels_dir))
    return manifest_document(entries, counts, excluded, seed, targets, build_time)


def run_plan(records: list[dict], targets: dict[str, int], seed: int = SEED) -> tuple[dict[str, list[dict]], dict, dict]:
    """Pure selection + statistics (no IO beyond records)."""
    per_class = select_eligible(records)
    selected = select_samples(per_class, targets, seed=seed)

    excluded: dict = {
        "ungrouped": {},
        "other_defect": {},
        "union_overlaps": {"ungrouped_and_other_defect": None},
    }
    ungrouped: dict[str, int] = {}
    other_defect: dict[str, int] = {}
    both = 0
    for rec in records:
        is_ungrouped = rec["group"] is None
        is_other = rec["derived"] is None
        if is_ungrouped:
            ungrouped[rec["derived"] or "_no_class"] = ungrouped.get(rec["derived"] or "_no_class", 0) + 1
        if is_other:
            other_defect[rec["group"] or "_ungrouped"] = other_defect.get(rec["group"] or "_ungrouped", 0) + 1
        if is_ungrouped and is_other:
            both += 1
    excluded["ungrouped"] = ungrouped
    excluded["other_defect"] = other_defect
    excluded["union_overlaps"]["ungrouped_and_other_defect"] = both

    counts: dict = {}
    for cls in PROBE_CLASSES:
        recs = selected[cls]
        groups = sorted({r["group"] for r in recs})
        counts[cls] = {
            "selected": len(recs),
            "groups": len(groups),
            "multi_defect": sum(1 for r in recs if multi_defect(r["classes"])),
        }
    counts["total_selected"] = sum(v["selected"] for v in counts.values())
    return selected, counts, excluded


def build_skz_probe(seed: int = SEED, targets: dict[str, int] | None = None) -> dict:
    """Build the probe end-to-end (download selected test images, copy, manifest, report)."""
    targets = dict(DEFAULT_TARGETS if targets is None else targets)
    build_time = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    records = _read_all_test_records(AUDIT_TEST_LABELS)
    selected, counts, excluded = run_plan(records, targets, seed=seed)

    AUDIT_TEST_IMAGES.mkdir(parents=True, exist_ok=True)
    flat = [rec for recs in selected.values() for rec in recs]
    downloaded = download_missing_images(flat, AUDIT_TEST_IMAGES)

    images_dir = PROBE / "images"
    labels_dir = PROBE / "annotations"
    (PROBE / "images").mkdir(parents=True, exist_ok=True)
    (PROBE / "annotations").mkdir(parents=True, exist_ok=True)
    copy_selected(selected, images_dir, labels_dir)

    manifest = build_manifest(
        selected, counts, excluded, seed, targets, build_time, images_dir, labels_dir
    )
    (PROBE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    summary = {
        "seed": seed,
        "generated_at": build_time,
        "downloaded_test_images": downloaded,
        "counts": counts,
        "excluded": excluded,
        "total_test_images": len(records),
    }
    (PROBE / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(manifest, summary, (PROBE / "report.md"))
    return summary


def write_report(manifest: dict, summary: dict, out: Path) -> None:
    lines = [
        "# SKZ Nozzle-Cam Probe — build report",
        "",
        f"Generated {summary['generated_at']} (UTC). Schema `{SCHEMA}` v{SCHEMA_VERSION}.",
        "",
        "## Exact local source used",
        "",
        f"- Labels (all 536 test images): `{manifest['source_loc']['local_labels_dir']}`",
        f"- Images: `{manifest['source_loc']['local_images_dir']}` (selected test images"
        " fetched from HF into this dir; missing ones downloaded during build)",
        f"- Remote: HF `{HF_DATASET_ID}`, split `test`, pattern "
        f"`{manifest['source_loc']['remote_pattern']}`",
        "",
        "## Class-mapping rules",
        "",
        "1. An image with **at least one defect-class box** is failing.",
        "2. Failure type per box: Spaghetti(3)->spaghetti, Stringing(4)->stringing, "
        "Unterextrusion(5)->under_extrusion.",
        "3. All failure types present are preserved in `all_failure_types` / "
        "`defect_class_names`.",
        "4. If multiple probe defect classes appear, the image is ONE failing image; its "
        "derived class uses documented priority `spaghetti > under_extrusion > stringing` "
        "and `multi_defect=true`.",
        "5. Non-probe defect classes (Warping(6), schlechte erste Schicht(7), "
        "DoublePrint(8)) never become a probe class; images with ONLY those are excluded "
        "as `other_defect`.",
        "6. **Normal is never inferred from Nozzle/Object/Purge-Line boxes.** A candidate "
        "normal image has no defect-class box; it is a candidate until annotation "
        "semantics are visually verified.",
        "",
        "## Sample counts by class",
        "",
        "| class | selected | independent groups | multi-defect included |",
        "|---|---|---|---|",
    ]
    for cls in PROBE_CLASSES:
        c = summary["counts"][cls]
        lines.append(f"| {cls} | {c['selected']} | {c['groups']} | {c['multi_defect']} |")
    ex = summary["excluded"]
    lines += [
        "",
        "## Excluded / ungrouped counts",
        "",
        f"- Ungrouped test images (no `Profile_` config token, group=None): "
        f"{sum(ex['ungrouped'].values())}",
        f"- `other_defect` images (only non-probe defect classes): "
        f"{sum(ex['other_defect'].values())}",
        f"- Ungrouped AND other-defect (overlap): {ex['union_overlaps']['ungrouped_and_other_defect']}",
        f"- Total test images considered: {summary['total_test_images']}",
        "",
        "## Grouping / leakage limitations",
        "",
        "- Group key = the `Profile_<params>__<variant>` token (print-configurable run). "
        "It is **inferred**, not verified: identical config tokens could denote the same "
        "print (many frames) or repeat runs with identical settings.",
        "- `grouping_verified` is `false` for every entry.",
        "- 148/536 test images (28%) carry no Profile token and are excluded; the probe "
        "contains **no ungrouped samples**.",
        "- Cross-split check performed in the audit: zero Profile-config overlap between "
        "test and val, and zero between test and the first 1000 enumerated train images. "
        "The remaining train images could not be enumerated via the HF API (tree "
        "pagination), so full train-side leakage is NOT claimed.",
        "- A document-group can legitimately span classes (e.g. a print whose later frames "
        "fail). Splits, when built, MUST be source-grouped via `split_by_group`.",
        "",
        "## License",
        "",
        "- HF dataset card observed license: **CC BY-SA 4.0**. "
        "- Zenodo umbrella record 18628777 observed license: **CC BY 4.0**. "
        "- Recorded without choosing an interpretation; resolve before any downstream "
        "release or integration.",
        "",
    ]
    out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("use scripts/build_skz_probe.py")