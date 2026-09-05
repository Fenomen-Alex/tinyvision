"""Stage 12C: build the external pilot dataset from Stage 12B provisional labels.

Research build only.

- No training.
- No modification of the existing Kaggle dataset (`datasets/processed/printing-errors/`).
- No modification of the frozen official test set.
- No additional video downloads.
- Every assignment is deterministic and recorded.

The pilot labels are PROVISIONAL and were not human visually verified. Nothing here
claims ground truth. Frame labels follow an explicit, documented mapping from the
Stage 12B provisional label vocabulary to the allowed dataset labels
(`normal` / `failing` / `uncertain`). Uncertain frames are preserved in metadata
but excluded from the training-ready folders.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Canonical paths
REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "datasets" / "external" / "video-frames"
KAGGLE = REPO / "datasets" / "processed" / "printing-errors"

EXTRACTION_FPS = 1.0
DUP_HAMMING_THRESHOLD = 4
VAL_FRACTION = 0.2

# source_id -> raw video filename (from Stage 12B acquisition)
SOURCE_VIDEOS = {
    "SP001": "SP001_3Ci-3ZYHBkM.mp4",
    "SP002": "SP002_qcVFnRcJcP4.mp4",
    "NORMAL001": "NORMAL001_dX4eGEFrthE.mp4",
    "UE001": "UE001_6V319bseO3U.mp4",
    "UE003": "UE003_k-inUpjRVLA.mp4",
    "STR001": "STR001_kuAMtoUJGgE.mp4",
}

# source_id -> channel / creator (from Stage 12B provenance info.json)
SOURCE_CREATOR = {
    "SP001": "GXS 3D",
    "SP002": "BetterLifeCreations",
    "NORMAL001": "TheeHustleHouse",
    "UE001": "Callum Coles",
    "UE003": "NeedItMakeIt",
    "STR001": "3D Printsmith",
}

# source_id -> failure type (from Stage 12A candidate metadata; normal is the
# non-failure class for NORMAL001). Preserved independently from the binary label.
SOURCE_FAILURE_TYPE = {
    "SP001": "spaghetti",
    "SP002": "spaghetti",
    "NORMAL001": "normal",
    "UE001": "under_extrusion",
    "UE003": "under_extrusion",
    "STR001": "stringing",
}

# Stage 12B labelability verdicts per source (from pilot_labels.json).
# Only SP001 and NORMAL001 carry per-frame provisional labels.
SOURCE_LABELED = {"SP001", "NORMAL001"}

# Stage 12B provisional label -> dataset label (normal/failing/uncertain).
# Explicit mapping; nothing is silently converted. Frames labeled "uncertain"
# remain uncertain and are excluded from the training-ready folders.
LABEL_MAP = {
    "normal_provisional": "normal",
    "suspect_growth": "failing",
    "uncertain": "uncertain",
    "scene_transition_or_final_state": "uncertain",
}
UNKNOWN_LABEL_FALLBACK = "uncertain"

# A source video is the minimum split unit. No two videos were identified as the
# same physical print in Stage 12B, so each source is its own session group.
@dataclass
class SourceMeta:
    source_id: str
    duration_s: float
    fps: float
    width: int
    height: int


def probe_media(src: Path) -> SourceMeta:
    """Probe duration/fps/resolution of a video once (deterministic)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=duration,avg_frame_rate,width,height", "-of", "json", str(src)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.stdout)
    stream = data["streams"][0]
    num, den = stream["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    duration = float(stream.get("duration") or 0.0)
    return SourceMeta(
        source_id=src.parent.name,
        duration_s=duration,
        fps=fps,
        width=int(stream["width"]),
        height=int(stream["height"]),
    )


def frame_grid(duration_s: float, extraction_fps: float = EXTRACTION_FPS) -> list[float]:
    """Integer multiples of 1/extraction_fps up to floor(duration). Deterministic."""
    interval = 1.0 / extraction_fps
    n = int(duration_s * extraction_fps) + 1
    return [round(i * interval, 4) for i in range(n)]


def source_frame_index(time_s: float, fps: float) -> int:
    """Source (input) frame index closest to the extraction timestamp."""
    return int(round(time_s * fps))


def canonical_name(source_id: str, time_s: float, source_idx: int) -> str:
    """Deterministic file name for a labeled frame."""
    return f"{source_id}__t{time_s:07.3f}__f{source_idx:06d}.jpg"


def extract_sequence(src: Path, out_dir: Path, grid: list[float],
                     extraction_fps: float = EXTRACTION_FPS) -> list[Path]:
    """Extract exactly one frame per grid timestamp (deterministic, -ss seeking).

    The returned list has the same length/order as `grid`; frame k corresponds to
    timestamp `grid[k]` seconds. This intentionally matches the Stage 12B label
    grid (integer seconds at 1 FPS) instead of using an ffmpeg rate filter, which
    can drop the final partial second of a clip.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.glob("frame_*.jpg"):
        f.unlink()
    paths: list[Path] = []
    for k, ts in enumerate(grid):
        out = out_dir / f"frame_{k:04d}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{ts}", "-i", str(src), "-frames:v", "1",
             "-q:v", "2", str(out)],
            capture_output=True, check=True,
        )
        paths.append(out)
    return paths


def load_pilot_labels() -> dict[str, dict[float, str]]:
    """Load Stage 12B per-frame provisional labels keyed by (source, time_s)."""
    raw = json.loads((EXT / "labels" / "pilot_labels.json").read_text())
    result: dict[str, dict[float, str]] = {}
    for sid, entries in raw.get("videos", {}).items():
        result[sid] = {}
        for e in entries:
            if "time_s" in e and "label" in e:
                result[sid][float(e["time_s"])] = str(e["label"])
    return result


def map_label(provisional: str) -> str:
    """Map a Stage 12B provisional label to the dataset label vocabulary."""
    return LABEL_MAP.get(provisional, UNKNOWN_LABEL_FALLBACK)


def phash(bgr: np.ndarray, size: int = 32) -> int:
    """64-bit DCT-based perceptual hash (deterministic).

    Resize to `size`x`size` gray, 2D DCT, threshold the top-left 8x8 block at its
    median to produce 64 bits.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(gray)
    block = dct[:8, :8].flatten()
    med = float(np.median(block))
    bits = (block > med).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_exact_duplicates(records: list[dict]) -> tuple[list[dict], dict]:
    """Remove exact byte-identical duplicate frames (md5) deterministically.

    Keeps the first occurrence (records arrive sorted by source then time).
    Matches the instruction to avoid accidental duplicate-frame inflation while
    NOT removing near-duplicate temporal neighbors of continuous footage.
    """
    kept: list[dict] = []
    removed: list[dict] = []
    seen_md5: set[str] = set()
    for rec in records:
        if rec["frame_md5"] in seen_md5:
            removed.append(rec)
        else:
            seen_md5.add(rec["frame_md5"])
            kept.append(rec)
    return kept, {
        "method": "exact md5 (first occurrence kept)",
        "removed_total": len(removed),
        "groups": len({r["frame_md5"] for r in removed}),
    }


def near_duplicate_report(records: list[dict]) -> dict:
    """Diagnostic phash near-duplicate report (no removal).

    Temporal neighbors of a continuous video legitimately look similar; the
    instruction forbids aggressively removing them. This report distinguishes
    within-source (expected) from cross-source (rare accidental) near-duplicates.
    """
    unique = []
    seen_md5: set[str] = set()
    for r in records:
        if r["frame_md5"] in seen_md5:
            continue
        seen_md5.add(r["frame_md5"])
        unique.append(r)
    within = 0
    cross = []
    pairs_checked = 0
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            pairs_checked += 1
            a, b = unique[i], unique[j]
            if hamming(a["phash"], b["phash"]) <= DUP_HAMMING_THRESHOLD:
                if a["source_id"] == b["source_id"]:
                    within += 1
                else:
                    cross.append((a["source_id"], a["timestamp"],
                                  b["source_id"], b["timestamp"]))
    return {
        "threshold_hamming": DUP_HAMMING_THRESHOLD,
        "unique_frames_considered": len(unique),
        "pairs_checked": pairs_checked,
        "within_source_near_duplicate_pairs": within,
        "cross_source_near_duplicate_pairs": len(cross),
        "cross_source_detail": [{"source_a": x[0], "time_a": x[1],
                                 "source_b": x[2], "time_b": x[3]} for x in cross],
        "policy": "diagnostic only; no removal (temporal neighbours preserved)",
    }


def assign_split(sources: list[str], val_fraction: float = VAL_FRACTION) -> dict[str, str]:
    """Deterministic source-level train/validation assignment (approx 80/20).

    Uses sha256 of the source id; stable across runs and platforms.
    """
    bound = max(1, int(round(val_fraction * 100)))
    result: dict[str, str] = {}
    for sid in sorted(sources):
        digest = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
        result[sid] = "validation" if (digest % 100) < bound else "train"
    return result


def build_frame_records(
    source_id: str,
    video_filename: str,
    meta: SourceMeta,
    frames: list[Path],
    provisional: dict[float, str],
) -> list[dict]:
    """Build the full per-frame record for one source (deterministic)."""
    records: list[dict] = []
    for k, fpath in enumerate(frames):
        time_s = round(k / EXTRACTION_FPS, 4)
        provisional_label = provisional.get(time_s)
        if source_id not in SOURCE_LABELED or provisional_label is None:
            label = "uncertain"
            basis = "no per-frame provisional label in Stage 12B"
        else:
            label = map_label(provisional_label)
            basis = f"stage12b: {provisional_label}"
        records.append({
            "source_id": source_id,
            "video_filename": video_filename,
            "timestamp": time_s,
            "source_frame_index": source_frame_index(time_s, meta.fps),
            "extraction_fps": EXTRACTION_FPS,
            "segment_id": f"{source_id}__s0",
            "failure_type": SOURCE_FAILURE_TYPE.get(source_id, "unknown"),
            "label": label,
            "width": meta.width,
            "height": meta.height,
            "provenance": {
                "source": "stage12b",
                "pilot_basis": basis,
                "verification": "provisional_unverified",
            },
            "frame_relative": str(fpath.relative_to(EXT)),
        })
    return records


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_statistics(records: list[dict], split_map: dict[str, str],
                       dup_removed_frames: int = 0) -> dict:
    """Dataset statistics over the frame records (deterministic)."""
    from collections import Counter

    total = len(records)
    by_label = Counter(r["label"] for r in records)
    usable = [r for r in records if r["label"] in ("normal", "failing")]
    by_source: dict[str, dict] = {}
    for r in records:
        m = by_source.setdefault(r["source_id"], {
            "creator": SOURCE_CREATOR[r["source_id"]],
            "failure_type": r["failure_type"],
            "total_frames": 0, "normal": 0, "failing": 0, "uncertain": 0,
            "usable": 0, "duration_s": None, "split": split_map.get(r["source_id"], "train"),
        })
        m["total_frames"] += 1
        m[r["label"]] += 1
        if r["label"] in ("normal", "failing"):
            m["usable"] += 1
    by_failure_type: dict[str, dict] = {}
    for r in records:
        ft = r["failure_type"]
        m = by_failure_type.setdefault(ft, {"sessions": 0, "frames": 0})
        m["frames"] += 1
    for r in records:
        by_failure_type[r["failure_type"]]["sessions"] = len({
            x["source_id"] for x in records if x["failure_type"] == r["failure_type"]
        })
    return {
        "total_extracted": total,
        "labeled_normal": by_label.get("normal", 0),
        "labeled_failing": by_label.get("failing", 0),
        "uncertain": by_label.get("uncertain", 0),
        "usable_after_dedup": len(usable),
        "dup_removed_frames": dup_removed_frames,
        "by_source": by_source,
        "by_failure_type": by_failure_type,
    }


def build_external_pilot() -> dict:
    """Build the full external pilot dataset. Returns the summary dict.

    Side effects (all under datasets/external/video-frames/):
    - extracted/<sid>/sequence/       deterministic 1 FPS extraction
    - labeled/normal/, labeled/failing/  training-ready copies
    - metadata/external_pilot_dataset.json   full per-frame records
    - metadata/external_pilot_statistics.json
    - splits/external_train.json, splits/external_val.json
    - dedup_removed.json
    """
    provisional = load_pilot_labels()
    all_records: list[dict] = []

    for source_id in sorted(SOURCE_VIDEOS):
        src = EXT / "raw" / source_id / SOURCE_VIDEOS[source_id]
        meta = probe_media(src)
        grid = frame_grid(meta.duration_s)
        seq_dir = EXT / "extracted" / source_id / "sequence"
        frames = extract_sequence(src, seq_dir, grid)
        records = build_frame_records(source_id, SOURCE_VIDEOS[source_id], meta, frames,
                                      provisional.get(source_id, {}))
        # attach md5 + phash to every record (reads only, for dedup)
        for rec in records:
            fpath = EXT / rec["frame_relative"]
            rec["frame_md5"] = md5_of(fpath)
            img = cv2.imread(str(fpath))
            rec["phash"] = phash(img) if img is not None else 0
        all_records.extend(records)
        print(f"  {source_id}: {len(records)} frames @ {EXTRACTION_FPS} FPS "
              f"({meta.width}x{meta.height}, {meta.duration_s:.1f}s)")

    all_records.sort(key=lambda r: (r["source_id"], r["timestamp"]))

    # Deduplication: remove only exact byte-identical duplicate frames.
    kept_records, dedup_stats = find_exact_duplicates(all_records)
    removed_keys = {(r["source_id"], r["timestamp"]) for r in all_records
                    if r not in kept_records}
    for rec in all_records:
        if (rec["source_id"], rec["timestamp"]) in removed_keys:
            rec["dup_reason"] = "exact_md5_duplicate"
    near_dup = near_duplicate_report(kept_records)

    # Write labeled training-ready copies for normal/failing (post-dedup).
    # Frames removed as exact duplicates are excluded from the labeled folders.
    labeled_normal_dir = EXT / "labeled" / "normal"
    labeled_failing_dir = EXT / "labeled" / "failing"
    labeled_normal_dir.mkdir(parents=True, exist_ok=True)
    labeled_failing_dir.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for rec in kept_records:
        if rec["label"] not in ("normal", "failing"):
            continue
        name = canonical_name(rec["source_id"], rec["timestamp"], rec["source_frame_index"])
        dest = (labeled_normal_dir if rec["label"] == "normal" else labeled_failing_dir) / name
        shutil.copyfile(EXT / rec["frame_relative"], dest)
        rec["labeled_relative"] = str(dest.relative_to(EXT))
        n_copied += 1
    print(f"  labeled copies written (post-dedup): {n_copied}")

    # Splits (source-level, deterministic, approx 80/20)
    split_map = assign_split(list(SOURCE_VIDEOS), VAL_FRACTION)
    train_ids = sorted(s for s, v in split_map.items() if v == "train")
    val_ids = sorted(s for s, v in split_map.items() if v == "validation")
    splits_dir = EXT / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    splits = {
        "method": "sha256(source_id) % 100 < int(20) -> validation (source-level, deterministic)",
        "val_fraction": VAL_FRACTION,
        "external_train": {
            "sources": train_ids,
            "frames": [r for r in kept_records if split_map.get(r["source_id"]) == "train"],
        },
        "external_validation": {
            "sources": val_ids,
            "frames": [r for r in kept_records if split_map.get(r["source_id"]) == "validation"],
        },
    }
    (splits_dir / "external_train.json").write_text(json.dumps(splits["external_train"], indent=2))
    (splits_dir / "external_val.json").write_text(json.dumps(splits["external_validation"], indent=2))

    # Persist full frame records (phash retained for reproducible dedup reports)
    dump_records = list(kept_records)
    dataset_path = EXT / "metadata" / "external_pilot_dataset.json"
    dataset_path.write_text(json.dumps({
        "extraction_fps": EXTRACTION_FPS,
        "dedup_policy": "exact_md5 only; near-duplicate report is diagnostic",
        "frames": dump_records,
    }, indent=2))

    stats = compute_statistics(kept_records, split_map, dedup_stats["removed_total"])
    stats["near_duplicate_report"] = near_dup
    stats_path = EXT / "metadata" / "external_pilot_statistics.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    (EXT / "metadata" / "dedup_removed.json").write_text(json.dumps({
        "exact_md5_removed": dedup_stats,
        "near_duplicate_diagnostic": near_dup,
    }, indent=2))

    return {
        "total_extracted": stats["total_extracted"],
        "labeled_normal": stats["labeled_normal"],
        "labeled_failing": stats["labeled_failing"],
        "uncertain": stats["uncertain"],
        "duplicate_count": stats["dup_removed_frames"],
        "by_source": stats["by_source"],
        "by_failure_type": stats["by_failure_type"],
        "splits": {"external_train": train_ids, "external_validation": val_ids},
        "dedup": dedup_stats,
    }


# ---------------------------------------------------------------------------
# Diversity analysis (external pilot vs existing Kaggle training data)
# ---------------------------------------------------------------------------

def _load_probe(path: Path, grid: int = 256):
    """Read image, resize to grid x grid, and compute brightness/contrast/saturation."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    small = cv2.resize(img, (grid, grid), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "saturation": float(hsv_s.mean()),
        "phash": phash(small),
    }


def _sample_paths(root: Path, limit: int, seed: int = 0) -> list[Path]:
    all_paths = sorted(root.rglob("*.png"))
    if len(all_paths) <= limit:
        return all_paths
    step = len(all_paths) / float(limit)
    return [all_paths[int(i * step)] for i in range(limit)]


def diversity_analysis(records: list[dict], kaggle_limit: int = 400,
                       pair_limit: int = 2000, seed: int = 0) -> dict:
    """Compare external labeled frames against the Kaggle processed dataset.

    Objects are compared on a common 256x256 grid via grayscale brightness /
    contrast, HSV saturation, and a 64-bit perceptual hash. Pairwise Hamming
    distance is sampled deterministically (seeded) for:
      - intra-Kaggle (how close Kaggle frames are to each other)
      - external-vs-Kaggle (how far external frames are from Kaggle)
    """
    import random
    import statistics

    rng = random.Random(seed)

    kaggle_paths = []
    for sub in ("failing", "normal"):
        kaggle_paths.extend(_sample_paths(KAGGLE / sub, kaggle_limit // 2, seed))
    kaggle_probes = [p for p in (_load_probe(p) for p in kaggle_paths) if p is not None]

    ext_paths = [EXT / r["frame_relative"] for r in records
                 if r["label"] in ("normal", "failing")]
    ext_probes = [p for p in (_load_probe(p) for p in ext_paths) if p is not None]

    def dists(a: list[dict], b: list[dict], n: int) -> list[int]:
        out = []
        for _ in range(n):
            x, y = rng.choice(a), rng.choice(b)
            out.append(hamming(x["phash"], y["phash"]))
        return out

    intra_k = dists(kaggle_probes, kaggle_probes, pair_limit)
    ext_vs_k = dists(ext_probes, kaggle_probes, pair_limit)

    def aggregate(name: str, probes: list[dict]) -> dict:
        return {
            "name": name,
            "n": len(probes),
            "brightness_mean": round(statistics.mean(p["brightness"] for p in probes), 2),
            "brightness_std": round(statistics.pstdev(p["brightness"] for p in probes), 2),
            "contrast_mean": round(statistics.mean(p["contrast"] for p in probes), 2),
            "contrast_std": round(statistics.pstdev(p["contrast"] for p in probes), 2),
            "saturation_mean": round(statistics.mean(p["saturation"] for p in probes), 2),
        }

    dims_ext: dict[str, int] = {}
    for r in records:
        if r["label"] in ("normal", "failing"):
            dims_ext[f"{r['width']}x{r['height']}"] = dims_ext.get(f"{r['width']}x{r['height']}", 0) + 1

    return {
        "common_grid": 256,
        "kaggle_sampled": len(kaggle_probes),
        "external_sampled": len(ext_probes),
        "kaggle": aggregate("kaggle", kaggle_probes),
        "external": aggregate("external", ext_probes),
        "image_dimensions_external": dims_ext,
        "kaggle_image_dimensions": {"256x256": len(kaggle_paths)},
        "phash_hamming_distance": {
            "intra_kaggle_mean": round(statistics.mean(intra_k), 2),
            "intra_kaggle_median": round(statistics.median(intra_k), 2),
            "external_vs_kaggle_mean": round(statistics.mean(ext_vs_k), 2),
            "external_vs_kaggle_median": round(statistics.median(ext_vs_k), 2),
            "pair_limit": pair_limit,
            "seed": seed,
        },
        "note": "Higher external-vs-kaggle than intra-kaggle Hamming distance indicates the external pilot occupies a visually different domain.",
    }


def run_diversity_analysis() -> dict:
    records = json.loads((EXT / "metadata" / "external_pilot_dataset.json").read_text())["frames"]
    result = diversity_analysis(records)
    (EXT / "metadata" / "external_pilot_diversity.json").write_text(json.dumps(result, indent=2))
    return result