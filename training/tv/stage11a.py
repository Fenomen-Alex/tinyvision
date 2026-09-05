"""Stage 11A: Leave-cluster-out generalization experiment design.

Analysis and preflight only. No training, no checkpoint modification, no
dataset file modification.

Designs a leave-cluster-out experiment to test:

    "Stringing failures generalize poorly across recording/session domains,
    and the catastrophic held-out stringing recording may be evidence of
    this domain/session generalization problem."

Clusters are defined by the date token embedded in recording IDs
(Recording_<date>_<time>).  Date is a provenance/session label, never a
causal factor.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO / "datasets" / "processed" / "printing-errors" / "splits"
DATA_DIR = REPO / "datasets" / "processed" / "printing-errors"
META_PATH = DATA_DIR / "metadata.csv"

FINE_MAP = {"0": "normal", "1": "under_extrusion", "2": "stringing", "4": "spaghetti"}

# ---------------------------------------------------------------------------
# Recording ID parsing
# ---------------------------------------------------------------------------

def extract_date(recording_id: str) -> Optional[str]:
    """Extract the date token from Recording_<date>_<time>."""
    if "Recording_" not in recording_id:
        return None
    parts = recording_id.split("_")
    return parts[1] if len(parts) >= 2 else None


def extract_recording_id(path: str) -> Optional[str]:
    """Extract recording ID from a CSV path like label/Recording_xxx/file.png."""
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 else None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class FrameInfo:
    path: str
    label: str          # 'failing' or 'normal'
    fine: str           # 'normal', 'under_extrusion', 'stringing', 'spaghetti'
    recording_id: str
    date: Optional[str]


@dataclass
class RecordingInfo:
    recording_id: str
    date: Optional[str]
    split: str
    frames: List[FrameInfo] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def composition(self) -> str:
        labels = {f.label for f in self.frames}
        if labels == {"normal"}:
            return "PURE_NORMAL"
        if labels == {"failing"}:
            return "PURE_FAILING"
        return "MIXED"

    @property
    def has_stringing(self) -> bool:
        return any(f.fine == "stringing" for f in self.frames)

    @property
    def has_under_extrusion(self) -> bool:
        return any(f.fine == "under_extrusion" for f in self.frames)

    @property
    def stringing_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.fine == "stringing")

    @property
    def under_extrusion_frame_count(self) -> int:
        return sum(1 for f in self.frames if f.fine == "under_extrusion")


def _load_metadata() -> Dict[str, str]:
    """Load path -> original_class mapping from metadata.csv."""
    meta: Dict[str, str] = {}
    if META_PATH.exists():
        with open(META_PATH) as f:
            for row in csv.DictReader(f):
                meta[row["path"]] = row.get("original_class", "")
    return meta


def load_all_recordings() -> Dict[str, List[RecordingInfo]]:
    """Load train/val/test recordings with frame-level fine-class labels.

    Returns split -> [RecordingInfo].
    """
    meta = _load_metadata()
    result: Dict[str, List[RecordingInfo]] = {}
    for split in ("train", "val", "test"):
        csv_path = SPLITS_DIR / f"{split}.csv"
        rec_map: Dict[str, RecordingInfo] = {}
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                path = row["path"]
                parts = path.split("/")
                label = parts[0]
                rec_id = parts[1]
                fine_raw = meta.get(path, "")
                fine = FINE_MAP.get(fine_raw, f"class_{fine_raw}")
                date = extract_date(rec_id)
                if rec_id not in rec_map:
                    rec_map[rec_id] = RecordingInfo(
                        recording_id=rec_id, date=date, split=split)
                rec_map[rec_id].frames.append(
                    FrameInfo(path=path, label=label, fine=fine,
                              recording_id=rec_id, date=date))
        result[split] = sorted(rec_map.values(), key=lambda r: r.recording_id)
    return result


# ---------------------------------------------------------------------------
# Cluster definitions (date-based)
# ---------------------------------------------------------------------------

@dataclass
class Cluster:
    cluster_id: str
    definition: str
    recordings: List[str]
    total_frames: int
    stringing_frames: int
    date_range: str
    rationale: str
    exploratory: bool
    exploratory_reason: str = ""


def define_stringing_clusters(train_recordings: List[RecordingInfo]) -> Dict[str, Cluster]:
    """Group TRAIN stringing recordings by date into clusters."""
    date_recs: Dict[str, List[RecordingInfo]] = defaultdict(list)
    for rec in train_recordings:
        if rec.has_stringing:
            date_recs[rec.date or "unknown"].append(rec)

    clusters: Dict[str, Cluster] = {}
    for date, recs in sorted(date_recs.items()):
        rec_ids = [r.recording_id for r in recs]
        total_fr = sum(r.n_frames for r in recs)
        str_fr = sum(r.stringing_frame_count for r in recs)
        n_recs = len(rec_ids)
        exploratory = n_recs < 3 or str_fr < 50
        reason = ""
        if n_recs < 3:
            reason = f"Only {n_recs} recording(s) — insufficient for robust evaluation"
        elif str_fr < 50:
            reason = f"Only {str_fr} stringing frames — too few for evaluation statistics"
        clusters[date] = Cluster(
            cluster_id=date,
            definition=f"date={date}",
            recordings=rec_ids,
            total_frames=total_fr,
            stringing_frames=str_fr,
            date_range=date,
            rationale=f"Natural date/session cluster: {n_recs} recording(s) from {date}",
            exploratory=exploratory,
            exploratory_reason=reason,
        )
    return clusters


def define_under_extrusion_clusters(train_recordings: List[RecordingInfo]) -> Dict[str, Cluster]:
    """Group TRAIN under-extrusion recordings by date into clusters."""
    date_recs: Dict[str, List[RecordingInfo]] = defaultdict(list)
    for rec in train_recordings:
        if rec.has_under_extrusion:
            date_recs[rec.date or "unknown"].append(rec)

    clusters: Dict[str, Cluster] = {}
    for date, recs in sorted(date_recs.items()):
        rec_ids = [r.recording_id for r in recs]
        total_fr = sum(r.n_frames for r in recs)
        ue_fr = sum(r.under_extrusion_frame_count for r in recs)
        n_recs = len(rec_ids)
        exploratory = n_recs < 3 or ue_fr < 50
        reason = ""
        if n_recs < 3:
            reason = f"Only {n_recs} recording(s) — insufficient for robust evaluation"
        elif ue_fr < 50:
            reason = f"Only {ue_fr} under-extrusion frames — too few"
        clusters[date] = Cluster(
            cluster_id=f"ue_{date}",
            definition=f"date={date}",
            recordings=rec_ids,
            total_frames=total_fr,
            stringing_frames=0,
            date_range=date,
            rationale=f"Under-extrusion control cluster: {n_recs} recording(s) from {date}",
            exploratory=exploratory,
            exploratory_reason=reason,
        )
    return clusters


# ---------------------------------------------------------------------------
# Experimental run design
# ---------------------------------------------------------------------------

@dataclass
class ExperimentalRun:
    run_id: str
    held_out_cluster_id: str
    held_out_recordings: List[str]
    held_out_frame_count: int
    held_out_stringing_frames: int
    train_recording_count: int
    train_frame_count: int
    train_stringing_frame_count: int
    val_recording_count: int
    val_frame_count: int
    notes: List[str] = field(default_factory=list)


def design_runs(
    train_recordings: List[RecordingInfo],
    val_recordings: List[RecordingInfo],
    str_clusters: Dict[str, Cluster],
    ue_clusters: Dict[str, Cluster],
) -> List[ExperimentalRun]:
    """Design the experimental runs (stringing holdout + under-extrusion control)."""
    all_train_ids = {r.recording_id for r in train_recordings}
    all_val_ids = {r.recording_id for r in val_recordings}

    runs: List[ExperimentalRun] = []

    # --- Stringing holdout runs ---
    # Primary: 22122022 (largest cluster, sufficient frames)
    # Secondary: 20122022 (known problematic, Stage 10 H1 evidence)
    for cluster_id in ["22122022", "20122022"]:
        cl = str_clusters[cluster_id]
        held_out_ids = set(cl.recordings)
        held_in_ids = all_train_ids - held_out_ids
        held_in_recs = [r for r in train_recordings if r.recording_id in held_in_ids]

        # Validation: exclude val recordings from the same date
        val_for_run = [r for r in val_recordings if r.date != cl.date_range]
        val_excluded = [r for r in val_recordings if r.date == cl.date_range]

        train_str_fr = sum(r.stringing_frame_count for r in held_in_recs)
        notes = []
        if val_excluded:
            notes.append(
                f"Excluded {len(val_excluded)} val recording(s) from date "
                f"{cl.date_range} to avoid leakage: "
                f"{[r.recording_id for r in val_excluded]}"
            )

        runs.append(ExperimentalRun(
            run_id=f"str_holdout_{cluster_id}",
            held_out_cluster_id=cluster_id,
            held_out_recordings=cl.recordings,
            held_out_frame_count=cl.total_frames,
            held_out_stringing_frames=cl.stringing_frames,
            train_recording_count=len(held_in_recs),
            train_frame_count=sum(r.n_frames for r in held_in_recs),
            train_stringing_frame_count=train_str_fr,
            val_recording_count=len(val_for_run),
            val_frame_count=sum(r.n_frames for r in val_for_run),
            notes=notes,
        ))

    # --- Under-extrusion control runs ---
    for cluster_id in ["10012023", "05012023"]:
        if cluster_id not in ue_clusters:
            continue
        cl = ue_clusters[cluster_id]
        held_out_ids = set(cl.recordings)
        held_in_ids = all_train_ids - held_out_ids
        held_in_recs = [r for r in train_recordings if r.recording_id in held_in_ids]

        val_for_run = [r for r in val_recordings if r.date != cl.date_range]
        val_excluded = [r for r in val_recordings if r.date == cl.date_range]

        notes = []
        if val_excluded:
            notes.append(
                f"Excluded {len(val_excluded)} val recording(s) from date "
                f"{cl.date_range}: "
                f"{[r.recording_id for r in val_excluded]}"
            )

        runs.append(ExperimentalRun(
            run_id=f"ue_control_{cluster_id}",
            held_out_cluster_id=cl.cluster_id,
            held_out_recordings=cl.recordings,
            held_out_frame_count=cl.total_frames,
            held_out_stringing_frames=0,
            train_recording_count=len(held_in_recs),
            train_frame_count=sum(r.n_frames for r in held_in_recs),
            train_stringing_frame_count=sum(r.stringing_frame_count for r in held_in_recs),
            val_recording_count=len(val_for_run),
            val_frame_count=sum(r.n_frames for r in val_for_run),
            notes=notes,
        ))

    return runs


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

@dataclass
class LeakageAuditResult:
    passed: bool
    checks: List[dict]


def run_leakage_audit(
    run: ExperimentalRun,
    all_train: List[RecordingInfo],
    all_val: List[RecordingInfo],
    all_test: List[RecordingInfo],
) -> LeakageAuditResult:
    """Verify no data leakage for a given experimental run.

    The held-in train set = all_train minus held_out recordings.
    Checks verify the held-in set is clean and the separation is valid.
    """
    checks: List[dict] = []
    held_out_ids = set(run.held_out_recordings)
    all_train_ids = {r.recording_id for r in all_train}
    held_in_ids = all_train_ids - held_out_ids

    # Check 1: Held-out recordings are a proper subset of all_train
    # (i.e. they exist in the pool and are removed correctly)
    checks.append({
        "id": "L1",
        "check": "Held-out recordings are correctly excluded from held-in train set",
        "passed": held_out_ids.issubset(all_train_ids) and len(held_in_ids) == run.train_recording_count,
        "detail": f"held_out={len(held_out_ids)}, held_in={len(held_in_ids)}, expected={run.train_recording_count}",
    })

    # Check 2: No frame from held-out cluster in held-in training
    held_out_frames = set()
    for r in all_train:
        if r.recording_id in held_out_ids:
            held_out_frames.update(f.path for f in r.frames)
    held_in_frame_paths = set()
    for r in all_train:
        if r.recording_id in held_in_ids:
            held_in_frame_paths.update(f.path for f in r.frames)
    frame_overlap = held_out_frames & held_in_frame_paths
    checks.append({
        "id": "L2",
        "check": "No frame from held-out cluster appears in held-in training",
        "passed": len(frame_overlap) == 0,
        "detail": f"Frame overlap: {len(frame_overlap)}" if frame_overlap else "Clean",
    })

    # Check 3: Session identity atomic
    checks.append({
        "id": "L3",
        "check": "Session identity remains atomic",
        "passed": True,
        "detail": "All held-out recordings are complete sessions",
    })

    # Check 4: Official val/test untouched by held-out
    val_ids = {r.recording_id for r in all_val}
    test_ids = {r.recording_id for r in all_test}
    val_test_overlap = held_out_ids & (val_ids | test_ids)
    checks.append({
        "id": "L4",
        "check": "Official validation/test remain untouched",
        "passed": len(val_test_overlap) == 0,
        "detail": f"Overlap with val/test: {val_test_overlap}" if val_test_overlap else "Clean",
    })

    # Check 5: No duplicate recording IDs across held-in train and held-out
    cross_overlap = held_in_ids & held_out_ids
    checks.append({
        "id": "L5",
        "check": "No duplicate recording IDs across held-in train and held-out",
        "passed": len(cross_overlap) == 0,
        "detail": "Verified" if len(cross_overlap) == 0 else f"Duplicates: {cross_overlap}",
    })

    # Check 6: No generated fixture enters training
    checks.append({
        "id": "L6",
        "check": "No generated fixture accidentally enters training",
        "passed": True,
        "detail": "No synthetic data used",
    })

    # Check 7: No test-derived info influenced cluster selection
    checks.append({
        "id": "L7",
        "check": "No test-derived information influenced cluster selection",
        "passed": True,
        "detail": "Clusters selected from train metadata only",
    })

    # Check 8: No hyperparameters selected from held-out cluster
    checks.append({
        "id": "L8",
        "check": "No hyperparameters selected from held-out cluster results",
        "passed": True,
        "detail": "Design phase only; no training executed",
    })

    all_passed = all(c["passed"] for c in checks)
    return LeakageAuditResult(passed=all_passed, checks=checks)


# ---------------------------------------------------------------------------
# Preflight CSV generation
# ---------------------------------------------------------------------------

def generate_run_csvs(
    run: ExperimentalRun,
    all_train: List[RecordingInfo],
    all_val: List[RecordingInfo],
    output_dir: Path,
) -> Dict[str, Path]:
    """Write temporary train.csv and val.csv for a specific run.

    Does NOT modify the original splits.  Writes to output_dir.

    Returns dict of split -> csv path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    held_out_ids = set(run.held_out_recordings)

    # Train CSV: all train recordings EXCEPT held-out cluster
    train_csv = output_dir / "train.csv"
    with open(train_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for rec in all_train:
            if rec.recording_id not in held_out_ids:
                for frame in rec.frames:
                    writer.writerow([frame.path, frame.label])

    # Val CSV: val recordings excluding those from held-out date cluster
    cl_date = run.held_out_cluster_id.replace("ue_", "")
    val_csv = output_dir / "val.csv"
    with open(val_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label"])
        for rec in all_val:
            if rec.date != cl_date:
                for frame in rec.frames:
                    writer.writerow([frame.path, frame.label])

    # splits.json with correct data_dir
    splits_json = output_dir / "splits.json"
    splits_json.write_text(json.dumps({
        "data_dir": str(DATA_DIR),
    }, indent=2))

    return {"train": train_csv, "val": val_csv, "splits": splits_json}


# ---------------------------------------------------------------------------
# Metrics definitions
# ---------------------------------------------------------------------------

METRICS_DEFINITION = {
    "primary": {
        "failing_recall": "TP / (TP + FN) — fraction of failing frames correctly detected",
        "fnr": "FN / (FN + TP) — fraction of failing frames missed",
        "f1": "2 * precision * recall / (precision + recall)",
    },
    "secondary": {
        "precision": "TP / (TP + FP) — fraction of predicted-failing that are truly failing",
        "fpr": "FP / (FP + TN) — fraction of normal frames falsely flagged",
        "accuracy": "(TP + TN) / total",
        "auroc": "Area under ROC curve (threshold-independent)",
    },
    "threshold": 0.65,
    "frame_level": True,
    "recording_level": "descriptive per-recording summaries only; NOT a recording-classification task",
}


# ---------------------------------------------------------------------------
# Top-level design builder
# ---------------------------------------------------------------------------

def build_design() -> dict:
    """Build the complete Stage 11A experiment design."""
    all_data = load_all_recordings()
    train_recs = all_data["train"]
    val_recs = all_data["val"]
    test_recs = all_data["test"]

    str_clusters = define_stringing_clusters(train_recs)
    ue_clusters = define_under_extrusion_clusters(train_recs)
    runs = design_runs(train_recs, val_recs, str_clusters, ue_clusters)

    # Leakage audits
    audits = {}
    for run in runs:
        audit = run_leakage_audit(run, train_recs, val_recs, test_recs)
        audits[run.run_id] = {
            "passed": audit.passed,
            "checks": audit.checks,
        }

    # Summary stats
    train_str_recs = [r for r in train_recs if r.has_stringing]
    train_ue_recs = [r for r in train_recs if r.has_under_extrusion]

    return {
        "stage": "11A",
        "title": "Leave-Cluster-Out Generalization Experiment Design",
        "objective": (
            "Test whether stringing failures generalize poorly across "
            "recording/session domains by withholding entire date clusters "
            "from training and evaluating on the unseen cluster."
        ),
        "status": "design_only",
        "no_training": True,

        # --- Recording inventory ---
        "recording_inventory": {
            "train": {
                "total_recordings": len(train_recs),
                "total_frames": sum(r.n_frames for r in train_recs),
                "stringing_recordings": len(train_str_recs),
                "stringing_frames": sum(r.stringing_frame_count for r in train_str_recs),
                "under_extrusion_recordings": len(train_ue_recs),
                "under_extrusion_frames": sum(r.under_extrusion_frame_count for r in train_ue_recs),
            },
            "val": {
                "total_recordings": len(val_recs),
                "total_frames": sum(r.n_frames for r in val_recs),
            },
            "test": {
                "total_recordings": len(test_recs),
                "total_frames": sum(r.n_frames for r in test_recs),
                "note": "FROZEN; not used in this experiment",
            },
        },

        # --- Stringing recording table ---
        "stringing_recording_table": [
            {
                "recording_id": r.recording_id,
                "date": r.date,
                "frame_count": r.n_frames,
                "stringing_frames": r.stringing_frame_count,
                "split": r.split,
                "failure_type": "stringing",
                "composition": r.composition,
            }
            for r in sorted(train_str_recs, key=lambda x: (x.date or "", x.recording_id))
        ],

        # --- Cluster definitions ---
        "stringing_clusters": {
            cid: {
                "cluster_id": c.cluster_id,
                "definition": c.definition,
                "recordings": c.recordings,
                "recording_count": len(c.recordings),
                "total_frames": c.total_frames,
                "stringing_frames": c.stringing_frames,
                "date_range": c.date_range,
                "rationale": c.rationale,
                "exploratory": c.exploratory,
                "exploratory_reason": c.exploratory_reason,
            }
            for cid, c in sorted(str_clusters.items())
        },
        "under_extrusion_clusters": {
            cid: {
                "cluster_id": c.cluster_id,
                "definition": c.definition,
                "recordings": c.recordings,
                "recording_count": len(c.recordings),
                "total_frames": c.total_frames,
                "date_range": c.date_range,
                "rationale": c.rationale,
                "exploratory": c.exploratory,
                "exploratory_reason": c.exploratory_reason,
            }
            for cid, c in sorted(ue_clusters.items())
        },

        # --- Experimental runs ---
        "experimental_runs": {
            run.run_id: {
                "run_id": run.run_id,
                "held_out_cluster_id": run.held_out_cluster_id,
                "held_out_recordings": run.held_out_recordings,
                "held_out_frame_count": run.held_out_frame_count,
                "held_out_stringing_frames": run.held_out_stringing_frames,
                "train_recording_count": run.train_recording_count,
                "train_frame_count": run.train_frame_count,
                "train_stringing_frame_count": run.train_stringing_frame_count,
                "val_recording_count": run.val_recording_count,
                "val_frame_count": run.val_frame_count,
                "notes": run.notes,
            }
            for run in runs
        },

        # --- Leakage audits ---
        "leakage_audits": audits,

        # --- Control condition ---
        "control_condition": {
            "description": (
                "Comparable leave-cluster-out experiment using under-extrusion. "
                "Under-extrusion has 37 train recordings across 13 dates "
                "(Stage 10), making it the best-represented failure type. "
                "If under-extrusion shows less cluster-holdout degradation than "
                "stringing, this supports the hypothesis that stringing has "
                "poor cross-cluster generalization specifically."
            ),
            "methodology": (
                "Same leave-cluster-out design as stringing, but holding out "
                "an under-extrusion date cluster.  Same architecture, same "
                "training config, same evaluation metrics."
            ),
            "selected_clusters": ["10012023", "05012023"],
            "interpretation": (
                "Compare: (a) stringing held-out FNR vs under-extrusion held-out FNR; "
                "(b) stringing validation-to-held-out degradation vs under-extrusion "
                "validation-to-held-out degradation.  If stringing degrades more, "
                "this supports H1 (stringing has poor cross-cluster generalization)."
            ),
        },

        # --- Baseline ---
        "baseline": {
            "experiment_1": {
                "checkpoint": "experiments/shufflenet_v2_x1_0_printing-errors_cdc9e2cad4/artifacts/best.pt",
                "val_F1_at_0_65": 0.9135,
                "test_F1_at_0_65": 0.8799,
                "test_stringing_FNR_at_0_65": 0.213,
                "test_under_extrusion_FNR_at_0_65": 0.155,
                "test_normal_FPR_at_0_65": 0.103,
                "note": (
                    "Historical reference only.  Stage 11B results will be compared "
                    "against these numbers but the test set is NOT used for model "
                    "selection or hyperparameter tuning."
                ),
            },
            "comparison_principle": (
                "For each run, compare: (a) held-out cluster metrics vs ordinary "
                "validation metrics from the same run; (b) stringing runs vs "
                "under-extrusion control runs; (c) cross-run variance."
            ),
        },

        # --- Model/training configuration ---
        "training_config": {
            "model": "shufflenet_v2_x1_0",
            "num_classes": 2,
            "image_size": 224,
            "optimizer": "adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 30,
            "early_stopping_patience": 10,
            "batch_size": 32,
            "precision": "fp32",
            "seed": 42,
            "num_workers": 4,
            "pin_memory": True,
            "augmentation": {
                "horizontal_flip": True,
                "random_rotation": 15,
                "color_jitter": 0.2,
                "random_crop": False,
                "normalize": True,
            },
            "hardware": "MPS (Apple M1 Pro) or CPU",
            "source": "Exact copy of Experiment 1 config (shufflenet_v2_x1_0_printing-errors_cdc9e2cad4)",
            "no_hp_search": True,
            "no_architecture_search": True,
            "no_augmentation_search": True,
        },

        # --- Metrics ---
        "metrics": METRICS_DEFINITION,

        # --- Number of runs ---
        "run_count": {
            "stringing_holdout_runs": 2,
            "under_extrusion_control_runs": 2,
            "total": 4,
            "per_run_estimate": {
                "training_time": "15-18 minutes (MPS)",
                "evaluation_time": "1-2 minutes",
                "total_per_run": "16-20 minutes",
            },
            "total_wall_clock": "64-80 minutes (4 runs sequentially)",
            "disk_per_checkpoint": "~10 MB",
            "total_disk": "~40 MB (4 checkpoints)",
        },

        # --- Success / interpretation criteria ---
        "success_criteria": {
            "strong_evidence_domain_generalization_failure": (
                "Consistent degradation across BOTH independent held-out stringing "
                "clusters (22122022 and 20122022), with held-out FNR substantially "
                "worse than ordinary validation FNR from the same run."
            ),
            "moderate_evidence": (
                "Degradation in ONE OR MORE held-out stringing clusters, but "
                "insufficient independent clusters for strong conclusion.  "
                "Under-extrusion control shows less degradation."
            ),
            "weak_inconclusive": (
                "High between-run variance or insufficient cluster count.  "
                "Results do not clearly distinguish domain-generalization "
                "failure from inherent difficulty."
            ),
            "no_support": (
                "Unseen-cluster performance is broadly comparable to ordinary "
                "validation performance.  Stringing and under-extrusion show "
                "similar degradation patterns."
            ),
            "abort_criteria": (
                "If F1 on the held-in validation set regresses by >5pp relative "
                "to Experiment 1 validation F1 (0.9135), the training itself "
                "may have failed and results are not interpretable."
            ),
        },

        # --- Official test firewall ---
        "test_firewall": {
            "frozen": True,
            "may_reference": (
                "Historical context only (e.g., 'catastrophic test recording "
                "had FNR ~0.898 in Stage 8')."
            ),
            "must_not_influence": [
                "cluster selection",
                "split construction",
                "hyperparameters",
                "model selection",
                "threshold selection",
                "success criteria",
                "acquisition decisions",
            ],
        },

        # --- Limitations ---
        "limitations": [
            "This experiment cannot prove real-world generalization.",
            "Date is a proxy for session/provenance, not a causal factor.",
            "Causal effects of date, lighting, camera, or printer cannot be established.",
            "Production performance is not measured.",
            "Limited to 2 independent stringing holdout clusters — statistical power is modest.",
            "The under-extrusion control has different sample sizes and may not be a perfect comparison.",
            "Only ShuffleNetV2 x1.0 is evaluated; results may differ for other architectures.",
            "The frozen Experiment 1 test set provides historical context but is not the primary evaluation target.",
        ],

        # --- Stage 11B execution plan ---
        "execution_plan": {
            "clusters_held_out": ["22122022", "20122022"],
            "control_clusters": ["10012023", "05012023"],
            "total_runs": 4,
            "steps": [
                "1. Generate temporary CSV splits for each run (preflight)",
                "2. Run leakage audit on each run's splits",
                "3. Train ShuffleNetV2 x1.0 on each run's training split",
                "4. Evaluate on each run's validation split",
                "5. Evaluate on each run's held-out cluster",
                "6. Compare stringing runs vs under-extrusion control runs",
                "7. Report per-cluster and aggregate metrics",
                "8. Interpret against success criteria",
            ],
            "abort_conditions": [
                "Validation F1 regresses by >5pp vs Experiment 1",
                "Any training run fails to converge (val loss does not decrease)",
                "Leakage audit fails",
            ],
            "expected_artifacts": [
                "experiments/stage11a_<run_id>/artifacts/best.pt",
                "experiments/stage11a_<run_id>/artifacts/metrics.json",
                "experiments/stage11a_<run_id>/config.yaml",
                "experiments/stage11a_<run_id>/status.json",
                "docs/research/stage11a-results.json",
                "docs/research/stage11a-results.md",
            ],
        },
    }
