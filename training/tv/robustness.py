"""Offline robustness & generalization analysis (Stage 2B).

Evaluates a FROZEN classifier across controlled image perturbations to determine
whether the model has learned defect characteristics or has overfit to
recording/domain-specific artifacts.

Design principles:
  * The model is never trained or modified here.
  * Original test images are never modified. Perturbations are applied
    on-the-fly to derived copies (in memory), so the frozen split is untouched.
  * Every perturbation has a fixed severity schedule; any stochastic perturbation
    uses a seeded generator so results are reproducible.
  * Metrics are reported both in aggregate and per-recording / per-failure-type /
    per-object (via the raw dataset metadata), and always relative to an
    identity (no-perturbation) baseline so degradation is explicit.
"""
from __future__ import annotations

import csv as csvmod
import io
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import ExperimentConfig
from .deepeval import (
    _binary_metrics,
    _get_device,
    _recording_metrics,
    _resolve_checkpoint,
)

# Original-class codes from the Kaggle printing-errors metadata CSV.
FAILURE_TYPE_NAMES = {0: "normal", 1: "under_extrusion", 2: "stringing", 4: "spaghetti"}


# ---------------------------------------------------------------------------
# Perturbation primitives (operate on an RGB uint8 numpy image (H, W, 3)).
# ---------------------------------------------------------------------------

def _apply_brightness(img: np.ndarray, gain: float) -> np.ndarray:
    out = img.astype(np.float32) * gain
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    out = alpha * (img.astype(np.float32) - 128.0) + 128.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _apply_noise(img: np.ndarray, sigma: float, rng) -> np.ndarray:
    if sigma <= 0:
        return img
    noise = rng.normal(0.0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])[1]
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return decoded[:, :, ::-1].copy()  # BGR -> RGB to match input convention


def _apply_downscale(img: np.ndarray, factor: float) -> np.ndarray:
    h, w = img.shape[:2]
    sh, sw = max(1, int(round(h * factor))), max(1, int(round(w * factor)))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_rotation(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, mat, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_crop(img: np.ndarray, keep: float) -> np.ndarray:
    h, w = img.shape[:2]
    ch = max(1, int(round(h * keep)))
    cw = max(1, int(round(w * keep)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def _apply_perspective(img: np.ndarray, strength: float, rng) -> np.ndarray:
    h, w = img.shape[:2]
    maxshift = strength * 0.04 * max(h, w)  # up to ~4% per unit strength
    shift = rng.uniform(-maxshift, maxshift, size=4)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([
        [shift[0], shift[1]],
        [w - 1 + shift[2], shift[3]],
        [w - 1 - shift[2], h - 1 + shift[1]],
        [shift[0], h - 1 - shift[3]],
    ])
    mat = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img, mat, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_combined(img: np.ndarray, level: dict, rng) -> np.ndarray:
    """A realistic stack of simultaneously-applied impairments."""
    out = img
    if level.get("downscale"):
        out = _apply_downscale(out, level["downscale"])
    if level.get("brightness"):
        out = _apply_brightness(out, level["brightness"])
    if level.get("contrast"):
        out = _apply_contrast(out, level["contrast"])
    if level.get("blur"):
        out = _apply_gaussian_blur(out, level["blur"])
    if level.get("noise"):
        out = _apply_noise(out, level["noise"], rng)
    if level.get("jpeg"):
        out = _apply_jpeg(out, level["jpeg"])
    return out


# Perturbation definitions with a fixed 3-level severity schedule.
# Each level is [mild, moderate, severe].
PERTURBATIONS: dict[str, dict] = {
    "brightness": {
        "levels": [1.15, 1.35, 1.6],
        "apply": lambda img, p, rng: _apply_brightness(img, p),
    },
    "darken": {
        "levels": [0.87, 0.73, 0.6],
        "apply": lambda img, p, rng: _apply_brightness(img, p),
    },
    "contrast": {
        "levels": [0.85, 0.7, 0.55],
        "apply": lambda img, p, rng: _apply_contrast(img, p),
    },
    "blur": {
        "levels": [1.0, 2.0, 3.0],
        "apply": lambda img, p, rng: _apply_gaussian_blur(img, p),
    },
    "noise": {
        "levels": [10.0, 25.0, 50.0],
        "apply": lambda img, p, rng: _apply_noise(img, p, rng),
    },
    "jpeg": {
        "levels": [70, 50, 25],
        "apply": lambda img, p, rng: _apply_jpeg(img, p),
    },
    "downscale": {
        "levels": [0.8, 0.6, 0.4],
        "apply": lambda img, p, rng: _apply_downscale(img, p),
    },
    "rotation": {
        "levels": [5.0, 10.0, 15.0],
        "apply": lambda img, p, rng: _apply_rotation(img, p),
    },
    "crop": {
        "levels": [0.9, 0.8, 0.7],
        "apply": lambda img, p, rng: _apply_crop(img, p),
    },
    "perspective": {
        "levels": [0.5, 1.0, 2.0],
        "apply": lambda img, p, rng: _apply_perspective(img, p, rng),
    },
    "combined": {
        "levels": [
            {"downscale": 0.85, "brightness": 1.1, "contrast": 0.9,
             "blur": 1.0, "noise": 8.0, "jpeg": 80},
            {"downscale": 0.7, "brightness": 1.2, "contrast": 0.8,
             "blur": 1.8, "noise": 18.0, "jpeg": 60},
            {"downscale": 0.55, "brightness": 1.3, "contrast": 0.65,
             "blur": 2.8, "noise": 32.0, "jpeg": 40},
        ],
        "apply": _apply_combined,
    },
}

SEVERITY_NAMES = ["mild", "moderate", "severe"]


def severity_schedule(perturbation: str) -> int:
    if perturbation == "identity":
        return 1
    return len(PERTURBATIONS[perturbation]["levels"])


# ---------------------------------------------------------------------------
# Metadata join (raw Kaggle CSV by image filename).
# ---------------------------------------------------------------------------

def load_resolved_metadata(data_dir, split="test"):
    """Attach per-sample raw metadata to the frozen split.

    ``data_dir`` may be relative to the project root; it is resolved by
    searching upward from the package directory (mirrors the loaders). Returns
    a list of dicts (one per split row) with keys: path, label, recording,
    image, failure_type, shape, brightness, mean_brightness.
    """
    data_dir = _resolve_data_dir(data_dir)
    split_csv = data_dir / "splits" / f"{split}.csv"

    import json as jsonmod
    manifest = {}
    splits_json = data_dir / "splits" / "splits.json"
    if splits_json.exists():
        manifest = jsonmod.loads(Path(splits_json).read_text())

    # Locate the raw Kaggle metadata CSV relative to the dataset/repo root.
    raw_path = None
    if manifest.get("raw_metadata_csv"):
        cand = Path(manifest["raw_metadata_csv"])
        cand = cand if cand.is_absolute() else _repo_root(data_dir) / cand
        if cand.exists():
            raw_path = cand
    if raw_path is None:
        for cand in [
            data_dir / "all_images_no_filter.csv",
            data_dir.parent / "Printing_Errors" / "general_data" / "all_images_no_filter.csv",
            _repo_root(data_dir) / "datasets" / "Printing_Errors" / "general_data" / "all_images_no_filter.csv",
        ]:
            if cand.exists():
                raw_path = cand
                break
    if raw_path is None:
        # Fall back to a broad search under the repo root.
        hits = sorted(_repo_root(data_dir).rglob("all_images_no_filter.csv"))
        if hits:
            raw_path = hits[0]

    raw_rows = {}
    resolved = []
    with open(split_csv) as f:
        reader = csvmod.DictReader(f)
        for row in reader:
            path = row["path"]
            if raw_path is not None and not raw_rows:
                with open(raw_path) as rf:
                    for rr in csvmod.DictReader(rf, delimiter=";"):
                        raw_rows[rr["image"]] = rr
            image_name = Path(path).name
            meta = raw_rows.get(image_name, {})
            resolved.append({
                "path": path,
                "abs_path": str((data_dir / path) if not Path(path).is_absolute() else path),
                "label": row["label"],
                "image": image_name,
                "recording": Path(path).parts[1] if len(Path(path).parts) >= 3 else "unknown",
                "failure_type": FAILURE_TYPE_NAMES.get(
                    int(meta["class"]), meta.get("class", "unknown")
                ) if meta.get("class") not in (None, "") else "normal",
                "shape": meta.get("shape", "unknown"),
                "brightness": meta.get("brightness"),
                "mean_brightness": meta.get("mean_brightness"),
            })
    return resolved


def _resolve_data_dir(data_dir: str | Path) -> Path:
    """Resolve a data_dir (possibly relative to the repo root) to an absolute path."""
    p = Path(data_dir)
    if p.is_absolute():
        return p
    # Search from the package directory upward to the project root.
    start = Path(__file__).resolve().parent.parent.parent
    root = start
    for _ in range(6):
        if (root / p / "splits").exists():
            return root / p
        parent = root.parent
        if parent == root:
            break
        root = parent
    # Fall back to interpreting relative to the package's project root.
    return start / p


def _repo_root(data_dir: Path) -> Path:
    """Project root that contains the given data_dir (used for raw-CSV discovery)."""
    p = data_dir.resolve()
    for _ in range(8):
        if (p / "datasets").exists() and (p / "training").exists():
            return p
        parent = p.parent
        if parent == p:
            break
        p = parent
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Inference with perturbation.
# ---------------------------------------------------------------------------

def _infer_with_perturbation(model, device, resolved, transform,
                             perturb_fn, fail_idx, batch_size):
    """Run inference over the resolved split applying ``perturb_fn`` per image.

    Attaches per-sample provenance (recording, failure_type, shape) from the
    resolved rows. Returns a list of sample dicts (deepeval convention).
    """
    import torch
    from PIL import Image

    samples = []
    images, meta_batch = [], []
    num_classes = None

    def flush():
        nonlocal images, meta_batch, num_classes
        if not images:
            return
        inputs = torch.stack(images).to(device)
        with torch.no_grad():
            outputs = model(inputs)
        if num_classes is None:
            num_classes = outputs.shape[1]
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        for i in range(inputs.size(0)):
            row = probs[i]
            pred = int(np.argmax(row))
            other_idx = 1 - fail_idx if num_classes == 2 else 0
            samples.append({
                "fail_score": float(row[fail_idx]),
                "norm_score": float(row[other_idx]),
                "true_idx": int(meta_batch[i]["label_idx"]),
                "pred_idx": pred,
                "conf": float(row.max()),
                "recording": meta_batch[i]["recording"],
                "failure_type": meta_batch[i]["failure_type"],
                "shape": meta_batch[i]["shape"],
                "brightness": meta_batch[i]["brightness"],
            })
        images, meta_batch = [], []

    for row in resolved:
        img_path = row["abs_path"]
        pil = Image.open(img_path).convert("RGB")
        arr = np.array(pil)
        if perturb_fn is not None:
            arr = perturb_fn(arr)
        tensor = transform(Image.fromarray(arr, mode="RGB"))
        images.append(tensor)
        meta_batch.append({
            "label_idx": 0 if row["label"] == "failing" else 1,
            "recording": row["recording"],
            "failure_type": row.get("failure_type"),
            "shape": row.get("shape"),
            "brightness": row.get("brightness"),
        })
        if len(images) == batch_size:
            flush()
    flush()
    return samples


# ---------------------------------------------------------------------------
# Aggregation & degradation.
# ---------------------------------------------------------------------------

def _segment(samples, y_true, y_pred):
    acc = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return round(acc, 4)


def _compute_all(samples, fail_idx, class_names):
    y_true = np.array([s["true_idx"] for s in samples])
    y_pred = np.array([s["pred_idx"] for s in samples])
    fail_scores = np.array([s["fail_score"] for s in samples])
    m = _binary_metrics(y_true, y_pred, pos_integer=fail_idx)
    try:
        from sklearn.metrics import roc_auc_score
        auroc = round(float(roc_auc_score(y_true, 1.0 - fail_scores)), 4)
    except ValueError:
        auroc = None
    out = {
        "accuracy": m["accuracy"],
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "fnr": m["fnr"],
        "fpr": m["fpr"],
        "confusion": {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]},
        "auroc": auroc,
    }
    # per-recording
    recs = _recording_metrics(samples, class_names, fail_idx)
    out["recordings"] = recs
    # per-failure-type and per-shape pooled failing view
    out["by_failure_type"] = _pool_by(samples, fail_idx, "failure_type")
    out["by_shape"] = _pool_by(samples, fail_idx, "shape")
    return out


def _pool_by(samples, fail_idx, field):
    groups = defaultdict(list)
    for s in samples:
        v = s.get(field)
        if v is None or str(v) == "" or str(v) == "unknown":
            continue
        groups[str(v)].append(s)
    out = {}
    for value, items in groups.items():
        y_true = np.array([s["true_idx"] for s in items])
        y_pred = np.array([s["pred_idx"] for s in items])
        m = _binary_metrics(y_true, y_pred, pos_integer=fail_idx)
        out[value] = {
            "n": int(len(items)),
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "fnr": m["fnr"], "fpr": m["fpr"], "accuracy": m["accuracy"],
        }
    return out


def _degradation(base: dict, perturbed: dict) -> dict:
    """Signed deltas (perturbed - baseline); for FNR/FPR positive = worse."""
    keys = ["accuracy", "precision", "recall", "f1", "auroc"]
    delta = {k: round(float(perturbed[k]) - float(base[k]), 4)
             for k in keys if perturbed.get(k) is not None and base.get(k) is not None}
    keys_r = ["fnr", "fpr"]  # higher = worse
    for k in keys_r:
        if perturbed.get(k) is not None and base.get(k) is not None:
            delta[f"{k}_delta"] = round(float(perturbed[k]) - float(base[k]), 4)
    return delta


def _print_report(report: dict) -> None:
    base = report["baseline"]["metrics"]
    print("=" * 78)
    print(f"ROBUSTNESS EVALUATION  model={report['model']}  split={report['split']}")
    print(f"checkpoint={report['checkpoint']}")
    print("=" * 78)
    print(f"n={report['n_samples']}  recordings={report['n_recordings']}  seed={report['seed']}")
    bl = base
    print(f"\nBASELINE (identity): acc={bl['accuracy']} prec={bl['precision']} "
          f"rec={bl['recall']} f1={bl['f1']} fnr={bl['fnr']} fpr={bl['fpr']} "
          f"auroc={bl['auroc']}")

    print("\n--- Degradation by perturbation/severity (failing-as-positive) ---")
    hdr = f"{'perturb':12s} {'sev':9s} {'acc':>6s} {'prec':>6s} {'rec':>6s} {'f1':>6s} {'auroc':>6s} {'fnr':>6s} {'fpr':>6s} {'dF1':>6s} {'dFNR':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for pname in report["order"]:
        for sev, res in zip(SEVERITY_NAMES, report["perturbations"][pname]["results"]):
            m = res["metrics"]
            d = res["degradation"]
            print(f"{pname:12s} {sev:9s} {m['accuracy']:6.3f} {m['precision']:6.3f} "
                  f"{m['recall']:6.3f} {m['f1']:6.3f} {str(m['auroc']):>6s} "
                  f"{m['fnr']:6.3f} {m['fpr']:6.3f} {d.get('f1','-'):>6} {d.get('fnr_delta','-'):>6}")

    print("\n--- Worst per-recording failing-recall for each perturbation (severe) ---")
    rec_names = list(report["baseline"]["metrics"]["recordings"].keys())
    for pname in report["order"]:
        sev_res = report["perturbations"][pname]["results"][2]
        recs = sev_res["metrics"]["recordings"]
        worst = sorted(recs.items(), key=lambda kv: kv[1].get("recall_failing", 1.0))[:3]
        worst_s = ", ".join(f"{k}({v.get('recall_failing','-')})" for k, v in worst)
        print(f"  {pname:12s} severe worst: {worst_s}")

    print("\n--- Failure-type failing-recall degradation, severe (vs baseline) ---")
    base_ft = base["by_failure_type"]
    for pname in report["order"]:
        sev_res = report["perturbations"][pname]["results"][2]
        ft = sev_res["metrics"]["by_failure_type"]
        if not ft:
            continue
        parts = []
        for k in sorted(set(base_ft) | set(ft)):
            b = base_ft.get(k, {}).get("recall")
            v = ft.get(k, {}).get("recall")
            if b is None or v is None:
                continue
            delta = round(v - b, 3)
            parts.append(f"{k}:{v:.3f}({delta:+})")
        print(f"  {pname:12s} " + "; ".join(parts))

    print("\n--- Worst single object failing-recall, severe ---")
    base_sh = base["by_shape"]
    for pname in report["order"]:
        sev_res = report["perturbations"][pname]["results"][2]
        sh = sev_res["metrics"]["by_shape"]
        if not sh:
            continue
        merged = {**base_sh, **sh}
        worst = sorted(
            sh.items(),
            key=lambda kv: (kv[1].get("recall", 1.0), -kv[1].get("n", 0)),
        )[:2]
        parts = [f"{k}:rec={v['recall']}(n={v['n']})" for k, v in worst]
        print(f"  {pname:12s} " + "; ".join(parts))


def evaluate_robustness(
    config: ExperimentConfig,
    checkpoint_path=None,
    base_dir="experiments",
    split="test",
    output_json=None,
    perturbations: list[str] | None = None,
    seed: int = 0,
    metadata_csv: str | None = None,
) -> dict:
    """Run the robustness evaluation on a frozen model.

    Returns a report dict with a baseline and per-perturbation results.
    """
    import torch
    from .models.factory import build_classifier
    from .data.loaders import _build_transforms

    ckpt_path = _resolve_checkpoint(config, checkpoint_path, base_dir)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = _get_device()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 2)
    model_name = ckpt.get("config_model", config.model)
    class_to_idx = ckpt.get("class_to_idx", {})
    fail_idx = None
    for k, v in class_to_idx.items():
        if k.lower() == "failing" or v == 0:
            fail_idx = int(v)
            break
    if fail_idx is None:
        fail_idx = 0
    class_names = {int(v): k for k, v in class_to_idx.items()}

    model = build_classifier(model_name, num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    resolved = load_resolved_metadata(config.dataset.data_dir, split)
    transform = _build_transforms(config.dataset.image_size, {}, is_train=False)
    batch_size = max(1, int(config.data.batch_size))

    order = perturbations or list(PERTURBATIONS.keys())

    def _run(perturb_fn):
        return _infer_with_perturbation(
            model, device, resolved, transform, perturb_fn,
            fail_idx, batch_size,
        )

    # Baseline (identity)
    base_samples = _run(None)
    base_metrics = _compute_all(base_samples, fail_idx, class_names)

    report = {
        "model": config.model,
        "experiment_id": ckpt.get("experiment_id", config.experiment_id),
        "split": split,
        "checkpoint": str(ckpt_path),
        "class_to_idx": class_to_idx,
        "failing_class_idx": fail_idx,
        "n_samples": len(base_samples),
        "n_recordings": len({s["recording"] for s in base_samples}),
        "seed": seed,
        "threshold": config.evaluation.decision_threshold,
        "order": order,
        "baseline": {"metrics": base_metrics},
        "perturbations": {},
    }

    rng = np.random.default_rng(seed)
    for pname in order:
        spec = PERTURBATIONS[pname]
        levels = spec["levels"]
        results = []
        for level in levels:
            apply_fn = lambda _img, _lvl=level: spec["apply"](_img, _lvl, rng)
            samples = _run(apply_fn)
            metrics = _compute_all(samples, fail_idx, class_names)
            results.append({
                "level": level,
                "metrics": metrics,
                "degradation": _degradation(base_metrics, metrics),
            })
        report["perturbations"][pname] = {
            "levels": levels,
            "results": results,
        }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return report


print_report = _print_report

