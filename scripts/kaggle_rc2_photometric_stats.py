#!/usr/bin/env python
"""Fit per-channel photometric statistics on the **training** split only.

Uses the exact deterministic preprocessing that training/validation/inference
share (``Resize((image_size, image_size))`` + ``ToTensor``) so the resulting
mean/std are directly usable as ``normalize_mean``/``normalize_std``. Never
touches validation or held-out frames.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from tv.data.loaders import _build_transforms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True,
                    help="Run dir containing splits/train.csv and splits/splits.json")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    splits = json.loads((data_dir / "splits" / "splits.json").read_text())
    root = Path(splits["data_dir"])
    tfm = _build_transforms(args.image_size, augmentation={"normalize": False},
                            is_train=False)

    from PIL import Image

    with (data_dir / "splits" / "train.csv").open() as f:
        rows = list(csv.DictReader(f))

    n = 0
    total = np.zeros(3, dtype=np.float64)
    total_sq = np.zeros(3, dtype=np.float64)
    for r in rows:
        p = Path(r["path"])
        if not p.is_absolute():
            p = root / p
        t = tfm(Image.open(p).convert("RGB")).numpy().astype(np.float64)  # (3,H,W)
        flat = t.reshape(3, -1)
        total += flat.sum(axis=1)
        total_sq += (flat ** 2).sum(axis=1)
        n += flat.shape[1]

    mean = total / n
    var = total_sq / n - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))
    result = {
        "split": "train",
        "n_images": len(rows),
        "n_pixels_per_channel": int(n),
        "image_size": args.image_size,
        "mean": [round(float(x), 6) for x in mean],
        "std": [round(float(x), 6) for x in std],
        "data_dir": str(data_dir),
    }
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
