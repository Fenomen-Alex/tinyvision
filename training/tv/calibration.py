"""Simple, validation-only score calibration (temperature scaling).

Temperature scaling rescales the logits by one scalar ``T > 0``::

    p(T) = softmax(z / T)

For every ``T > 0`` this is a strictly monotonic transform of the logit
difference, so it **preserves the ranking and the argmax** of every sample.  It
therefore cannot change which class a sample is assigned at a fixed 0.5
boundary; it only rescales the reported confidences.  Any decision improvement
obtained from calibration is mathematically equivalent to choosing a new
threshold on the uncalibrated scores.

This matters for the rc2 real-domain diagnostic: the held-out failing stream's
frames are ranked above the normal frames (AUROC ~0.97) but every one of them
has ``argmax == normal``.  No positive temperature can move them across 0.5, so
calibration cannot rescue a fixed-0.5 detector there.

Calibration parameters are fit on validation data only (see
``docs/research/rc2-realdomain-results.md``); held-out data is never used.
"""
from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def argmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return softmax(logits, temperature).argmax(axis=1)


def negative_log_likelihood(logits: np.ndarray, y: np.ndarray,
                            temperature: float = 1.0) -> float:
    p = softmax(logits, temperature)
    y = np.asarray(y).astype(int)
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    grid: np.ndarray | None = None) -> dict:
    """Fit a single temperature on validation logits by minimising NLL.

    Ties are broken toward T closer to 1.0 for stability.  Returns the fitted
    temperature, the achieved NLL, and the uncalibrated (T=1) NLL.
    """
    logits = np.asarray(logits, dtype=float)
    y = np.asarray(y).astype(int)
    if grid is None:
        grid = np.linspace(0.1, 10.0, 199)
    nll_t1 = negative_log_likelihood(logits, y, 1.0)
    best_t, best = 1.0, nll_t1
    for t in grid:
        v = negative_log_likelihood(logits, y, float(t))
        if v < best - 1e-12 or (abs(v - best) <= 1e-12 and abs(t - 1.0) < abs(best_t - 1.0)):
            best, best_t = v, float(t)
    return {"temperature": best_t, "nll": best, "nll_t1": nll_t1}


def expected_calibration_error(prob_positive: np.ndarray, y_true: np.ndarray,
                               positive: int = 0, bins: int = 10) -> float:
    """ECE for a binary detector, using P(positive) and a 0.5 decision rule."""
    p = np.asarray(prob_positive, dtype=float)
    y = np.asarray(y_true).astype(int)
    pred = np.where(p >= 0.5, positive, 1 - positive)
    conf = np.where(p >= 0.5, p, 1 - p)
    acc = (pred == y).astype(float)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1.0:
            mask = (conf >= lo) & (conf <= 1.0)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)
