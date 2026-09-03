"""Metrics computation for classification and detection evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ClassificationMetrics:
    precision: float
    recall: float
    f1: float
    accuracy: float
    fnr: float  # false negative rate
    fpr: float  # false positive rate
    auroc: float | None
    confusion_matrix: list[list[int]]
    sample_count: int
    classes: list[int]

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "fnr": self.fnr,
            "fpr": self.fpr,
            "auroc": self.auroc,
            "confusion_matrix": self.confusion_matrix,
            "sample_count": self.sample_count,
            "classes": self.classes,
        }

    def summary(self) -> str:
        lines = [
            f"Samples:     {self.sample_count}",
            f"Precision:   {self.precision:.4f}",
            f"Recall:      {self.recall:.4f}",
            f"F1:          {self.f1:.4f}",
            f"Accuracy:    {self.accuracy:.4f}",
            f"FNR:         {self.fnr:.4f}",
            f"FPR:         {self.fpr:.4f}",
        ]
        if self.auroc is not None:
            lines.append(f"AUROC:       {self.auroc:.4f}")
        lines.append(f"Confusion:   {self.confusion_matrix}")
        return "\n".join(lines)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute classification metrics from ground truth and predictions.

    Binary (two-class) evaluation treats class 1 as the positive class for
    precision/recall/F1 and the FNR/FPR rates. For three or more classes the
    precision/recall/F1 are macro-averaged, and FNR/FPR are the macro-average
    of each class's one-vs-rest rate.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_scores: Predicted probability scores for the positive class
            (binary) or per-class probabilities of shape (n, n_classes)
            (multiclass). Required for AUROC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = sorted(set(int(c) for c in np.unique(np.concatenate([y_true, y_pred]))))
    n_classes = len(classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes).tolist()

    accuracy = float(accuracy_score(y_true, y_pred))

    # --- AUROC: needs scores; handle single-class and shape edge cases ---
    auroc: float | None = None
    if y_scores is not None and n_classes > 1:
        try:
            if n_classes == 2:
                scores = np.asarray(y_scores)
                # scores may be the positive-class column vector
                if scores.ndim == 2 and scores.shape[1] > 1:
                    scores = scores[:, 1]
                scores = np.asarray(scores, dtype=float).ravel()
                auroc = float(roc_auc_score(y_true, scores))
            else:
                probs = np.asarray(y_scores, dtype=float)
                if probs.ndim == 1:
                    # only per-sample max scores provided; cannot compute AUROC
                    auroc = None
                else:
                    auroc = float(
                        roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
                    )
        except ValueError:
            auroc = None

    # --- precision/recall/F1 ---
    if n_classes <= 2:
        # Binary path: always evaluate as binary classification (labels=[0,1])
        # even when only one class is present in this particular test split.
        precision = float(
            precision_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
        )
        recall = float(
            recall_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
        )
        f1 = float(
            f1_score(y_true, y_pred, labels=[0, 1], pos_label=1, zero_division=0)
        )
        # Binary confusion matrix fixed to labels=[0,1]: [TN FP; FN TP]
        cm2 = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm2[0][0]), int(cm2[0][1]), int(cm2[1][0]), int(cm2[1][1])
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    else:
        precision = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        # Macro per-class FNR / FPR (one-vs-rest)
        c = np.array(cm)
        fnr_rates, fpr_rates = [], []
        for i in range(n_classes):
            tp = int(c[i, i])
            fn = int(c[i, :].sum() - tp)
            fp = int(c[:, i].sum() - tp)
            tn = int(c.sum() - tp - fn - fp)
            fnr_rates.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)
            fpr_rates.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr = float(np.mean(fnr_rates))
        fpr = float(np.mean(fpr_rates))

    total = int(np.sum(cm))
    return ClassificationMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        fnr=fnr,
        fpr=fpr,
        auroc=auroc,
        confusion_matrix=cm,
        sample_count=total,
        classes=classes,
    )
