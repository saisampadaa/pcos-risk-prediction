"""
Phase 5 — full evaluation metric suite, confidence intervals, and comparison plots.

Everything here operates on out-of-fold (OOF) predictions from Phase 4's
grouped cross-validation (`src.train.run_all_models_cv_with_oof`): every row
of `train_pool` gets exactly one prediction, always from a fold that did not
train on it. This gives leakage-free confusion matrices / ROC / PR /
calibration curves without touching `holdout_validation`, which stays frozen
until a final model is chosen.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score, roc_curve,
)

from src.config import RANDOM_SEED


def specificity_score(y_true, y_pred) -> float:
    """True-negative rate: TN / (TN + FP). Not in sklearn.metrics directly."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def compute_full_metrics(y_true, y_pred, y_proba) -> Dict[str, float]:
    """The full Phase 5 metric set for one set of predictions."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_proba),
    }


def bootstrap_metric_ci(
    oof_model_df: pd.DataFrame,
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Patient-level block bootstrap confidence interval for one metric.

    Resamples unique `patient_id`s WITH replacement (not individual rows) so
    that a patient's multiple augmented-copy rows are always resampled
    together - consistent with treating the patient, not the row, as the
    unit of independence everywhere else in this project.

    `metric_fn(y_true, y_pred, y_proba) -> float`. Returns (point_estimate, ci_low, ci_high).
    For multiple metrics on the same model, prefer `bootstrap_all_metrics_ci`
    - it resamples once per iteration and reuses that resample for every
    metric, instead of re-resampling from scratch for each one.
    """
    point, ci_low, ci_high, _ = bootstrap_all_metrics_ci(
        oof_model_df, {"metric": metric_fn}, n_boot=n_boot, seed=seed, ci=ci
    )
    return point["metric"], ci_low["metric"], ci_high["metric"]


def bootstrap_all_metrics_ci(
    oof_model_df: pd.DataFrame,
    metric_fns: Dict[str, Callable],
    n_boot: int = 1000,
    seed: int = RANDOM_SEED,
    ci: float = 0.95,
):
    """Same patient-level block bootstrap as `bootstrap_metric_ci`, but computes
    every metric in `metric_fns` from the SAME set of resamples (one resample
    per iteration, not one per metric) - n_boot resamples total instead of
    n_boot * len(metric_fns).

    Returns (point_estimates, ci_low, ci_high, raw_boot_scores), each a dict
    keyed by metric name (raw_boot_scores maps to a list of n_boot values).
    """
    rng = np.random.default_rng(seed)
    unique_patients = oof_model_df["patient_id"].unique()
    n_patients = len(unique_patients)
    grouped = {pid: g for pid, g in oof_model_df.groupby("patient_id")}

    point_estimates = {
        name: fn(oof_model_df["y_true"], oof_model_df["y_pred"], oof_model_df["y_proba"])
        for name, fn in metric_fns.items()
    }

    boot_scores = {name: [] for name in metric_fns}
    for _ in range(n_boot):
        sampled_ids = rng.choice(unique_patients, size=n_patients, replace=True)
        resampled = pd.concat([grouped[pid] for pid in sampled_ids], ignore_index=True)
        yt, yp, ypr = resampled["y_true"], resampled["y_pred"], resampled["y_proba"]
        for name, fn in metric_fns.items():
            try:
                boot_scores[name].append(fn(yt, yp, ypr))
            except ValueError:
                pass  # a resample with only one class present for this metric; skip

    alpha = (1 - ci) / 2
    ci_low, ci_high = {}, {}
    for name, scores in boot_scores.items():
        ci_low[name], ci_high[name] = np.quantile(scores, [alpha, 1 - alpha])

    return point_estimates, ci_low, ci_high, boot_scores


# metric_fn wrappers matching the (y_true, y_pred, y_proba) -> float signature
METRIC_FNS = {
    "accuracy": lambda yt, yp, ypr: accuracy_score(yt, yp),
    "precision": lambda yt, yp, ypr: precision_score(yt, yp, zero_division=0),
    "recall_sensitivity": lambda yt, yp, ypr: recall_score(yt, yp, zero_division=0),
    "specificity": lambda yt, yp, ypr: specificity_score(yt, yp),
    "f1": lambda yt, yp, ypr: f1_score(yt, yp, zero_division=0),
    "roc_auc": lambda yt, yp, ypr: roc_auc_score(yt, ypr),
    "pr_auc": lambda yt, yp, ypr: average_precision_score(yt, ypr),
    "balanced_accuracy": lambda yt, yp, ypr: balanced_accuracy_score(yt, yp),
    "brier_score": lambda yt, yp, ypr: brier_score_loss(yt, ypr),
}


def build_metrics_table_with_ci(oof_df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """One row per model: point estimate + 95% CI for every metric in METRIC_FNS.

    Uses `bootstrap_all_metrics_ci` so each model gets exactly n_boot resamples
    total (shared across all 9 metrics), not n_boot per metric.
    """
    rows = []
    for model_name, group in oof_df.groupby("model"):
        point, lo, hi, _ = bootstrap_all_metrics_ci(group, METRIC_FNS, n_boot=n_boot)
        row = {"model": model_name}
        for metric_name in METRIC_FNS:
            row[metric_name] = point[metric_name]
            row[f"{metric_name}_ci_low"] = lo[metric_name]
            row[f"{metric_name}_ci_high"] = hi[metric_name]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def get_confusion_matrix(oof_model_df: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(oof_model_df["y_true"], oof_model_df["y_pred"], labels=[0, 1])


def get_roc_curve(oof_model_df: pd.DataFrame):
    return roc_curve(oof_model_df["y_true"], oof_model_df["y_proba"])


def get_pr_curve(oof_model_df: pd.DataFrame):
    return precision_recall_curve(oof_model_df["y_true"], oof_model_df["y_proba"])


def get_calibration_curve(oof_model_df: pd.DataFrame, n_bins: int = 10):
    return calibration_curve(oof_model_df["y_true"], oof_model_df["y_proba"], n_bins=n_bins, strategy="quantile")
