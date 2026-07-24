"""
Phase 7 — held-out patient validation and dataset-shift checks.

Terminology used throughout this module and its notebook: "holdout
validation", never "external validation". Phase 1 established that
`holdout_validation` (109 patients) comes from the same source study as
`train_pool` (432 patients, plus their augmented copies) - a patient-level
split of one population, not two independent cohorts. This still answers a
real question (does the frozen model work on patients it never saw in any
form?) but it is not evidence of generalization to a different population,
clinic, or measurement process, and must never be described as if it were.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.evaluate import compute_full_metrics, bootstrap_all_metrics_ci, METRIC_FNS


def score_holdout(pipeline, X_holdout: pd.DataFrame, y_holdout: pd.Series, patient_ids: pd.Series) -> pd.DataFrame:
    """Score the frozen pipeline on holdout_validation ONCE (.predict / .predict_proba
    only - the pipeline is never re-fit here). Returns a one-model OOF-shaped
    DataFrame compatible with every src.evaluate helper (get_confusion_matrix,
    get_roc_curve, etc.), keyed by Patient File No. for the bootstrap.

    `patient_ids` must be passed separately since `split_features_target`
    already drops the ID columns from X_holdout.
    """
    proba = pipeline.predict_proba(X_holdout)[:, 1]
    pred = pipeline.predict(X_holdout)
    return pd.DataFrame({
        "patient_id": patient_ids.values,
        "y_true": y_holdout.values,
        "y_pred": pred,
        "y_proba": proba,
    })


def holdout_metrics_with_ci(holdout_scored: pd.DataFrame, n_boot: int = 1000) -> Dict[str, tuple]:
    """Patient-level bootstrap CI for every metric in METRIC_FNS, on the holdout set.

    Each holdout patient contributes exactly one row, so this bootstrap is
    equivalent to a plain row-level bootstrap here - using the same function
    as Phase 5/6 keeps the method identical across the whole project.
    """
    point, lo, hi, _ = bootstrap_all_metrics_ci(holdout_scored, METRIC_FNS, n_boot=n_boot)
    return {name: (point[name], lo[name], hi[name]) for name in METRIC_FNS}


def build_comparison_table(cv_metrics_row: pd.Series, holdout_metrics: Dict[str, tuple]) -> pd.DataFrame:
    """The Metric | train_pool CV | holdout_validation | Difference table."""
    rows = []
    for metric in METRIC_FNS:
        cv_value = cv_metrics_row[metric]
        holdout_value, lo, hi = holdout_metrics[metric]
        rows.append({
            "metric": metric,
            "train_pool_cv": cv_value,
            "holdout_validation": holdout_value,
            "holdout_ci_low": lo,
            "holdout_ci_high": hi,
            "difference": holdout_value - cv_value,
        })
    return pd.DataFrame(rows).set_index("metric")


def compare_numeric_distributions(train_df: pd.DataFrame, holdout_df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """Kolmogorov-Smirnov test per numeric column: is the holdout distribution
    different from the train-partition distribution? Both inputs should be
    ONE ROW PER PATIENT (not the augmented train_pool) for a fair comparison.
    """
    rows = []
    for col in numeric_cols:
        train_vals = train_df[col].dropna()
        holdout_vals = holdout_df[col].dropna()
        stat, p_value = ks_2samp(train_vals, holdout_vals)
        rows.append({
            "feature": col,
            "train_mean": train_vals.mean(),
            "holdout_mean": holdout_vals.mean(),
            "ks_statistic": stat,
            "p_value": p_value,
        })
    return pd.DataFrame(rows).sort_values("p_value")


def compare_categorical_distributions(train_df: pd.DataFrame, holdout_df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Chi-square test per categorical/binary column: same idea as the KS test, for
    non-numeric features. Both inputs should be one row per patient.
    """
    rows = []
    for col in cat_cols:
        train_counts = train_df[col].value_counts()
        holdout_counts = holdout_df[col].value_counts()
        categories = sorted(set(train_counts.index) | set(holdout_counts.index))
        contingency = pd.DataFrame({
            "train": [train_counts.get(c, 0) for c in categories],
            "holdout": [holdout_counts.get(c, 0) for c in categories],
        })
        chi2, p_value, _, _ = chi2_contingency(contingency.T)
        rows.append({"feature": col, "chi2": chi2, "p_value": p_value})
    return pd.DataFrame(rows).sort_values("p_value")
