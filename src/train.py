"""
Phase 4 — model registry and a leakage-safe, grouped cross-validation runner.

Every model is wrapped as `Pipeline([("preprocess", ...), ("model", ...)])` so
the preprocessing pipeline (imputer medians, scaler mean/std, one-hot
categories) is re-fit from scratch on each fold's training rows only, then
just applied to that fold's validation rows — never the other way around.

Cross-validation is grouped by Patient File No. (GroupKFold), not plain
KFold: train_pool contains multiple augmented-copy rows per patient (Phase 1
finding), so an ungrouped split would let copies of the same patient land in
both the training and validation side of a fold, inflating every score.
"""

from typing import Callable, Dict

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import RANDOM_SEED
from src.preprocessing import build_preprocessing_pipeline

N_CV_SPLITS = 5

# Each entry: (display_name, estimator_factory, needs_manual_sample_weight)
# needs_manual_sample_weight=True means the estimator has no class_weight
# param, so we compute balanced sample weights per training fold ourselves
# and pass them as a fit param, instead of relying on class_weight="balanced".
MODEL_SPECS = {
    "dummy_baseline": (
        lambda: DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED),
        False,
    ),
    "logistic_regression": (
        lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_SEED),
        False,
    ),
    "decision_tree": (
        lambda: DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_SEED),
        False,
    ),
    "random_forest": (
        lambda: RandomForestClassifier(class_weight="balanced", n_estimators=300, random_state=RANDOM_SEED),
        False,
    ),
    "gradient_boosting": (
        lambda: GradientBoostingClassifier(random_state=RANDOM_SEED),
        True,  # sklearn's GradientBoostingClassifier has no class_weight param
    ),
    "xgboost": (
        lambda: XGBClassifier(random_state=RANDOM_SEED, eval_metric="logloss"),
        True,  # handled via scale_pos_weight instead of sample_weight, see below
    ),
}


def build_model_pipeline(estimator) -> Pipeline:
    """Wrap one classifier with a *fresh, unfit* copy of the Phase 3 preprocessing pipeline."""
    return Pipeline([
        ("preprocess", build_preprocessing_pipeline()),
        ("model", estimator),
    ])


def _score_split(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    proba = pipe.predict_proba(X)[:, 1]
    pred = pipe.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def run_grouped_cv(
    model_name: str,
    estimator_factory: Callable,
    needs_manual_weighting: bool,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = N_CV_SPLITS,
) -> pd.DataFrame:
    """Run grouped k-fold CV for one model, scoring both the training fold and
    the held-out validation fold each time (so train-vs-val gap is visible).

    Returns one row per (fold, split) with all 5 metrics - the long format
    the caller aggregates into a comparison table.
    """
    cv = GroupKFold(n_splits=n_splits)
    records = []
    oof_records = []

    for fold_i, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        estimator = estimator_factory()
        fit_kwargs = {}
        if needs_manual_weighting:
            if isinstance(estimator, XGBClassifier):
                n_pos = (y_tr == 1).sum()
                n_neg = (y_tr == 0).sum()
                estimator.set_params(scale_pos_weight=n_neg / max(n_pos, 1))
            else:
                fit_kwargs["model__sample_weight"] = compute_sample_weight("balanced", y_tr)

        pipe = build_model_pipeline(estimator)
        pipe.fit(X_tr, y_tr, **fit_kwargs)

        for split_name, X_s, y_s in [("train", X_tr, y_tr), ("val", X_val, y_val)]:
            metrics = _score_split(pipe, X_s, y_s)
            metrics.update({"model": model_name, "fold": fold_i, "split": split_name})
            records.append(metrics)

        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_pred = pipe.predict(X_val)
        oof_records.append(pd.DataFrame({
            "row_index": X_val.index,
            "patient_id": groups.loc[X_val.index].values,
            "model": model_name,
            "fold": fold_i,
            "y_true": y_val.values,
            "y_pred": val_pred,
            "y_proba": val_proba,
        }))

    return pd.DataFrame(records), pd.concat(oof_records, ignore_index=True)


def run_all_models_cv(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = N_CV_SPLITS
) -> pd.DataFrame:
    """Run grouped CV for every model in MODEL_SPECS, concatenated into one long DataFrame.

    Convenience wrapper for Phase 4 (metrics only). Use `run_all_models_cv_with_oof`
    when the out-of-fold predictions themselves are also needed (Phase 5: confusion
    matrices, ROC/PR/calibration curves).
    """
    metrics_df, _ = run_all_models_cv_with_oof(X, y, groups, n_splits)
    return metrics_df


def run_all_models_cv_with_oof(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = N_CV_SPLITS
):
    """Run grouped CV for every model, returning (metrics_long_df, oof_predictions_df).

    oof_predictions_df has exactly one row per (model, training-partition row):
    every row of train_pool gets predicted exactly once, by the fold where it
    was held out - so these predictions are leakage-free without touching
    holdout_validation, and can be used for confusion matrices / ROC / PR /
    calibration curves per model.
    """
    all_metrics = []
    all_oof = []
    for name, (factory, needs_weighting) in MODEL_SPECS.items():
        metrics, oof = run_grouped_cv(name, factory, needs_weighting, X, y, groups, n_splits)
        all_metrics.append(metrics)
        all_oof.append(oof)
    return pd.concat(all_metrics, ignore_index=True), pd.concat(all_oof, ignore_index=True)


def summarize_cv_results(long_results: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long (model, fold, split) results into mean+/-std per (model, split)."""
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    summary = (
        long_results.groupby(["model", "split"])[metric_cols]
        .agg(["mean", "std"])
    )
    return summary
