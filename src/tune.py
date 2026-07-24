"""
Phase 6 — hyperparameter tuning for the three models Phase 5 flagged as
promising (Random Forest, Gradient Boosting, Logistic Regression).

Decision Tree and XGBoost are not tuned here: Phase 5 found Decision Tree
worst on every real metric (kept only to illustrate overfitting), and
XGBoost showed no advantage over Gradient Boosting on this dataset.

Tuning uses RandomizedSearchCV with GroupKFold(5) grouped by Patient File
No. - the same grouping used throughout this project, so a search can never
reward a hyperparameter combination for memorizing one patient's augmented
copies across the train/validation split of a fold.

Scoring metric: average_precision (PR-AUC). Chosen over accuracy/ROC-AUC
because Phase 5 found PR-AUC the more honest metric under this dataset's
31%-prevalence imbalance, and over a single-threshold metric (F1, recall)
because tuning should not commit to one decision threshold - that choice
belongs to Phase 10 (risk-category thresholds), not to the search here.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from src.config import RANDOM_SEED
from src.preprocessing import build_preprocessing_pipeline

N_CV_SPLITS = 5
N_ITER = 40
SCORING = "average_precision"

PARAM_DISTRIBUTIONS = {
    "logistic_regression": {
        "model__C": np.logspace(-3, 2, 100),
        "model__penalty": ["l1", "l2"],
    },
    "random_forest": {
        "model__n_estimators": randint(150, 600),
        "model__max_depth": randint(3, 20),
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "model__max_features": ["sqrt", "log2", 0.3, 0.5],
    },
    "gradient_boosting": {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(2, 6),
        "model__learning_rate": uniform(0.01, 0.29),
        "model__min_samples_leaf": randint(1, 10),
        "model__subsample": uniform(0.6, 0.4),
    },
}


def _base_estimator(model_key: str):
    if model_key == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000, random_state=RANDOM_SEED
        )
    if model_key == "random_forest":
        return RandomForestClassifier(class_weight="balanced", random_state=RANDOM_SEED)
    if model_key == "gradient_boosting":
        return GradientBoostingClassifier(random_state=RANDOM_SEED)
    raise ValueError(f"Unknown model_key: {model_key}")


def tune_model(
    model_key: str,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_iter: int = N_ITER,
    n_splits: int = N_CV_SPLITS,
) -> RandomizedSearchCV:
    """Run a grouped RandomizedSearchCV for one model. Returns the fitted search object
    (`.best_estimator_` is the full preprocessing+model Pipeline refit on all of X, y).
    """
    pipeline = Pipeline([
        ("preprocess", build_preprocessing_pipeline()),
        ("model", _base_estimator(model_key)),
    ])

    fit_params = {}
    if model_key == "gradient_boosting":
        # GradientBoostingClassifier has no class_weight param - approximate
        # balanced weighting with sample_weight computed on the full training
        # set passed in; RandomizedSearchCV subsets this array consistently
        # with X/y for every inner fold.
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS[model_key],
        n_iter=n_iter,
        scoring=SCORING,
        cv=GroupKFold(n_splits=n_splits),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups, **fit_params)
    return search


def tune_all(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_iter: int = N_ITER
) -> Dict[str, RandomizedSearchCV]:
    """Tune all three candidate models. Returns {model_key: fitted RandomizedSearchCV}."""
    return {
        model_key: tune_model(model_key, X, y, groups, n_iter=n_iter)
        for model_key in PARAM_DISTRIBUTIONS
    }


def tuned_estimator_factory(model_key: str, best_params: Dict):
    """Build a zero-arg factory returning a fresh estimator with the winning
    hyperparameters set - for re-running Phase 5's grouped-CV metrics suite
    (confusion matrix / ROC / PR / calibration) on the tuned model, the same
    way it was run on the untuned default in Phase 5, for a fair before/after
    comparison.
    """
    stripped_params = {k.replace("model__", "", 1): v for k, v in best_params.items()}

    def factory():
        estimator = _base_estimator(model_key)
        estimator.set_params(**stripped_params)
        return estimator

    return factory
