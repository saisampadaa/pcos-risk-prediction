"""
Reproducible preprocessing pipeline (Phase 3).

Everything here is a scikit-learn Transformer, so the whole chain can be
fit ONCE on train_pool and reused (via .transform, never re-fit) on
holdout_validation and on any new patient submitted through the API later.
No step here ever looks at holdout_validation's values to make a decision.

Column-name and dirty-value fixes from Phase 1 (src/feature_mapping.py) are
assumed already applied by the data_splitting step that produced train_pool
/ holdout_validation. This module picks up from there: it corrects
physiologically implausible values found during Phase 2 EDA, enforces BMI
self-consistency, imputes, encodes, and scales.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import PATIENT_ID_COL, TARGET_COL

ID_COLUMNS = ["Sl. No", PATIENT_ID_COL]

# Thresholds are the ones found and justified in notebooks/02_eda.ipynb
# ("Data-entry errors and outliers found in this EDA"). Values outside these
# bounds are treated as missing, not deleted - the rest of that patient's
# record is otherwise unaffected.
PLAUSIBLE_RANGES = {
    "Vit D3 (ng/mL)": (0, 200),
    "Pulse rate(bpm) ": (40, 200),
    "FSH(mIU/mL)": (0, 100),
    "LH(mIU/mL)": (0, 100),
    "BP _Systolic (mmHg)": (50, 250),
    "BP _Diastolic (mmHg)": (30, 150),
}
VALID_CYCLE_CODES = {2, 4}  # 2 = Regular, 4 = Irregular, per the Instructions codebook

BINARY_COLUMNS = [
    "Pregnant(Y/N)", "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)",
    "Hair loss(Y/N)", "Pimples(Y/N)", "Fast food (Y/N)", "Reg.Exercise(Y/N)",
]
NOMINAL_CATEGORICAL_COLUMNS = ["Blood Group", "Cycle(R/I)"]

NUMERIC_COLUMNS = [
    " Age (yrs)", "Weight (Kg)", "Height(Cm) ", "BMI", "RR (breaths/min)", "Hb(g/dl)",
    "Cycle length(days)", "Marraige Status (Yrs)", "No. of abortions",
    "  I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)", "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH",
    "Hip(inch)", "Waist(inch)", "Waist:Hip Ratio", "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)",
    "Vit D3 (ng/mL)", "PRG(ng/mL)", "RBS(mg/dl)", "Pulse rate(bpm) ",
    "BP _Systolic (mmHg)", "BP _Diastolic (mmHg)", "Follicle No. (L)", "Follicle No. (R)",
    "Avg. F size (L) (mm)", "Avg. F size (R) (mm)", "Endometrium (mm)",
]


class ClipStandardized(BaseEstimator, TransformerMixin):
    """Clip standardized (post-StandardScaler) values to +/- `bound` standard
    deviations, per numeric column.

    Why this exists: Phase 8's SHAP analysis found a holdout patient with
    PRG(ng/mL) = 85.0, while train_pool's max is 25.3 - a z-score of ~49
    once scaled. That single value's contribution to the linear model's
    prediction (coefficient x z-score) dwarfed every other feature combined
    (SHAP impact -9.8 vs everything else under 1.0), for a feature that
    wasn't even the intended top predictor. StandardScaler has no concept of
    "too far outside the training distribution" - it will happily produce a
    z-score of 49 for any wildly out-of-range input, real or a data-entry
    error, and a linear model has no mechanism to discount it. Clipping to
    +/-5 (a generous bound - only engages for genuinely extreme values,
    ~99.99994% of a normal distribution falls within it) keeps any single
    unusual input from dominating a prediction, without meaningfully
    affecting ordinary in-range values.

    This complements, not replaces, InvalidValueCorrector: that step catches
    SPECIFIC, individually-verified data-entry errors (Vit D3, pulse rate,
    FSH, blood pressure - found by manually inspecting Phase 2's EDA). This
    step is the general safety net for any OTHER numeric column with an
    extreme value that wasn't individually checked - including, as found
    here, ones nobody thought to check by hand.
    """

    def __init__(self, bound: float = 5.0):
        self.bound = bound

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.clip(lower=-self.bound, upper=self.bound)
        return np.clip(X, -self.bound, self.bound)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return self.feature_names_in_


class InvalidValueCorrector(BaseEstimator, TransformerMixin):
    """Replace physiologically implausible values with NaN, and recompute BMI.

    Two fixes, both discovered in Phase 2 EDA (notebooks/02_eda.ipynb):

    1. A handful of columns have data-entry-error values (e.g. Vit D3 = 6014.66
       ng/mL, pulse rate = 13 bpm) - out-of-range values are set to NaN so the
       imputer handles them like any other missing value.
    2. BMI is recomputed from Height/Weight for every row, rather than trusting
       the stored column. Reason: comparing stored vs. recomputed BMI showed
       62.9% of the augmented `lifestyle.csv` rows disagree by >0.5 (up to
       3.34 kg/m²) - the augmentation jittered Weight, Height, and BMI as three
       independent draws instead of deriving BMI from the jittered
       Height/Weight, leaving them internally inconsistent. The original
       clinical_2 rows barely change (only 0.9% differ >0.5, ordinary
       rounding). Recomputing is a strict improvement in both cases since BMI
       is a deterministic function of height and weight, not an independent
       measurement.

    Stateless (no fitting from data), so it is safe to apply identically to
    train_pool, holdout_validation, and any single new patient at inference
    time - it never uses training-set statistics.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col, (low, high) in PLAUSIBLE_RANGES.items():
            if col in X.columns:
                out_of_range = (X[col] < low) | (X[col] > high)
                X.loc[out_of_range, col] = np.nan

        if "Cycle(R/I)" in X.columns:
            X.loc[~X["Cycle(R/I)"].isin(VALID_CYCLE_CODES), "Cycle(R/I)"] = np.nan

        if {"Weight (Kg)", "Height(Cm) ", "BMI"}.issubset(X.columns):
            X["BMI"] = X["Weight (Kg)"] / ((X["Height(Cm) "] / 100) ** 2)

        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features)
        return self.feature_names_in_


def split_features_target(df: pd.DataFrame):
    """Drop ID columns and separate X (features) from y (target).

    Patient File No. is dropped from X here, not earlier, so callers can
    still use it for group-aware cross-validation (GroupKFold) before
    handing X to the model.
    """
    X = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns] + [TARGET_COL], errors="ignore")
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    return X, y


def build_preprocessing_pipeline() -> Pipeline:
    """Build the full, unfit preprocessing pipeline.

    Structure:
      1. InvalidValueCorrector - dataset-specific cleaning (see class docstring)
      2. ColumnTransformer:
         - numeric columns:   median impute -> StandardScaler
         - binary Y/N columns: most-frequent impute (already 0/1, not scaled -
           keeps them directly interpretable in SHAP later)
         - nominal categorical (Blood Group, Cycle(R/I)): most-frequent impute
           -> one-hot encode (these are category codes, not ordered magnitudes,
           so they must not be scaled as if they were numbers)

    Output is a pandas DataFrame with meaningful column names (via
    `set_output(transform="pandas")`), not a bare numpy array, so later
    SHAP plots and the API layer can still label features by name.

    IMPORTANT: call `.fit()` on train_pool features only. Call `.transform()`
    (never `.fit()` or `.fit_transform()` again) on holdout_validation and on
    any new patient at inference time.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clip", ClipStandardized(bound=5.0)),
    ])
    binary_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("binary", binary_pipeline, BINARY_COLUMNS),
            ("categorical", categorical_pipeline, NOMINAL_CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("correct_invalid_values", InvalidValueCorrector()),
        ("column_transform", column_transformer),
    ])
    pipeline.set_output(transform="pandas")
    return pipeline
