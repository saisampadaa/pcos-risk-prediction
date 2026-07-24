"""
Phase 8 — SHAP explainability for the frozen Logistic Regression pipeline.

The final model is linear, so `shap.LinearExplainer` gives EXACT SHAP values
(no sampling approximation) - verified in development: base_value + sum of
SHAP values reproduces `model.decision_function()` exactly.

SHAP operates on the pipeline's PREPROCESSED feature space (49 columns:
scaled numeric, binary indicators, one-hot categories) - that's what the
model actually sees. `get_top_contributors` maps each transformed column back
to its raw, human-readable value (e.g. "Follicle No. (R) = 14", not a scaled
number), since a clinician or patient reading an explanation should see the
value they'd recognize, not the standardized one the model computes on.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import RISK_THRESHOLDS
from src.preprocessing import BINARY_COLUMNS, NOMINAL_CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def get_preprocessed_frame(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Run only the preprocessing step of the pipeline (never re-fit) and return
    the transformed, column-named DataFrame the model actually sees.
    """
    return pipeline.named_steps["preprocess"].transform(X)


def build_explainer(pipeline: Pipeline, X_background_raw: pd.DataFrame):
    """Build a shap.LinearExplainer for the model step, using `X_background_raw`
    (untransformed) as the background/reference distribution. Pass a patient-level,
    unaugmented DataFrame here (not train_pool as-is) so the background reflects
    real patients, not lifestyle.csv's repeated jittered copies.
    """
    X_background_t = get_preprocessed_frame(pipeline, X_background_raw)
    model = pipeline.named_steps["model"]
    explainer = shap.LinearExplainer(model, X_background_t)
    return explainer, X_background_t


def compute_shap_values(explainer, X_transformed: pd.DataFrame):
    """Return a shap.Explanation for already-preprocessed rows."""
    return explainer(X_transformed)


def _parse_transformed_name(name: str) -> Tuple[str, str]:
    """Map a ColumnTransformer output name (e.g. 'numeric__BMI',
    'categorical__Cycle(R/I)_2.0') back to (raw_column_name, clean_display_name).

    For one-hot categorical columns, the display name keeps the specific
    category value (e.g. "Cycle(R/I) = 2.0") rather than collapsing every
    dummy of the same raw column to an identical, ambiguous label.
    """
    prefix, rest = name.split("__", 1)
    if prefix == "categorical":
        for col in NOMINAL_CATEGORICAL_COLUMNS:
            if rest.startswith(col + "_"):
                category_value = rest[len(col) + 1:]
                return col, f"{col.strip()} = {category_value}"
        return rest, rest.strip()
    return rest, rest.strip()


FEATURE_DISPLAY_MAP: Dict[str, Tuple[str, str]] = {}


def _build_display_map(transformed_columns) -> Dict[str, Tuple[str, str]]:
    return {name: _parse_transformed_name(name) for name in transformed_columns}


def build_single_explanation(
    shap_values,
    row_idx: int,
    raw_row: pd.Series,
    transformed_columns,
    display_feature_names,
) -> "shap.Explanation":
    """Build a one-row shap.Explanation for `shap.plots.waterfall`, with the
    `.data` (the value shown next to each feature label) set to RAW,
    human-readable values instead of the scaled/clipped numbers the model
    actually computes on.

    Why this is needed: shap's own plot reads `.data` from whatever array was
    passed to the explainer call - which, for this pipeline, is the scaled
    and clipped numeric feature space (e.g. it would label a row "5.0 =
    PRG(ng/mL)" for a patient whose real reading is 85.0, since 85.0 gets
    standardized then clipped to the +/-5 bound). This does not change the
    SHAP values themselves (already computed correctly on the transformed
    space) - only what's printed as the feature's value in the plot.
    """
    display_map = _build_display_map(transformed_columns)
    raw_values = []
    for i, col in enumerate(transformed_columns):
        raw_col, _ = display_map[col]
        if col.startswith("numeric__"):
            raw_values.append(raw_row.get(raw_col, np.nan))
        else:
            raw_values.append(shap_values.data[row_idx][i])

    return shap.Explanation(
        values=shap_values.values[row_idx],
        base_values=shap_values.base_values[row_idx],
        data=np.array(raw_values, dtype=object),
        feature_names=list(display_feature_names),
    )


def clean_display_columns(transformed_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a preprocessed DataFrame with human-readable column
    names (prefixes stripped, one-hot category value kept) - for feeding to
    shap's built-in plots (beeswarm, waterfall), which label axes from
    whatever column names the input DataFrame has.
    """
    display_names = [_parse_transformed_name(c)[1] for c in transformed_df.columns]
    renamed = transformed_df.copy()
    renamed.columns = display_names
    return renamed


def get_top_contributors(
    shap_row: pd.Series,
    raw_row: pd.Series,
    transformed_columns,
    n: int = 5,
) -> Tuple[List[Dict], List[Dict]]:
    """For one patient, return (top_increasing, top_decreasing) contributors,
    each a list of {"feature": display name, "value": raw value, "shap_impact": float},
    sorted by |impact| within each direction. This is exactly the structure
    the Phase 9 AI agent consumes.

    IMPORTANT: for one-hot categorical columns, the label here is the RAW
    column name (e.g. "Cycle(R/I)"), not the specific one-hot category (e.g.
    "Cycle(R/I)_2.0" / "Regular"). A one-hot dummy contributes to a patient's
    prediction based on whether it is 0 or 1, so a dummy for a category the
    patient does NOT have can still show a nonzero SHAP impact (e.g. "not
    Regular" pushing risk up) - labeling it with the patient's own raw value
    (e.g. Cycle(R/I) = 4, Irregular) is accurate; labeling it with the
    category the dummy represents (2.0, Regular) would contradict the raw
    value shown right next to it. The category-specific label is used only
    in the global beeswarm/dependence plots, where it's aggregated across
    many patients and this ambiguity doesn't arise.
    """
    display_map = _build_display_map(transformed_columns)
    records = []
    for col, shap_val in zip(transformed_columns, shap_row):
        raw_col, _ = display_map[col]
        raw_value = raw_row.get(raw_col, None)
        records.append({"feature": raw_col.strip(), "value": raw_value, "shap_impact": float(shap_val)})

    # One-hot categorical columns (e.g. 8 Blood Group dummies) split a single
    # raw feature's effect across several columns. SHAP values are additive,
    # so summing the dummies belonging to the same raw column recovers that
    # feature's total contribution for this patient - the standard way to
    # collapse a one-hot group back into one number (rather than showing up
    # to 8 near-duplicate "Blood Group" rows with different partial impacts).
    records_df = (
        pd.DataFrame(records)
        .groupby("feature", as_index=False)
        .agg(value=("value", "first"), shap_impact=("shap_impact", "sum"))
        .sort_values("shap_impact", ascending=False)
    )
    increasing = records_df[records_df["shap_impact"] > 0].head(n).to_dict("records")
    decreasing = records_df[records_df["shap_impact"] < 0].sort_values("shap_impact").head(n).to_dict("records")
    return increasing, decreasing


def categorize_risk(probability: float) -> str:
    """Lower / Moderate / Elevated risk category. Thresholds are documented,
    arbitrary demo choices (RISK_THRESHOLDS in src/config.py), NOT clinically
    validated cutoffs - never present them as such.
    """
    if probability < RISK_THRESHOLDS["lower_upper_bound"]:
        return "Lower"
    if probability < RISK_THRESHOLDS["moderate_upper_bound"]:
        return "Moderate"
    return "Elevated"


def build_structured_explanation(
    pipeline: Pipeline,
    explainer,
    X_row_raw: pd.Series,
    X_row_transformed: pd.Series,
    shap_row: pd.Series,
    transformed_columns,
    n_factors: int = 3,
) -> Dict:
    """Build the exact structured-explanation JSON shape the Phase 9 AI agent
    expects (predicted_probability, risk_category, top_increasing_factors,
    top_decreasing_factors) for one patient.
    """
    proba = float(pipeline.predict_proba(X_row_raw.to_frame().T)[0, 1])
    increasing, decreasing = get_top_contributors(shap_row, X_row_raw, transformed_columns, n=n_factors)
    return {
        "predicted_probability": round(proba, 4),
        "risk_category": categorize_risk(proba),
        "top_increasing_factors": increasing,
        "top_decreasing_factors": decreasing,
    }
