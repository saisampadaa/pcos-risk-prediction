"""
Shared prediction/explanation logic used by both the FastAPI backend
(Phase 11, app/main.py) and the Streamlit demo (Phase 10, app/streamlit_app.py).
Keeping this in one module means the two UIs can never compute a prediction
or an explanation differently - they call the same functions.

Everything expensive (the frozen pipeline, the SHAP background sample, the
SHAP explainer) is loaded ONCE at import time as module-level singletons,
not per-request - a FastAPI worker or a Streamlit session should not be
re-reading the pipeline off disk or re-fitting the SHAP explainer on every
prediction.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd

from src.config import INTERIM_DIR, MODELS_DIR, PATIENT_ID_COL
from src.data_loader import load_clinical_2_full
from src.feature_mapping import harmonize
from src.explain import (
    build_explainer, build_structured_explanation, categorize_risk, get_preprocessed_frame,
)
from src.agent import generate_explanation
from app.schemas import BLOOD_GROUP_CODES, CYCLE_CODES, PatientInput

# ---------------------------------------------------------------------------
# Module-level singletons - loaded once, reused across every request/session.
# ---------------------------------------------------------------------------
_pipeline = None
_explainer = None
_background_columns = None


def _load_singletons():
    global _pipeline, _explainer, _background_columns
    if _pipeline is not None:
        return

    _pipeline = joblib.load(MODELS_DIR / "pcos_risk_pipeline.joblib")

    patient_split = pd.read_csv(INTERIM_DIR / "patient_split.csv")
    c2 = harmonize(load_clinical_2_full())
    train_ids = set(patient_split.loc[patient_split["split"] == "train", PATIENT_ID_COL])
    train_patients_df = c2[c2[PATIENT_ID_COL].isin(train_ids)].reset_index(drop=True)
    X_background = train_patients_df.drop(columns=["Sl. No", PATIENT_ID_COL, "PCOS (Y/N)"], errors="ignore")

    _explainer, X_bg_transformed = build_explainer(_pipeline, X_background)
    _background_columns = list(X_bg_transformed.columns)


# Raw dataset column name for every PatientInput field except the three
# derived ones (bmi, fsh_lh_ratio, waist_hip_ratio) and the two coded fields
# (blood_group, cycle_regularity), which are handled separately below.
FIELD_TO_RAW_COLUMN = {
    "age_years": " Age (yrs)",
    "weight_kg": "Weight (Kg)",
    "height_cm": "Height(Cm) ",
    "pulse_rate_bpm": "Pulse rate(bpm) ",
    "resp_rate": "RR (breaths/min)",
    "hemoglobin": "Hb(g/dl)",
    "bp_systolic": "BP _Systolic (mmHg)",
    "bp_diastolic": "BP _Diastolic (mmHg)",
    "cycle_length_days": "Cycle length(days)",
    "marriage_years": "Marraige Status (Yrs)",
    "n_abortions": "No. of abortions",
    "beta_hcg_1": "  I   beta-HCG(mIU/mL)",
    "beta_hcg_2": "II    beta-HCG(mIU/mL)",
    "fsh": "FSH(mIU/mL)",
    "lh": "LH(mIU/mL)",
    "tsh": "TSH (mIU/L)",
    "amh": "AMH(ng/mL)",
    "prl": "PRL(ng/mL)",
    "vit_d3": "Vit D3 (ng/mL)",
    "prg": "PRG(ng/mL)",
    "rbs": "RBS(mg/dl)",
    "hip_inch": "Hip(inch)",
    "waist_inch": "Waist(inch)",
    "follicle_left": "Follicle No. (L)",
    "follicle_right": "Follicle No. (R)",
    "avg_follicle_size_left": "Avg. F size (L) (mm)",
    "avg_follicle_size_right": "Avg. F size (R) (mm)",
    "endometrium_mm": "Endometrium (mm)",
}

BINARY_FIELD_TO_RAW_COLUMN = {
    "pregnant": "Pregnant(Y/N)",
    "weight_gain": "Weight gain(Y/N)",
    "hirsutism": "hair growth(Y/N)",
    "skin_darkening": "Skin darkening (Y/N)",
    "hair_loss": "Hair loss(Y/N)",
    "acne": "Pimples(Y/N)",
    "fast_food": "Fast food (Y/N)",
    "regular_exercise": "Reg.Exercise(Y/N)",
}


def patient_input_to_dataframe(patient: PatientInput) -> pd.DataFrame:
    """Convert the clean API/UI input into a one-row DataFrame with the exact
    raw column names and derived fields the trained pipeline expects.

    CRITICAL: the returned columns are reordered to match the exact order the
    pipeline was fit on (`correct_invalid_values.feature_names_in_`), not just
    matched by name. scikit-learn's `set_output(transform="pandas")` (used by
    build_preprocessing_pipeline) relabels a transformer's output columns
    using its fitted `get_feature_names_out()` - which returns the FIT-TIME
    order - regardless of what order the DataFrame actually holds its values
    in. A same-named-but-differently-ordered input therefore gets its values
    silently relabeled under the wrong column names with no error and no
    warning. This was caught by hand (predictions came back correct but SHAP
    attributions were scrambled - e.g. a high follicle count showing up as a
    *decreasing* factor) before this reordering step was added; every
    dataframe fed to the pipeline from here on MUST go through this function.
    """
    _load_singletons()
    data = patient.model_dump()
    row = {raw_col: data[field] for field, raw_col in FIELD_TO_RAW_COLUMN.items()}
    for field, raw_col in BINARY_FIELD_TO_RAW_COLUMN.items():
        row[raw_col] = int(data[field])

    row["Blood Group"] = BLOOD_GROUP_CODES[data["blood_group"]]
    row["Cycle(R/I)"] = CYCLE_CODES[data["cycle_regularity"]]

    # Derived fields - computed here, not collected from the user (see schemas.py docstring).
    row["BMI"] = data["weight_kg"] / ((data["height_cm"] / 100) ** 2)
    row["FSH/LH"] = data["fsh"] / data["lh"] if data["lh"] else 0.0
    row["Waist:Hip Ratio"] = data["waist_inch"] / data["hip_inch"] if data["hip_inch"] else 0.0

    df = pd.DataFrame([row])
    fit_time_columns = _pipeline.named_steps["preprocess"].named_steps["correct_invalid_values"].feature_names_in_
    return df[list(fit_time_columns)]


def predict_and_explain(patient: PatientInput, n_factors: int = 5) -> dict:
    """Run the full Phase 11 pipeline for one patient: preprocess -> predict ->
    SHAP-explain -> structure the result. Does NOT call the AI agent - callers
    that want the plain-language summary call `generate_explanation` themselves
    (kept separate so /predict can be used without paying for an LLM call).
    """
    _load_singletons()

    X_raw = patient_input_to_dataframe(patient)
    X_transformed = get_preprocessed_frame(_pipeline, X_raw)

    proba = float(_pipeline.predict_proba(X_raw)[0, 1])
    shap_values = _explainer(X_transformed)
    shap_row = pd.Series(shap_values.values[0], index=_background_columns)

    structured = build_structured_explanation(
        _pipeline, _explainer, X_raw.iloc[0], X_transformed.iloc[0], shap_row,
        _background_columns, n_factors=n_factors,
    )
    # build_structured_explanation already computes its own probability/category
    # from X_raw - reuse it directly rather than recomputing here.
    structured["predicted_probability"] = round(proba, 4)
    structured["risk_category"] = categorize_risk(proba)
    return structured


def get_model_info() -> dict:
    import json
    with open(MODELS_DIR / "model_metadata.json") as f:
        return json.load(f)
