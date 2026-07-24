"""
Pydantic request/response models for the PCOS risk API (Phase 11) and the
Streamlit app (Phase 10) - both import from here so the two surfaces can
never drift apart on what a "patient" input looks like.

Field names here are clean, human-readable identifiers - NOT the raw
dataset column names (those have awkward spacing/casing baked in from the
source spreadsheet, e.g. "  I   beta-HCG(mIU/mL)"). `services.py` maps
between the two; this file is what a person filling out a form actually sees.

Three derived fields are intentionally NOT collected from the user:
BMI, FSH/LH ratio, and Waist:Hip ratio are arithmetic functions of other
fields already on the form (weight/height, FSH/LH, waist/hip) - asking for
them separately would just invite inconsistent manual entry. `services.py`
computes all three.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# Blood Group codes per the original data-collection Instructions sheet (Phase 1).
BLOOD_GROUP_CODES = {
    "A+": 11, "A-": 12, "B+": 13, "B-": 14,
    "O+": 15, "O-": 16, "AB+": 17, "AB-": 18,
}
CYCLE_CODES = {"Regular": 2, "Irregular": 4}


class PatientInput(BaseModel):
    """Everything the final model needs, in the units a clinician/researcher
    filling in a patient's completed workup would naturally have on hand.
    This is a research decision-support form, not a home self-screening
    questionnaire - several fields require lab results or an ultrasound.
    """

    # Demographics
    age_years: float = Field(..., ge=10, le=70, description="Age in years")
    weight_kg: float = Field(..., ge=25, le=200, description="Weight in kilograms")
    height_cm: float = Field(..., ge=100, le=210, description="Height in centimeters")
    blood_group: Literal["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

    # Vitals
    pulse_rate_bpm: float = Field(..., ge=40, le=200, description="Pulse rate, beats per minute")
    resp_rate: float = Field(..., ge=10, le=40, description="Respiratory rate, breaths per minute")
    hemoglobin: float = Field(..., ge=5, le=20, description="Hemoglobin, g/dl")
    bp_systolic: float = Field(..., ge=70, le=220, description="Systolic blood pressure, mmHg")
    bp_diastolic: float = Field(..., ge=40, le=140, description="Diastolic blood pressure, mmHg")

    # Menstrual / reproductive history
    cycle_regularity: Literal["Regular", "Irregular"]
    cycle_length_days: float = Field(..., ge=1, le=90, description="Cycle length in days")
    marriage_years: float = Field(0, ge=0, le=60, description="Years married (0 if not applicable)")
    pregnant: bool = False
    n_abortions: int = Field(0, ge=0, le=20)

    # Hormonal / blood labs
    beta_hcg_1: float = Field(..., ge=0, le=100000, description="Beta-HCG, first reading (mIU/mL)")
    beta_hcg_2: float = Field(..., ge=0, le=100000, description="Beta-HCG, second reading (mIU/mL)")
    fsh: float = Field(..., ge=0, le=100, description="FSH (mIU/mL)")
    lh: float = Field(..., ge=0, le=100, description="LH (mIU/mL)")
    tsh: float = Field(..., ge=0, le=50, description="TSH (mIU/L)")
    amh: float = Field(..., ge=0, le=100, description="AMH (ng/mL)")
    prl: float = Field(..., ge=0, le=200, description="Prolactin (ng/mL)")
    vit_d3: float = Field(..., ge=0, le=200, description="Vitamin D3 (ng/mL)")
    prg: float = Field(..., ge=0, le=30, description="Progesterone (ng/mL)")
    rbs: float = Field(..., ge=30, le=400, description="Random blood sugar (mg/dl)")

    # Body measurements
    hip_inch: float = Field(..., ge=20, le=70, description="Hip circumference, inches")
    waist_inch: float = Field(..., ge=15, le=60, description="Waist circumference, inches")

    # Self-reported symptoms
    weight_gain: bool = False
    hirsutism: bool = Field(False, description="Excess hair growth")
    skin_darkening: bool = False
    hair_loss: bool = False
    acne: bool = Field(False, description="Pimples")
    fast_food: bool = Field(False, description="Frequent fast-food consumption")
    regular_exercise: bool = False

    # Ultrasound findings
    follicle_left: float = Field(..., ge=0, le=50, description="Follicle count, left ovary")
    follicle_right: float = Field(..., ge=0, le=50, description="Follicle count, right ovary")
    avg_follicle_size_left: float = Field(..., ge=0, le=40, description="Average follicle size, left (mm)")
    avg_follicle_size_right: float = Field(..., ge=0, le=40, description="Average follicle size, right (mm)")
    endometrium_mm: float = Field(..., ge=0, le=30, description="Endometrial thickness (mm)")


class Factor(BaseModel):
    feature: str
    value: float
    shap_impact: float


class PredictionResult(BaseModel):
    probability: float
    risk_category: Literal["Lower", "Moderate", "Elevated"]


class ExplanationResult(BaseModel):
    top_increasing_factors: List[Factor]
    top_decreasing_factors: List[Factor]


class PredictResponse(BaseModel):
    prediction: PredictionResult
    explanation: ExplanationResult
    agent_summary: Optional[str] = None
    agent_method: Optional[str] = None
    disclaimer: str


class ModelInfoResponse(BaseModel):
    model_type: str
    hyperparameters: dict
    training_rows: int
    training_patients: int
    n_features_total: int
    n_features_nonzero: int
    cv_metrics: dict
    selection_rationale: str
    known_limitations: List[str]
