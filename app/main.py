"""
Phase 11 — FastAPI prediction API.

Endpoints:
  GET  /health       - liveness check
  POST /predict       - full pipeline: preprocess -> predict -> SHAP-explain -> AI summary
  POST /explain        - same as /predict but skips the AI agent call (no LLM cost/latency)
  GET  /model-info     - frozen model metadata (Phase 6)

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException

from app.schemas import (
    ExplanationResult, ModelInfoResponse, PatientInput, PredictionResult, PredictResponse,
)
from app.services import get_model_info, predict_and_explain
from src.agent import DISCLAIMER, generate_explanation

app = FastAPI(
    title="PCOS Risk Estimation API",
    description=(
        "Academic research decision-support prototype. Estimates statistical PCOS risk "
        "from patterns in historical clinical/lifestyle data. Not a medical diagnosis."
    ),
    version="1.0.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    try:
        return get_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found on this server.")


def _build_response(patient: PatientInput, include_agent: bool) -> PredictResponse:
    try:
        structured = predict_and_explain(patient)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model artifacts not found on this server.")

    agent_summary = None
    agent_method = None
    if include_agent:
        agent_result = generate_explanation(structured)
        agent_summary = agent_result["explanation"]
        agent_method = agent_result["method"]

    return PredictResponse(
        prediction=PredictionResult(
            probability=structured["predicted_probability"],
            risk_category=structured["risk_category"],
        ),
        explanation=ExplanationResult(
            top_increasing_factors=structured["top_increasing_factors"],
            top_decreasing_factors=structured["top_decreasing_factors"],
        ),
        agent_summary=agent_summary,
        agent_method=agent_method,
        disclaimer=DISCLAIMER,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(patient: PatientInput):
    """Full pipeline including the AI plain-language summary (Phase 9)."""
    return _build_response(patient, include_agent=True)


@app.post("/explain", response_model=PredictResponse)
def explain(patient: PatientInput):
    """Same as /predict, but skips the AI agent call - just the model
    probability, risk category, and SHAP factors. Use this when you only
    need the numbers and don't want to pay LLM latency/cost.
    """
    return _build_response(patient, include_agent=False)
