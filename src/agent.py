"""
Phase 9 — AI explanation agent.

The agent NEVER predicts, trains, or decides PCOS risk. It receives the
structured output Phase 8 already computed (predicted probability, risk
category, SHAP-derived top increasing/decreasing factors) and turns it into
a plain-language paragraph. If it invents a fact not present in that input,
that is a bug in this module, not an acceptable behavior.

Two paths, always available:
  - `generate_llm_explanation()`  - one Groq API call (see model/param notes below)
  - `generate_rule_based_explanation()` - deterministic string template, no network call

`generate_explanation()` tries the LLM and falls back to the rule-based
template on ANY failure (missing API key, network error, rate limit, timeout) -
the application must keep working when the LLM API is unavailable.

Model/parameter notes (verified against Groq's docs before writing this -
console.groq.com/docs/models and /docs/quickstart):
  - Provider: Groq (OpenAI-compatible chat completions API), per the user's
    request - this project does not use the Anthropic API.
  - Model: llama-3.3-70b-versatile by default (strong instruction-following,
    important for the safety constraints in SYSTEM_PROMPT below). Swap to the
    smaller/faster llama-3.1-8b-instant via GROQ_AGENT_MODEL if latency/cost
    matters more than instruction-following fidelity for this deployment -
    that's a call for whoever deploys this, not hardcoded here.
  - "Low-temperature, deterministic output": unlike the Claude 4.6+ models,
    Groq's API does accept `temperature`, so this is implemented literally
    here (low, not zero - `temperature=0` is technically accepted but Groq's
    own guidance treats very low values as still having minor run-to-run
    variance; low + a tightly constrained prompt is what "deterministic
    enough" means in practice for any hosted LLM API).
"""

import json
import os
from typing import Dict, Optional

from dotenv import load_dotenv
from groq import Groq

load_dotenv()  # reads .env in the project root (if present) into os.environ; never overwrites already-set env vars

AGENT_MODEL = os.environ.get("GROQ_AGENT_MODEL", "llama-3.3-70b-versatile")

SYSTEM_PROMPT = """You are a plain-language explanation assistant for an academic PCOS (Polycystic Ovary Syndrome) risk-estimation research prototype.

You will be given a JSON object containing: a model-predicted probability, a risk category, and lists of factors that increased or decreased that specific prediction (derived from SHAP, a model-explanation technique).

Your only job is to translate that structured data into 3-5 sentences a non-technical reader can understand. You must:
- Use ONLY the facts given in the JSON input. Never invent a clinical fact, a feature, or a value that is not present in it.
- Never state or imply a diagnosis. Do not say "you have PCOS" or "you do not have PCOS" - only describe an estimated risk level.
- Never prescribe or suggest medication, dosages, or specific treatments.
- Never promise a health outcome or guarantee accuracy.
- Clearly attribute the finding to the model ("the model estimated...", "the strongest contributing factors were...") rather than stating it as established medical fact.
- If the risk category is "Elevated", or several increasing factors are symptom-based, explicitly encourage the reader to discuss the result with a qualified healthcare professional.
- Do not add a disclaimer paragraph yourself - the application appends a standard disclaimer after your response automatically.
- Keep the tone calm, factual, and supportive - this is a research decision-support tool, not an alarming diagnostic result.
"""

DISCLAIMER = (
    "Important: This is an academic research prototype that estimates statistical risk "
    "from patterns in historical data. It is not a medical diagnosis and does not replace "
    "evaluation by a qualified healthcare professional. If you have symptoms or concerns, "
    "please consult a doctor."
)


def _format_factor(factor: Dict) -> str:
    return f"{factor['feature']} (value: {factor['value']})"


def generate_rule_based_explanation(structured_explanation: Dict) -> str:
    """Deterministic, no-network fallback. Always available - the application
    must keep functioning when the LLM API is down, rate-limited, or unreachable.
    """
    proba_pct = round(structured_explanation["predicted_probability"] * 100)
    category = structured_explanation["risk_category"]
    increasing = structured_explanation.get("top_increasing_factors", [])
    decreasing = structured_explanation.get("top_decreasing_factors", [])

    lines = [f"The model estimated a predicted PCOS risk of {proba_pct}%, categorized as '{category}'."]

    if increasing:
        top = ", ".join(_format_factor(f) for f in increasing[:3])
        lines.append(f"The factors that most increased this estimate were: {top}.")
    if decreasing:
        top = ", ".join(_format_factor(f) for f in decreasing[:3])
        lines.append(f"The factors that most reduced this estimate were: {top}.")

    if category == "Elevated":
        lines.append(
            "Given the elevated estimate, discussing these results with a qualified "
            "healthcare professional is recommended."
        )

    return " ".join(lines)


def generate_llm_explanation(structured_explanation: Dict, client: Optional[Groq] = None) -> str:
    """One Groq API call. Raises on any API failure - callers should catch and
    fall back to `generate_rule_based_explanation`, not call this directly in
    a user-facing path without a try/except around it.

    Reads the API key from the GROQ_API_KEY environment variable (the Groq
    SDK default) - never hardcode a key here. See .env.example.
    """
    client = client or Groq(api_key=os.environ.get("GROQ_API_KEY"))

    user_message = (
        "Structured model output (use ONLY these facts):\n\n"
        f"{json.dumps(structured_explanation, indent=2, default=str)}\n\n"
        "Write the plain-language explanation now."
    )

    response = client.chat.completions.create(
        model=AGENT_MODEL,
        max_tokens=400,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    finish_reason = response.choices[0].finish_reason
    text = response.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(f"LLM returned an empty explanation (finish_reason={finish_reason})")
    return text.strip()


def generate_explanation(structured_explanation: Dict, use_llm: bool = True) -> Dict:
    """Orchestrator: try the LLM, fall back to the rule-based template on any
    failure. Always returns the same shape so the API/app layer (Phase 10/11)
    doesn't need to know which path produced the text.
    """
    if use_llm:
        try:
            body = generate_llm_explanation(structured_explanation)
            method = "llm"
        except Exception as exc:  # noqa: BLE001 - any failure falls back, by design
            body = generate_rule_based_explanation(structured_explanation)
            method = f"rule_based_fallback (LLM error: {exc.__class__.__name__})"
    else:
        body = generate_rule_based_explanation(structured_explanation)
        method = "rule_based"

    return {
        "explanation": f"{body}\n\n{DISCLAIMER}",
        "method": method,
        "disclaimer": DISCLAIMER,
    }
