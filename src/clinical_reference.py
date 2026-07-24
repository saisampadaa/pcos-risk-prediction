"""
Clinical reference ranges and descriptive pattern summary.

This module is deliberately rule-based and deterministic - NOT part of the
SHAP explanation and NOT part of the LLM agent's job. It answers three
things the user asked for directly, using only real, already-collected
values (never inventing a fact):

1. BMI category, against the standard WHO cutoffs.
2. Hormonal/lab test flags, against general clinical reference ranges.
3. A descriptive summary of which clinically-relevant symptom clusters are
   present - e.g. "hyperandrogenic signs", "ultrasound criterion met",
   "possible insulin-resistance-associated indicators". This is NOT a PCOS
   "subtype" diagnosis. True phenotyping requires tests this dataset does
   not have (fasting insulin, free/total testosterone, HOMA-IR) - the
   module says so explicitly rather than pretending otherwise.

Reference ranges below are widely-cited general values (WHO BMI categories,
ADA random-glucose thresholds, standard endocrine lab ranges) - they vary
somewhat by lab/assay/population and are NOT a substitute for the reference
interval printed on an actual lab report. Every function that uses one says
so in its output.
"""

from typing import Dict, List


def bmi_category(bmi: float) -> Dict[str, str]:
    """WHO adult BMI categories."""
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25.0:
        category = "Normal weight"
    elif bmi < 30.0:
        category = "Overweight"
    else:
        category = "Obese"
    return {
        "value": round(bmi, 1),
        "category": category,
        "normal_range": "18.5-24.9 kg/m² (WHO adult classification)",
    }


# Each entry: (low, high, unit, note). `low`/`high` are None where there is no
# lower/upper clinical concern in that direction for this context.
_HORMONE_RANGES = {
    "AMH(ng/mL)": (1.0, 4.0, "ng/mL", "Higher values are commonly seen in PCOS (reflects higher antral follicle count); lower values may reflect reduced ovarian reserve."),
    "FSH(mIU/mL)": (3.5, 12.5, "mIU/mL", "Follicular-phase reference range."),
    "LH(mIU/mL)": (2.4, 12.6, "mIU/mL", "Follicular-phase reference range."),
    "TSH (mIU/L)": (0.4, 4.0, "mIU/L", "Standard adult euthyroid range."),
    "PRL(ng/mL)": (4.7, 23.3, "ng/mL", "Non-pregnant adult women."),
    "Vit D3 (ng/mL)": (30.0, 100.0, "ng/mL", "Below 20 is generally considered deficient, 20-29 insufficient (Endocrine Society)."),
}


def hormone_flags(values: Dict[str, float]) -> List[Dict]:
    """For each hormone present in `values`, flag Low/Normal/High against the
    general reference ranges above. `values` keys must match _HORMONE_RANGES.
    """
    results = []
    for name, val in values.items():
        if name not in _HORMONE_RANGES or val is None:
            continue
        low, high, unit, note = _HORMONE_RANGES[name]
        if val < low:
            status = "Low"
        elif val > high:
            status = "High"
        else:
            status = "Normal"
        results.append({
            "test": name,
            "value": val,
            "unit": unit,
            "status": status,
            "reference_range": f"{low}-{high} {unit}",
            "note": note,
        })
    return results


def glucose_flag(rbs: float) -> Dict:
    """Random blood sugar, per commonly-cited ADA screening thresholds.
    (RBS is a random, non-fasting glucose test - not itself an insulin or
    HOMA-IR measurement.)
    """
    if rbs >= 200:
        status = "High (consider follow-up)"
    elif rbs >= 140:
        status = "Borderline (consider follow-up)"
    else:
        status = "Normal"
    return {"value": rbs, "unit": "mg/dl", "status": status, "reference_range": "<140 mg/dl (random, non-fasting)"}


def waist_hip_flag(ratio: float) -> Dict:
    """WHO central-obesity threshold for women."""
    status = "Elevated (central-obesity threshold)" if ratio > 0.85 else "Normal"
    return {"value": round(ratio, 2), "status": status, "reference_range": "<= 0.85 (WHO, women)"}


def describe_clinical_pattern(raw_row: Dict) -> Dict[str, List[str]]:
    """Descriptive summary of which clinically-relevant symptom clusters are
    present, in the patient's own recorded values. This groups already-known
    facts under clinically meaningful headings - it does NOT diagnose a PCOS
    "subtype". Real phenotyping needs tests this dataset doesn't have
    (fasting insulin, HOMA-IR, free/total testosterone) - flagged explicitly
    in the "not_assessed" list rather than guessed at.
    """
    present, absent = [], []

    hyperandrogenic_signs = {
        "hirsutism": "Excess hair growth (hirsutism)",
        "acne": "Acne / pimples",
        "skin_darkening": "Skin darkening (acanthosis nigricans)",
        "hair_loss": "Hair loss",
    }
    any_hyperandrogenic = any(raw_row.get(k) for k in hyperandrogenic_signs)
    if any_hyperandrogenic:
        found = [label for k, label in hyperandrogenic_signs.items() if raw_row.get(k)]
        present.append(f"Hyperandrogenic signs present: {', '.join(found)} (part of the Rotterdam diagnostic criteria for PCOS)")
    else:
        absent.append("No hyperandrogenic signs reported (hirsutism, acne, skin darkening, hair loss)")

    follicle_l = raw_row.get("follicle_left", 0) or 0
    follicle_r = raw_row.get("follicle_right", 0) or 0
    if max(follicle_l, follicle_r) >= 12:
        present.append(f"Ultrasound follicle-count criterion met (L={follicle_l}, R={follicle_r}; Rotterdam threshold is >=12 per ovary)")
    else:
        absent.append(f"Follicle counts below the Rotterdam ultrasound threshold of 12 per ovary (L={follicle_l}, R={follicle_r})")

    if raw_row.get("cycle_regularity") == "Irregular":
        present.append("Irregular menstrual cycle reported (part of the Rotterdam diagnostic criteria - oligo/anovulation)")
    else:
        absent.append("Regular menstrual cycle reported")

    bmi = raw_row.get("bmi")
    weight_gain = raw_row.get("weight_gain")
    if (bmi is not None and bmi >= 25) or weight_gain:
        detail = []
        if bmi is not None and bmi >= 25:
            detail.append(f"BMI {bmi:.1f} (overweight or above)")
        if weight_gain:
            detail.append("reported weight gain")
        present.append(f"Weight-related factors present: {', '.join(detail)}")
    else:
        absent.append("No weight-related risk factors flagged (BMI in normal range, no reported weight gain)")

    thyroid = raw_row.get("tsh")
    if thyroid is not None and not (0.4 <= thyroid <= 4.0):
        present.append(f"Thyroid marker (TSH={thyroid}) outside the standard reference range - a thyroid disorder can produce PCOS-like symptoms and is usually checked as part of ruling out other causes")
    else:
        absent.append("Thyroid marker (TSH) within the standard reference range")

    insulin_resistance_proxies = []
    if raw_row.get("skin_darkening"):
        insulin_resistance_proxies.append("skin darkening")
    whr = raw_row.get("waist_hip_ratio")
    if whr is not None and whr > 0.85:
        insulin_resistance_proxies.append(f"elevated waist:hip ratio ({whr:.2f})")
    rbs = raw_row.get("rbs")
    if rbs is not None and rbs >= 140:
        insulin_resistance_proxies.append(f"elevated random blood sugar ({rbs} mg/dl)")
    if insulin_resistance_proxies:
        present.append(
            f"Possible insulin-resistance-associated indicators present: {', '.join(insulin_resistance_proxies)}. "
            "These are indirect proxies, not a diagnosis - insulin resistance itself requires a fasting insulin "
            "level or HOMA-IR calculation, which this tool does not collect."
        )

    return {
        "present": present,
        "absent": absent,
        "not_assessed": [
            "Fasting insulin / HOMA-IR (direct insulin resistance test)",
            "Free or total testosterone (direct hyperandrogenism lab test)",
            "Formal PCOS phenotype (A/B/C/D) classification - requires a clinician to weigh all Rotterdam criteria together",
        ],
    }


TIERED_GUIDANCE = {
    "Lower": (
        "Your estimated risk is in the lower range. No urgent action is indicated by this tool. "
        "If you develop new symptoms (irregular cycles, unusual hair growth, acne, or unexplained weight change), "
        "it's still reasonable to mention them at your next routine check-up."
    ),
    "Moderate": (
        "Your estimated risk is in the moderate range. Consider discussing these results with a healthcare "
        "provider at your next visit, particularly if any of the flagged factors above are new or bothering you. "
        "General lifestyle factors commonly discussed in PCOS management include regular physical activity and a "
        "balanced diet - a clinician can help tailor this to you specifically."
    ),
    "Elevated": (
        "Your estimated risk is in the elevated range. Discussing these results with a qualified healthcare "
        "professional (a gynecologist or endocrinologist) is recommended, rather than waiting for a routine visit. "
        "A full workup for PCOS typically includes tests this screening tool does not perform - such as fasting "
        "insulin, free/total testosterone, and a pelvic ultrasound if one hasn't been done - so a clinician visit "
        "is the appropriate next step, not a replacement for one."
    ),
}
