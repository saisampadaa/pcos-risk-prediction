"""
Phase 10 — Streamlit demo app.

Calls `app/services.py` and `src/agent.py` directly (not over HTTP) - simplest
setup for a local capstone demo, no need to run the FastAPI server separately.

Run: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.schemas import PatientInput
from app.services import predict_and_explain
from src.agent import generate_explanation

st.set_page_config(page_title="PCOS Risk Estimator (Research Prototype)", layout="centered")

st.title("PCOS Risk Estimation — Research Prototype")
st.caption(
    "Academic decision-support tool. Estimates statistical risk from patterns in historical "
    "clinical/lifestyle data. This is not a diagnostic tool - several fields below require lab "
    "results or an ultrasound finding, so this form is meant for a clinician or researcher "
    "entering a patient's completed workup, not for self-screening at home."
)

with st.form("patient_form"):
    st.subheader("Demographics")
    c1, c2, c3 = st.columns(3)
    age_years = c1.number_input("Age (years)", 10.0, 70.0, 28.0)
    weight_kg = c2.number_input("Weight (kg)", 25.0, 200.0, 60.0)
    height_cm = c3.number_input("Height (cm)", 100.0, 210.0, 160.0)
    blood_group = st.selectbox("Blood group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])

    st.subheader("Vitals")
    c1, c2, c3 = st.columns(3)
    pulse_rate_bpm = c1.number_input("Pulse rate (bpm)", 40.0, 200.0, 76.0)
    resp_rate = c2.number_input("Respiratory rate (breaths/min)", 10.0, 40.0, 18.0)
    hemoglobin = c3.number_input("Hemoglobin (g/dl)", 5.0, 20.0, 12.0)
    c1, c2 = st.columns(2)
    bp_systolic = c1.number_input("BP systolic (mmHg)", 70.0, 220.0, 120.0)
    bp_diastolic = c2.number_input("BP diastolic (mmHg)", 40.0, 140.0, 80.0)

    st.subheader("Menstrual / reproductive history")
    c1, c2 = st.columns(2)
    cycle_regularity = c1.selectbox("Cycle regularity", ["Regular", "Irregular"])
    cycle_length_days = c2.number_input("Cycle length (days)", 1.0, 90.0, 30.0)
    c1, c2, c3 = st.columns(3)
    marriage_years = c1.number_input("Years married (0 if N/A)", 0.0, 60.0, 0.0)
    pregnant = c2.checkbox("Currently pregnant")
    n_abortions = c3.number_input("Number of prior abortions", 0, 20, 0)

    st.subheader("Hormonal / blood labs")
    c1, c2, c3 = st.columns(3)
    beta_hcg_1 = c1.number_input("Beta-HCG, reading 1 (mIU/mL)", 0.0, 100000.0, 2.0)
    beta_hcg_2 = c2.number_input("Beta-HCG, reading 2 (mIU/mL)", 0.0, 100000.0, 2.0)
    fsh = c3.number_input("FSH (mIU/mL)", 0.0, 100.0, 6.0)
    c1, c2, c3 = st.columns(3)
    lh = c1.number_input("LH (mIU/mL)", 0.0, 100.0, 6.0)
    tsh = c2.number_input("TSH (mIU/L)", 0.0, 50.0, 2.5)
    amh = c3.number_input("AMH (ng/mL)", 0.0, 100.0, 4.0)
    c1, c2, c3 = st.columns(3)
    prl = c1.number_input("Prolactin (ng/mL)", 0.0, 200.0, 15.0)
    vit_d3 = c2.number_input("Vitamin D3 (ng/mL)", 0.0, 200.0, 30.0)
    prg = c3.number_input("Progesterone (ng/mL)", 0.0, 30.0, 0.4)
    rbs = st.number_input("Random blood sugar (mg/dl)", 30.0, 400.0, 95.0)

    st.subheader("Body measurements")
    c1, c2 = st.columns(2)
    hip_inch = c1.number_input("Hip circumference (inch)", 20.0, 70.0, 38.0)
    waist_inch = c2.number_input("Waist circumference (inch)", 15.0, 60.0, 32.0)

    st.subheader("Self-reported symptoms")
    c1, c2, c3, c4 = st.columns(4)
    weight_gain = c1.checkbox("Weight gain")
    hirsutism = c2.checkbox("Excess hair growth")
    skin_darkening = c3.checkbox("Skin darkening")
    hair_loss = c4.checkbox("Hair loss")
    c1, c2, c3 = st.columns(3)
    acne = c1.checkbox("Acne / pimples")
    fast_food = c2.checkbox("Frequent fast food")
    regular_exercise = c3.checkbox("Regular exercise")

    st.subheader("Ultrasound findings")
    c1, c2 = st.columns(2)
    follicle_left = c1.number_input("Follicle count, left ovary", 0.0, 50.0, 6.0)
    follicle_right = c2.number_input("Follicle count, right ovary", 0.0, 50.0, 6.0)
    c1, c2 = st.columns(2)
    avg_follicle_size_left = c1.number_input("Avg. follicle size, left (mm)", 0.0, 40.0, 5.0)
    avg_follicle_size_right = c2.number_input("Avg. follicle size, right (mm)", 0.0, 40.0, 5.0)
    endometrium_mm = st.number_input("Endometrial thickness (mm)", 0.0, 30.0, 8.0)

    use_llm = st.checkbox("Generate AI plain-language summary (requires GROQ_API_KEY)", value=True)
    submitted = st.form_submit_button("Estimate risk")

if submitted:
    patient = PatientInput(
        age_years=age_years, weight_kg=weight_kg, height_cm=height_cm, blood_group=blood_group,
        pulse_rate_bpm=pulse_rate_bpm, resp_rate=resp_rate, hemoglobin=hemoglobin,
        bp_systolic=bp_systolic, bp_diastolic=bp_diastolic,
        cycle_regularity=cycle_regularity, cycle_length_days=cycle_length_days,
        marriage_years=marriage_years, pregnant=pregnant, n_abortions=n_abortions,
        beta_hcg_1=beta_hcg_1, beta_hcg_2=beta_hcg_2, fsh=fsh, lh=lh, tsh=tsh, amh=amh,
        prl=prl, vit_d3=vit_d3, prg=prg, rbs=rbs, hip_inch=hip_inch, waist_inch=waist_inch,
        weight_gain=weight_gain, hirsutism=hirsutism, skin_darkening=skin_darkening,
        hair_loss=hair_loss, acne=acne, fast_food=fast_food, regular_exercise=regular_exercise,
        follicle_left=follicle_left, follicle_right=follicle_right,
        avg_follicle_size_left=avg_follicle_size_left, avg_follicle_size_right=avg_follicle_size_right,
        endometrium_mm=endometrium_mm,
    )

    with st.spinner("Running model and computing explanation..."):
        structured = predict_and_explain(patient)

    proba_pct = round(structured["predicted_probability"] * 100)
    category = structured["risk_category"]
    color = {"Lower": "green", "Moderate": "orange", "Elevated": "red"}[category]

    st.markdown("---")
    st.markdown(f"### Estimated PCOS risk: **{proba_pct}%**")
    st.markdown(f"### Risk category: :{color}[{category}]")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Main factors increasing the estimate:**")
        for f in structured["top_increasing_factors"]:
            st.markdown(f"- {f['feature']} ({f['value']})")
    with col2:
        st.markdown("**Factors reducing the estimate:**")
        for f in structured["top_decreasing_factors"]:
            st.markdown(f"- {f['feature']} ({f['value']})")

    with st.spinner("Generating explanation..."):
        agent_result = generate_explanation(structured, use_llm=use_llm)
    st.markdown("**Explanation:**")
    st.info(agent_result["explanation"])
    st.caption(f"Explanation method: {agent_result['method']}")

    st.markdown("---")
    st.markdown("## Understanding your result")
    st.caption(
        "The sections below are rule-based (not machine-learning output) and use general clinical "
        "reference ranges - they can vary by lab/assay and are not a substitute for your lab report's "
        "own reference interval."
    )

    summary = structured["clinical_summary"]

    st.markdown("### BMI")
    bmi = summary["bmi"]
    st.metric("BMI", f"{bmi['value']} kg/m²", bmi["category"])
    st.caption(f"Normal range: {bmi['normal_range']}")

    st.markdown("### Hormonal tests")
    for h in summary["hormone_tests"]:
        flag_color = {"Normal": "green", "High": "red", "Low": "orange"}[h["status"]]
        st.markdown(f"- **{h['test']}**: {h['value']} {h['unit']} — :{flag_color}[{h['status']}] (reference: {h['reference_range']})")
        st.caption(h["note"])

    st.markdown("### Metabolic indicators")
    g = summary["glucose"]
    w = summary["waist_hip_ratio"]
    st.markdown(f"- **Random blood sugar**: {g['value']} {g['unit']} — {g['status']} (reference: {g['reference_range']})")
    st.markdown(f"- **Waist:Hip ratio**: {w['value']} — {w['status']} (reference: {w['reference_range']})")

    st.markdown("### Clinical patterns observed")
    st.caption("This groups your already-entered values under clinically meaningful headings. It is NOT a PCOS subtype diagnosis.")
    patterns = summary["observed_patterns"]
    for p in patterns["present"]:
        st.markdown(f"✅ {p}")
    for p in patterns["absent"]:
        st.markdown(f"⬜ {p}")
    with st.expander("Not assessed by this tool"):
        for p in patterns["not_assessed"]:
            st.markdown(f"- {p}")

    st.markdown("### What this means / next steps")
    st.info(summary["guidance"])

    st.markdown("---")
    st.warning(
        "This application is an academic decision-support prototype. It does not provide a "
        "medical diagnosis. Risk-category thresholds (Lower < 30%, Moderate 30-70%, Elevated ≥ 70%) "
        "are documented, arbitrary choices for this demo - not clinically validated cutoffs."
    )
