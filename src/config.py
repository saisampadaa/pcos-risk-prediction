"""
Central path and constant definitions for the PCOS risk prediction capstone.

Every other module imports paths from here instead of hardcoding strings,
so the project can be moved or reorganized without touching multiple files.
"""

from pathlib import Path

# PROJECT_ROOT is the repository root (parent of the `src/` folder this file lives in).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
RESULTS_DIR = REPORTS_DIR / "results"

# Raw source files, as delivered. Do not overwrite these — treat data/raw as read-only.
CLINICAL_1_PATH = RAW_DIR / "PCOS_clinical_1.csv"      # 541 rows, 6 cols: infertility-related subset
CLINICAL_2_PATH = RAW_DIR / "PCOS_clinical_2.xlsx"     # 541 rows, 44 cols, sheet "Full_new" + "Instructions" codebook
LIFESTYLE_PATH = RAW_DIR / "PCOS_lifestyle.csv"        # 2000 rows, 44 cols, derived/augmented (see notebook 01)

RANDOM_SEED = 42

TARGET_COL = "PCOS (Y/N)"
PATIENT_ID_COL = "Patient File No."

# Risk-category cut points for the app/API (Phase 9-11). Arbitrary, documented
# demo thresholds chosen for interpretability - NOT clinically validated
# cutoffs, and must never be presented to a user as if they were.
RISK_THRESHOLDS = {
    "lower_upper_bound": 0.3,     # probability < 0.3  -> "Lower"
    "moderate_upper_bound": 0.7,  # 0.3 <= probability < 0.7 -> "Moderate"; >= 0.7 -> "Elevated"
}

for _dir in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, RESULTS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
