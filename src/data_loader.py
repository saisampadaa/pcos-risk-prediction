"""
Safe loaders for the three raw PCOS data files.

Design rules:
- Never write to anything under data/raw. These functions only read.
- Do not silently "fix" dirty values here (e.g. the stray "a" in AMH, or the
  trailing "." in some beta-HCG values). Loading and cleaning are separate
  concerns; cleaning decisions belong in the preprocessing pipeline (Phase 3)
  where they can be documented and applied consistently to train/test splits.
- Column names are returned exactly as they appear in the source files.
  Standardization happens in a separate, explicit step (see
  `data_validation.build_standardized_columns`) so the mapping is always
  visible and reversible.
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from src.config import CLINICAL_1_PATH, CLINICAL_2_PATH, LIFESTYLE_PATH


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected data file not found: {path}\n"
            "Check that data/raw/ contains the original files and that you are "
            "running from the project root."
        )


def load_clinical_1() -> pd.DataFrame:
    """Load PCOS_clinical_1.csv: 541 rows, 6 columns (infertility-workup subset).

    Columns: Sl. No, Patient File No. (10001-10541), PCOS (Y/N),
    I beta-HCG(mIU/mL), II beta-HCG(mIU/mL), AMH(ng/mL).
    """
    _require_file(CLINICAL_1_PATH)
    return pd.read_csv(CLINICAL_1_PATH)


def load_clinical_2_full() -> pd.DataFrame:
    """Load the 'Full_new' sheet of PCOS_clinical_2.xlsx: 541 rows, 44 columns.

    This is the full-feature clinical dataset (same 541 patients as
    PCOS_clinical_1.csv, Patient File No. 1-541 instead of 10001-10541).
    Drops the trailing 'Unnamed: 44' column: it is an Excel artifact with only
    2 stray non-numeric values ("." and "7") out of 541 rows, not a real
    feature (see notebooks/01_dataset_audit.ipynb).
    """
    _require_file(CLINICAL_2_PATH)
    df = pd.read_excel(CLINICAL_2_PATH, sheet_name="Full_new")
    return df.drop(columns=["Unnamed: 44"], errors="ignore")


def load_clinical_2_instructions() -> pd.DataFrame:
    """Load the 'Instructions' sheet of PCOS_clinical_2.xlsx (data-collection codebook).

    Not a data table — documents units, Yes/No encoding (Yes=1, No=0), and the
    Blood Group code map (A+=11, A-=12, B+=13, B-=14, O+=15, O-=16, AB+=17, AB-=18).
    """
    _require_file(CLINICAL_2_PATH)
    return pd.read_excel(CLINICAL_2_PATH, sheet_name="Instructions", header=None)


def load_lifestyle() -> pd.DataFrame:
    """Load PCOS_lifestyle.csv: 2000 rows, 44 columns.

    See notebooks/01_dataset_audit.ipynb, section "Patient linkage check":
    this file is not 2000 independent patients. Its Patient File No. values
    only range 1-541 and repeat, and repeated rows for the same Patient File
    No. share the same Age and PCOS label but have small (jittered)
    differences in Weight/Height/BMI relative to the matching patient in
    PCOS_clinical_2.xlsx. Treat it as an expanded/resampled version of the
    541-patient population, not an independent sample.
    """
    _require_file(LIFESTYLE_PATH)
    return pd.read_csv(LIFESTYLE_PATH)


def load_all() -> Dict[str, pd.DataFrame]:
    """Load all three raw datasets into a single dict, keyed by short name."""
    return {
        "clinical_1": load_clinical_1(),
        "clinical_2_full": load_clinical_2_full(),
        "lifestyle": load_lifestyle(),
    }
