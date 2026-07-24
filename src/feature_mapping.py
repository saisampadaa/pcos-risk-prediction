"""
Column-name harmonization and known data-entry-error cleanup, shared by
`clinical_2` and `lifestyle` so both can be safely concatenated into one
modeling pool (see `data_loader.assemble_modeling_pools`).

This is intentionally separate from `preprocessing.py` (Phase 3): this module
only fixes *identification* problems (a column meaning the same thing under
two spellings, a value that is text because of a typo) discovered during the
Phase 1 audit. It does not impute, encode, or scale anything — those are
modeling choices that belong in the preprocessing pipeline and must be fit on
the training split only.
"""

from typing import List

import pandas as pd

# Verified in notebooks/01_dataset_audit.ipynb section 4: this is the only
# column-name mismatch between clinical_2 and lifestyle (a typo in clinical_2
# and its source Instructions sheet, not a different feature).
COLUMN_RENAME_MAP = {
    "No. of aborptions": "No. of abortions",
}

# Columns that load as text (object dtype) because of a small number of
# data-entry errors, verified in notebooks/01_dataset_audit.ipynb section 3:
# AMH(ng/mL) has a stray "a" in a few rows; beta-HCG II has a trailing "."
# in a few rows. Coercing turns those specific bad values into NaN, to be
# handled by the imputer in the Phase 3 pipeline (fit on train only).
NUMERIC_LIKE_TEXT_COLUMNS: List[str] = ["AMH(ng/mL)", "II    beta-HCG(mIU/mL)"]


def harmonize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the verified rename map so column names match across files."""
    return df.rename(columns=COLUMN_RENAME_MAP)


def clean_known_dirty_values(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce the known-dirty numeric-like columns to float, in place semantics avoided (returns a copy).

    Any row-level imputation of the resulting NaNs happens later, in the
    Phase 3 pipeline, fit on the training split only.
    """
    df = df.copy()
    for col in NUMERIC_LIKE_TEXT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    """Full harmonization: rename known-typo columns, then clean known-dirty values."""
    return clean_known_dirty_values(harmonize_columns(df))
