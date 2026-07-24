"""
Reusable dataset-audit functions for Phase 1.

These are intentionally generic (they take any DataFrame + a name string) so
the same functions are reused in the audit notebook, in later EDA, and in
regression checks after the cleaning pipeline is built — instead of copying
inspection code for each dataset.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def audit_dataframe(df: pd.DataFrame, name: str, target_col: Optional[str] = None) -> Dict:
    """Run the standard Phase-1 checks on one dataframe and return them as a dict.

    Prints a short human-readable summary and returns the full detail so it
    can be saved (e.g. json.dump) or compared programmatically.
    """
    summary: Dict = {"name": name, "shape": df.shape}

    missing = df.isna().sum()
    summary["missing_by_column"] = missing[missing > 0].to_dict()
    summary["n_duplicate_rows"] = int(df.duplicated().sum())
    summary["dtypes"] = df.dtypes.astype(str).to_dict()
    summary["n_unique_by_column"] = df.nunique().to_dict()

    numeric_df = df.select_dtypes(include=[np.number])
    summary["numeric_describe"] = numeric_df.describe().T

    if target_col and target_col in df.columns:
        summary["target_balance"] = df[target_col].value_counts().to_dict()

    print(f"=== {name} ===")
    print(f"shape: {summary['shape']}")
    print(f"duplicate rows: {summary['n_duplicate_rows']}")
    print(f"columns with missing values: {summary['missing_by_column'] or 'none'}")
    if target_col and target_col in df.columns:
        print(f"target balance ({target_col}): {summary['target_balance']}")
    print()

    return summary


def find_non_numeric_in_numeric_like(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    """Flag columns that are stored as object/text but should be numeric.

    Returns a table of (column, bad_value, row_count) for every value that
    fails `pd.to_numeric`. Use this before blindly casting a column to float —
    it shows exactly which raw values are breaking the conversion (e.g. a
    stray "a" or a trailing "." typo) instead of silently coercing to NaN.
    """
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad_mask = coerced.isna() & df[col].notna()
        if bad_mask.any():
            for value, count in df.loc[bad_mask, col].value_counts().items():
                rows.append({"column": col, "bad_value": value, "row_count": count})
    return pd.DataFrame(rows)


def compare_columns(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a presence matrix of columns across multiple named datasets.

    Row index = every column name seen in any dataset.
    Columns = dataset names, value = True/False whether that dataset has it.
    Useful for spotting near-duplicate columns caused by typos
    (e.g. 'No. of abortions' vs 'No. of aborptions').
    """
    all_cols: List[str] = sorted({c.strip() for df in dfs.values() for c in df.columns})
    matrix = pd.DataFrame(index=all_cols)
    for name, df in dfs.items():
        stripped_cols = {c.strip() for c in df.columns}
        matrix[name] = [c in stripped_cols for c in all_cols]
    return matrix


def check_id_linkage(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    id_col_a: str,
    id_col_b: str,
    id_offset: int,
    compare_cols: List[str],
) -> pd.DataFrame:
    """Check whether two dataframes describe the same records under an ID offset.

    Merges df_a (with id_col_a - id_offset) against df_b (on id_col_b) and
    returns the merged rows with the requested comparison columns from both,
    so you can visually confirm the values line up record-for-record.
    """
    a = df_a.copy()
    a["_mapped_id"] = a[id_col_a] - id_offset
    merged = a.merge(df_b, left_on="_mapped_id", right_on=id_col_b, suffixes=("_a", "_b"))
    cols = ["_mapped_id"] + [f"{c}_a" for c in compare_cols] + [f"{c}_b" for c in compare_cols]
    cols = [c for c in cols if c in merged.columns]
    return merged[cols]


def check_patient_duplication(
    df_augmented: pd.DataFrame,
    df_base: pd.DataFrame,
    id_col: str,
    compare_cols: List[str],
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Quantify how "augmented" rows relate to their matching base-population row.

    For every id in `df_augmented` that repeats more than once, compares each
    repeat's numeric columns against the single matching row in `df_base`
    (same id). Returns per-id summary stats: repeat count, mean absolute
    difference per column, and whether the label ever disagrees.

    This is the evidence check for whether a "larger" dataset is an
    independent sample or a resampled/perturbed version of a smaller one.
    """
    counts = df_augmented[id_col].value_counts()
    repeated_ids = counts[counts > 1].index

    rows = []
    for pid in repeated_ids:
        aug_rows = df_augmented[df_augmented[id_col] == pid]
        base_row = df_base[df_base[id_col] == pid]
        if base_row.empty:
            continue
        base_row = base_row.iloc[0]

        record = {"id": pid, "n_repeats": len(aug_rows)}
        for col in compare_cols:
            if col not in aug_rows.columns or col not in base_row.index:
                continue
            diffs = (aug_rows[col] - base_row[col]).abs()
            record[f"{col}_mean_abs_diff"] = diffs.mean()

        if label_col and label_col in aug_rows.columns and label_col in base_row.index:
            record["label_mismatch"] = bool((aug_rows[label_col] != base_row[label_col]).any())

        rows.append(record)

    return pd.DataFrame(rows)
