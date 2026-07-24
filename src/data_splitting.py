"""
Patient-level split and modeling-pool assembly.

Why this file exists (not in the original planned structure): Phase 1 audit
found that `lifestyle.csv` (2000 rows) is a jittered, repeated expansion of
the same 541 patients in `clinical_2` (see notebooks/01_dataset_audit.ipynb).
A plain random row-level split, or treating clinical_2/clinical_1 as an
independent external-validation set, would leak patient identity across
train/test. This module fixes that by splitting on the unique patient ID
*before* any augmented rows are assigned anywhere.

Resulting pools:
- train_pool: every row (original clinical_2 row + all lifestyle copies)
  for patients assigned to train. This is where lifestyle.csv earns its
  keep - extra training volume/noise-robustness for these patients only.
- holdout_validation: only the ORIGINAL, unaugmented clinical_2 row for
  patients assigned to test. This is the honest stand-in for "external
  validation" - real, never-seen-in-training measurements, from the same
  source population (not a different hospital/cohort - documented as a
  limitation, not claimed as independent-population validation).
- holdout_augmented_robustness: the lifestyle.csv jittered copies of the
  SAME held-out test patients. Never used for training or for scoring
  metrics. Only used post-hoc to check whether the frozen model's predicted
  probability stays stable under small measurement perturbations of a
  patient it has never trained on (a robustness diagnostic for Phase 8/9).

Every function here is deterministic given RANDOM_SEED, so the split is
reproducible across notebooks and across re-runs.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PATIENT_ID_COL, TARGET_COL, RANDOM_SEED
from src.feature_mapping import harmonize


def build_patient_split(
    clinical_2_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign each of the 541 unique patients to 'train' or 'test', stratified by PCOS label.

    Returns a small DataFrame: [Patient File No., PCOS (Y/N), split].
    Save this once (data/interim/patient_split.csv) and reuse it everywhere -
    do not regenerate it ad hoc in different notebooks, or the split will
    silently drift between analyses.
    """
    patients = clinical_2_df[[PATIENT_ID_COL, TARGET_COL]].drop_duplicates(subset=PATIENT_ID_COL)

    train_ids, test_ids = train_test_split(
        patients[PATIENT_ID_COL],
        test_size=test_size,
        random_state=seed,
        stratify=patients[TARGET_COL],
    )

    split_map = patients.copy()
    split_map["split"] = split_map[PATIENT_ID_COL].apply(
        lambda pid: "train" if pid in set(train_ids) else "test"
    )
    return split_map.reset_index(drop=True)


def assemble_modeling_pools(
    clinical_2_df: pd.DataFrame,
    lifestyle_df: pd.DataFrame,
    patient_split: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the three pools described in this module's docstring.

    Both source dataframes are harmonized (column renamed, known-dirty values
    coerced to numeric) before assembly, so train_pool and holdout_validation
    share identical column names and dtypes.
    """
    clinical_2_h = harmonize(clinical_2_df)
    lifestyle_h = harmonize(lifestyle_df)

    train_ids = set(patient_split.loc[patient_split["split"] == "train", PATIENT_ID_COL])
    test_ids = set(patient_split.loc[patient_split["split"] == "test", PATIENT_ID_COL])

    clinical_2_train = clinical_2_h[clinical_2_h[PATIENT_ID_COL].isin(train_ids)]
    lifestyle_train = lifestyle_h[lifestyle_h[PATIENT_ID_COL].isin(train_ids)]
    train_pool = pd.concat([clinical_2_train, lifestyle_train], ignore_index=True)

    holdout_validation = clinical_2_h[clinical_2_h[PATIENT_ID_COL].isin(test_ids)].reset_index(drop=True)
    holdout_augmented_robustness = lifestyle_h[lifestyle_h[PATIENT_ID_COL].isin(test_ids)].reset_index(drop=True)

    assert set(train_pool[PATIENT_ID_COL]).isdisjoint(holdout_validation[PATIENT_ID_COL]), (
        "Leakage check failed: a patient appears in both train_pool and holdout_validation."
    )

    return train_pool.reset_index(drop=True), holdout_validation, holdout_augmented_robustness
