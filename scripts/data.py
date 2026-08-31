"""Loading of processed matrices, shared by all model scripts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils import repo_root


def load_processed(cfg: dict):
    p = repo_root() / cfg["paths"]["processed"]
    expr = pd.read_parquet(p / "expression.parquet")
    met = pd.read_parquet(p / "methylation.parquet")
    clin = pd.read_parquet(p / "clinical.parquet").set_index("patient")
    lab = pd.read_parquet(p / "labels.parquet").set_index("patient")
    order = list(lab.index)
    return expr.loc[order], met.loc[order], clin.loc[order], lab.loc[order]


def clinical_matrix(clin: pd.DataFrame, logger=None, max_missing: float = 0.5):
    """Numeric design matrix from clinical covariates.

    Ordinal: age, stage, grade. Categorical: PAM50 (one-hot).
    Columns missing in >max_missing of patients are dropped and logged --
    TCGA-BRCA in particular records almost no histological grade.
    """
    cols, names = [], []
    for c in ("age", "stage", "grade"):
        if c not in clin.columns:
            continue
        v = pd.to_numeric(clin[c], errors="coerce")
        miss = v.isna().mean()
        if miss > max_missing:
            if logger:
                logger.info("clinical: dropping '%s' (%.0f%% missing)", c, 100 * miss)
            continue
        if logger:
            logger.info("clinical: keeping '%s' (%.1f%% missing, median-imputed)", c, 100 * miss)
        cols.append(v.values.astype(float))
        names.append(c)

    if "pam50" in clin.columns:
        pam = clin["pam50"].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
        miss = (pam == "Unknown").mean()
        if miss <= max_missing:
            d = pd.get_dummies(pam, prefix="pam50")
            if logger:
                logger.info("clinical: keeping PAM50 one-hot (%d levels, %.1f%% unknown)",
                            d.shape[1], 100 * miss)
            for c in d.columns:
                cols.append(d[c].values.astype(float))
                names.append(c)
        elif logger:
            logger.info("clinical: dropping PAM50 (%.0f%% unknown)", 100 * miss)

    X = np.column_stack(cols) if cols else np.zeros((len(clin), 0))
    return X, names
