"""Fold-wise metric collection and comparison helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def fold_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    out = {}
    # A fold with a single class present cannot yield an AUC.
    if len(np.unique(y_true)) < 2:
        return {"auc": np.nan, "ap": np.nan, "n": len(y_true), "n_pos": int(y_true.sum())}
    out["auc"] = roc_auc_score(y_true, y_score)
    out["ap"] = average_precision_score(y_true, y_score)
    out["n"] = len(y_true)
    out["n_pos"] = int(y_true.sum())
    return out


def summarise(df: pd.DataFrame, model: str) -> dict:
    """Mean and 95% CI across outer folds (normal approximation on fold means)."""
    res = {"model": model, "n_folds": int(df["auc"].notna().sum())}
    for m in ("auc", "ap"):
        v = df[m].dropna().values
        mean = float(np.mean(v))
        se = float(stats.sem(v)) if len(v) > 1 else 0.0
        half = 1.96 * se
        res[f"{m}_mean"] = round(mean, 4)
        res[f"{m}_sd"] = round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4)
        res[f"{m}_ci_lo"] = round(mean - half, 4)
        res[f"{m}_ci_hi"] = round(mean + half, 4)
    return res


def paired_fold_test(a: pd.DataFrame, b: pd.DataFrame, metric: str = "auc") -> dict:
    """Paired comparison on identical folds -- valid only because folds match."""
    m = a[["fold_id", metric]].merge(b[["fold_id", metric]], on="fold_id",
                                     suffixes=("_a", "_b")).dropna()
    if len(m) < 3:
        return {"n_folds": len(m), "mean_diff": np.nan, "p_value": np.nan}
    d = m[f"{metric}_b"] - m[f"{metric}_a"]
    t, p = stats.ttest_rel(m[f"{metric}_b"], m[f"{metric}_a"])
    try:
        _, pw = stats.wilcoxon(m[f"{metric}_b"], m[f"{metric}_a"])
    except ValueError:
        pw = np.nan
    return {"n_folds": int(len(m)), "mean_diff": round(float(d.mean()), 4),
            "t_stat": round(float(t), 3), "p_value": float(p),
            "wilcoxon_p": float(pw), "b_wins_folds": int((d > 0).sum())}
