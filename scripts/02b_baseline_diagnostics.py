"""GATE 2 follow-up -- why is the clinical baseline this strong?

Model A reached AUC 0.779, above the 0.60-0.70 band the spec expects. That is
below the 0.80 "something leaked" threshold but still needs an explanation.
Two candidate causes are tested here:

  1. PAM50 is EXPRESSION-DERIVED. Including it makes "clinical" baseline partly
     an omics model and inflates the bar unfairly.
  2. Stage IV patients are metastatic AT DIAGNOSIS. A PFI event for them is
     progression of known metastatic disease, not relapse from remission.
     Their outcome is close to deterministic given stage.

Reports univariate AUCs and refits Model A under each restriction, on the same
shared folds.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, summarise
from folds import build_or_load_folds
from utils import get_logger, load_config, save_table, set_seed


def run_variant(X, y, folds, cfg, name, log):
    rows = []
    for i, (tr, te) in enumerate(folds):
        if len(np.unique(y[tr])) < 2:
            continue
        pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=cfg["seed"]))])
        pipe.fit(X[tr], y[tr])
        m = fold_metrics(y[te], pipe.predict_proba(X[te])[:, 1])
        m["fold_id"] = i
        rows.append(m)
    df = pd.DataFrame(rows)
    s = summarise(df, name)
    log.info("%-42s AUC=%.4f [%.4f, %.4f]  AP=%.4f  (%d features)",
             name, s["auc_mean"], s["auc_ci_lo"], s["auc_ci_hi"], s["ap_mean"], X.shape[1])
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("02b_baseline_diagnostics", cfg)

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)
    folds = build_or_load_folds(y, cfg, log)
    X, names = clinical_matrix(clin, log)
    names_arr = np.array(names)

    # ---- 1. stage composition ----
    stage = pd.to_numeric(clin["stage"], errors="coerce")
    tab = pd.crosstab(stage.fillna(-1), y, dropna=False)
    tab.columns = [f"label_{c}" for c in tab.columns]
    tab["n"] = tab.sum(axis=1)
    tab["relapse_rate"] = (tab.get("label_1", 0) / tab["n"]).round(3)
    tab.index.name = "stage (-1 = missing)"
    log.info("stage composition vs 3-year relapse label:\n%s", tab.to_string())
    save_table(tab.reset_index(), cfg, "gate2_stage_composition.csv", log)

    # ---- 2. univariate AUC per clinical feature ----
    uni = []
    for j, nm in enumerate(names):
        v = X[:, j].astype(float)
        ok = ~np.isnan(v)
        if len(np.unique(y[ok])) < 2 or np.nanstd(v) == 0:
            continue
        a = roc_auc_score(y[ok], v[ok])
        uni.append({"feature": nm, "auc": round(max(a, 1 - a), 4),
                    "direction": "higher=relapse" if a >= 0.5 else "higher=relapse-free"})
    uni_df = pd.DataFrame(uni).sort_values("auc", ascending=False)
    log.info("univariate AUC of each clinical covariate:\n%s", uni_df.to_string(index=False))
    save_table(uni_df, cfg, "gate2_univariate_auc.csv", log)

    # ---- 3. variants ----
    res = []
    res.append(run_variant(X, y, folds, cfg, "A_full (age+stage+PAM50)", log))

    is_pam = np.array([n.startswith("pam50_") for n in names])
    if is_pam.any():
        res.append(run_variant(X[:, ~is_pam], y, folds, cfg,
                               "A_no_PAM50 (age+stage only, TRUE clinical)", log))
        res.append(run_variant(X[:, is_pam], y, folds, cfg, "A_PAM50_only", log))

    for nm in ("age", "stage"):
        if nm in names_arr:
            j = int(np.where(names_arr == nm)[0][0])
            res.append(run_variant(X[:, [j]], y, folds, cfg, f"A_{nm}_only", log))

    # ---- 4. exclude stage IV (metastatic at diagnosis) ----
    m4 = (stage.values == 4)
    log.info("stage IV at diagnosis: %d patients, %d of them relapse-positive",
             int(m4.sum()), int(y[m4].sum()))
    if m4.sum() > 0:
        keep = ~m4
        idx_map = {old: new for new, old in enumerate(np.where(keep)[0])}
        sub_folds = [(np.array([idx_map[i] for i in tr if i in idx_map]),
                      np.array([idx_map[i] for i in te if i in idx_map]))
                     for tr, te in folds]
        res.append(run_variant(X[keep], y[keep], sub_folds, cfg,
                               "A_full_excl_stageIV", log))
        res.append(run_variant(X[keep][:, ~is_pam], y[keep], sub_folds, cfg,
                               "A_no_PAM50_excl_stageIV", log))

    out = pd.DataFrame(res)
    save_table(out, cfg, "gate2_baseline_variants.csv", log)
    log.info("GATE 2 DIAGNOSTIC COMPLETE -- see gate2_baseline_variants.csv")


if __name__ == "__main__":
    main()
