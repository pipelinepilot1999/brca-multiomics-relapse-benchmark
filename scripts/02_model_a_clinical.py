"""PHASE 2 / GATE 2 -- Model A: clinical-only logistic regression baseline.

This is the bar every omics model has to clear. Clinical-only models in BRCA
typically land at AUC 0.60-0.70; anything above 0.80 means something leaked.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, summarise
from folds import build_or_load_folds
from utils import flag, get_logger, load_config, save_json, save_table, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("02_model_a_clinical", cfg)

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)
    log.info("cohort: n=%d, positives=%d (%.1f%%)", len(y), y.sum(), 100 * y.mean())

    X, names = clinical_matrix(clin, log)
    log.info("Model A design matrix: %d patients x %d clinical features: %s",
             X.shape[0], X.shape[1], names)
    if X.shape[1] == 0:
        raise SystemExit("no usable clinical covariates")

    folds = build_or_load_folds(y, cfg, log)

    rows = []
    for i, (tr, te) in enumerate(folds):
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=cfg["seed"])),
        ])
        pipe.fit(X[tr], y[tr])
        s = pipe.predict_proba(X[te])[:, 1]
        m = fold_metrics(y[te], s)
        m.update({"fold_id": i, "repeat": i // cfg["cv"]["n_splits"],
                  "fold": i % cfg["cv"]["n_splits"]})
        rows.append(m)
        if i % 10 == 0:
            log.info("  fold %2d/%d auc=%.3f ap=%.3f", i + 1, len(folds), m["auc"], m["ap"])

    per_fold = pd.DataFrame(rows)
    save_table(per_fold, cfg, "model_a_per_fold.csv", log)

    summ = summarise(per_fold, "A_clinical")
    summ["features"] = ";".join(names)
    summ["n_features"] = len(names)
    save_json(summ, cfg, "model_a_summary.json", log)

    # out-of-fold predictions, averaged over repeats, for downstream use
    log.info("GATE 2: Model A AUC = %.4f [%.4f, %.4f]  AP = %.4f [%.4f, %.4f]  over %d folds",
             summ["auc_mean"], summ["auc_ci_lo"], summ["auc_ci_hi"],
             summ["ap_mean"], summ["ap_ci_lo"], summ["ap_ci_hi"], summ["n_folds"])

    if summ["auc_mean"] > 0.80:
        flag(log, f"Model A AUC {summ['auc_mean']:.3f} > 0.80 on clinical covariates alone. "
                  "Spec says find the leak before continuing.")
    elif not (0.55 <= summ["auc_mean"] <= 0.75):
        flag(log, f"Model A AUC {summ['auc_mean']:.3f} is outside the usual 0.60-0.70 band for "
                  "clinical-only BRCA relapse models. Worth explaining.")


if __name__ == "__main__":
    main()
