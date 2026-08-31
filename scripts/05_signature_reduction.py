"""PHASE 5 / GATE 5 -- reduce to a ranked, interpretable panel.

The deliverable is a signature, not just a model: a performance-versus-panel-size
curve plus the marker list at each size.

Leakage control matters as much here as in Phase 3. The panel at size N is NOT
chosen from a global ranking and then re-scored -- that would score a panel that
already saw the test folds. Instead, inside each fold the top-N features are
re-selected from the training half only. The REPORTED panel is the consensus:
the N features selected most often across folds.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, summarise
from folds import build_or_load_folds
from utils import get_logger, load_config, repo_root, save_json, save_table, set_seed

warnings.filterwarnings("ignore")


def pick_best_model(cfg, log) -> str:
    """Choose the ranking source by measured outer-fold AUC, not by assumption."""
    tdir = repo_root() / cfg["paths"]["tables"]
    best, best_auc = None, -np.inf
    for tag, f in (("B", "model_b_summary.json"), ("C", "model_c_summary.json")):
        p = tdir / f
        if p.exists():
            s = json.load(open(p))
            log.info("  candidate model %s: AUC=%.4f", tag, s["auc_mean"])
            if s["auc_mean"] > best_auc:
                best, best_auc = tag, s["auc_mean"]
    log.info("best-performing omics model: %s (AUC=%.4f)", best, best_auc)
    return best or "B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("05_signature_reduction", cfg)

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)

    Xc, cnames = clinical_matrix(clin, log)
    X = np.hstack([Xc, expr.values, met.values]).astype(np.float32)
    names = np.array([f"clin:{c}" for c in cnames]
                     + [f"expr:{g}" for g in expr.columns]
                     + [f"meth:{g}" for g in met.columns])
    log.info("feature pool: %d", len(names))

    pick_best_model(cfg, log)
    folds = build_or_load_folds(y, cfg, log)
    if args.fast:
        folds = folds[: cfg["cv"]["n_splits"]]

    sizes = cfg["signature"]["panel_sizes"]
    curve, panels = [], {}

    for N in sizes:
        counts = np.zeros(len(names))
        rows = []
        for i, (tr, te) in enumerate(folds):
            pos = int(y[tr].sum()); neg = len(tr) - pos
            pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("select", SelectKBest(f_classif, k=min(N, X.shape[1]))),
                ("clf", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                      tree_method="hist", scale_pos_weight=neg / max(pos, 1),
                                      max_depth=3, learning_rate=0.05, n_estimators=400,
                                      subsample=0.8, colsample_bytree=0.6,
                                      random_state=cfg["seed"], n_jobs=8, verbosity=0)),
            ])
            pipe.fit(X[tr], y[tr])
            counts[pipe.named_steps["select"].get_support(indices=True)] += 1
            m = fold_metrics(y[te], pipe.predict_proba(X[te])[:, 1])
            m["fold_id"] = i
            rows.append(m)

        df = pd.DataFrame(rows)
        s = summarise(df, f"panel_{N}")
        s["panel_size"] = N
        curve.append(s)
        log.info("panel %4d: AUC=%.4f [%.4f, %.4f]  AP=%.4f",
                 N, s["auc_mean"], s["auc_ci_lo"], s["auc_ci_hi"], s["ap_mean"])

        order = np.argsort(counts)[::-1][:N]
        panels[N] = pd.DataFrame({
            "panel_size": N,
            "rank": np.arange(1, len(order) + 1),
            "feature": names[order],
            "block": [f.split(":")[0] for f in names[order]],
            "gene": [f.split(":", 1)[1] for f in names[order]],
            "fold_selection_frequency": (counts[order] / len(folds)).round(3),
        })

    curve_df = pd.DataFrame(curve)
    save_table(curve_df, cfg, "signature_panel_curve.csv", log)
    all_panels = pd.concat(panels.values(), ignore_index=True)
    save_table(all_panels, cfg, "signature_panels.csv", log)

    # plateau: smallest panel within 1 SE of the best mean AUC
    best_i = int(np.argmax(curve_df["auc_mean"]))
    thresh = curve_df["auc_mean"].iloc[best_i] - curve_df["auc_sd"].iloc[best_i] / np.sqrt(len(folds))
    plateau = int(curve_df[curve_df["auc_mean"] >= thresh]["panel_size"].min())
    log.info("GATE 5: best panel = %d (AUC %.4f); plateau (within 1 SE) begins at %d markers",
             int(curve_df["panel_size"].iloc[best_i]), curve_df["auc_mean"].iloc[best_i], plateau)

    p50 = panels.get(50)
    if p50 is not None:
        comp = p50["block"].value_counts().to_dict()
        log.info("50-marker panel composition: %s", comp)
        log.info("50-marker panel:")
        for _, r in p50.iterrows():
            log.info("   %2d. %-24s %-12s (selected in %.0f%% of folds)",
                     r["rank"], r["gene"], r["block"], 100 * r["fold_selection_frequency"])

    save_json({"plateau_panel_size": plateau,
               "best_panel_size": int(curve_df["panel_size"].iloc[best_i]),
               "best_auc": float(curve_df["auc_mean"].iloc[best_i]),
               "panel_50_composition": (p50["block"].value_counts().to_dict()
                                        if p50 is not None else {})},
              cfg, "signature_plateau.json", log)


if __name__ == "__main__":
    main()
