"""PHASE 3 / GATE 3 -- Model B: XGBoost on clinical + expression + methylation.

Leakage control: univariate feature selection lives INSIDE a sklearn Pipeline,
so it is refit on the training half of every split (outer and inner). Selecting
features on the full dataset before CV is the single most common way this
project produces a fake AUC.

Nested CV: inner RandomizedSearchCV for hyper-parameters, outer folds (shared
with Models A and C) for performance. SHAP values are accumulated out-of-fold.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, summarise
from folds import build_or_load_folds
from utils import flag, get_logger, load_config, save_json, save_table, set_seed

warnings.filterwarnings("ignore", category=UserWarning)


def build_feature_matrix(expr, met, clin, log):
    """Concatenate the three blocks with prefixed, traceable feature names."""
    Xc, cnames = clinical_matrix(clin, log)
    blocks, names, block_of = [], [], []
    if Xc.shape[1]:
        blocks.append(Xc)
        names += [f"clin:{c}" for c in cnames]
        block_of += ["clinical"] * Xc.shape[1]
    blocks.append(expr.values.astype(np.float32))
    names += [f"expr:{g}" for g in expr.columns]
    block_of += ["expression"] * expr.shape[1]
    blocks.append(met.values.astype(np.float32))
    names += [f"meth:{g}" for g in met.columns]
    block_of += ["methylation"] * met.shape[1]

    X = np.hstack(blocks).astype(np.float32)
    log.info("Model B matrix: %d patients x %d features (clinical=%d, expression=%d, methylation=%d)",
             X.shape[0], X.shape[1], Xc.shape[1], expr.shape[1], met.shape[1])
    return X, names, np.array(block_of)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fast", action="store_true", help="smoke mode: fewer folds/candidates")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("03_model_b_xgboost", cfg)

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)
    X, names, block_of = build_feature_matrix(expr, met, clin, log)
    names_arr = np.array(names)

    folds = build_or_load_folds(y, cfg, log)
    if args.fast:
        folds = folds[: cfg["cv"]["n_splits"]]
        log.info("FAST mode: using %d folds", len(folds))

    k = min(cfg["model_b"]["select_k"], X.shape[1])
    grid = {f"clf__{p}": v for p, v in cfg["model_b"]["param_grid"].items()}
    n_iter = 3 if args.fast else cfg["cv"]["n_random_search"]

    oof = np.full((len(y), len(folds)), np.nan, dtype=np.float32)
    shap_sum = np.zeros(len(names))
    shap_hits = np.zeros(len(names))
    sel_count = np.zeros(len(names))
    rows, best_params = [], []

    for i, (tr, te) in enumerate(folds):
        pos, neg = int(y[tr].sum()), int(len(tr) - y[tr].sum())
        spw = neg / max(pos, 1)

        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("select", SelectKBest(f_classif, k=k)),      # refit per split -> no leakage
            ("clf", XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", scale_pos_weight=spw,
                random_state=cfg["seed"], n_jobs=2, verbosity=0)),
        ])

        inner = StratifiedKFold(n_splits=cfg["cv"]["inner_splits"], shuffle=True,
                                random_state=cfg["seed"] + i)
        search = RandomizedSearchCV(pipe, grid, n_iter=n_iter, scoring="roc_auc",
                                    cv=inner, n_jobs=4, random_state=cfg["seed"] + i,
                                    refit=True, error_score="raise")
        search.fit(X[tr], y[tr])
        best = search.best_estimator_

        s = best.predict_proba(X[te])[:, 1]
        oof[te, i] = s
        m = fold_metrics(y[te], s)
        m.update({"fold_id": i, "repeat": i // cfg["cv"]["n_splits"],
                  "fold": i % cfg["cv"]["n_splits"],
                  "inner_best_auc": round(float(search.best_score_), 4)})
        rows.append(m)
        best_params.append({"fold_id": i, **{kk.replace("clf__", ""): vv
                                             for kk, vv in search.best_params_.items()}})

        # SHAP on this fold's held-out patients, in the fold's selected subspace
        sel_idx = best.named_steps["select"].get_support(indices=True)
        sel_count[sel_idx] += 1
        try:
            import shap
            Xte = best.named_steps["select"].transform(
                best.named_steps["impute"].transform(X[te]))
            expl = shap.TreeExplainer(best.named_steps["clf"])
            sv = expl.shap_values(Xte)
            if isinstance(sv, list):
                sv = sv[-1]
            shap_sum[sel_idx] += np.abs(sv).mean(axis=0)
            shap_hits[sel_idx] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("SHAP failed on fold %d: %s", i, exc)

        if i % 5 == 0 or i == len(folds) - 1:
            log.info("  fold %2d/%d auc=%.3f ap=%.3f (inner best %.3f)",
                     i + 1, len(folds), m["auc"], m["ap"], m["inner_best_auc"])

    per_fold = pd.DataFrame(rows)
    save_table(per_fold, cfg, "model_b_per_fold.csv", log)
    save_table(pd.DataFrame(best_params), cfg, "model_b_best_params.csv", log)

    summ = summarise(per_fold, "B_xgboost_multiomics")
    summ["n_features_total"] = int(X.shape[1])
    summ["select_k"] = int(k)
    save_json(summ, cfg, "model_b_summary.json", log)

    # out-of-fold risk score per patient (mean over the repeats that held them out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        oof_mean = np.nanmean(oof, axis=1)
    pd.DataFrame({"patient": lab.index, "label": y, "risk_oof": oof_mean}) \
        .to_csv(save_table.__globals__["repo_root"]() / cfg["paths"]["tables"]
                / "model_b_oof_predictions.csv", index=False)
    log.info("wrote model_b_oof_predictions.csv")

    # SHAP ranking
    mean_shap = np.divide(shap_sum, np.maximum(shap_hits, 1))
    rank = pd.DataFrame({
        "feature": names_arr,
        "block": block_of,
        "mean_abs_shap": mean_shap,
        "folds_selected": sel_count.astype(int),
        "selection_frequency": sel_count / len(folds),
    }).sort_values("mean_abs_shap", ascending=False)
    save_table(rank, cfg, "model_b_shap_ranking.csv", log)

    log.info("GATE 3: Model B AUC = %.4f [%.4f, %.4f]  AP = %.4f [%.4f, %.4f]",
             summ["auc_mean"], summ["auc_ci_lo"], summ["auc_ci_hi"],
             summ["ap_mean"], summ["ap_ci_lo"], summ["ap_ci_hi"])
    log.info("top 30 features by mean |SHAP|:")
    for j, r in rank.head(30).iterrows():
        log.info("   %-28s %-12s shap=%.5f  selected in %.0f%% of folds",
                 r["feature"], r["block"], r["mean_abs_shap"], 100 * r["selection_frequency"])

    if summ["auc_mean"] > 0.85:
        flag(log, f"Model B AUC {summ['auc_mean']:.3f} > 0.85. Spec says stop and look for leakage.")


if __name__ == "__main__":
    main()
