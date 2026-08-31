"""PHASE 9 -- rectifying the three title claims with evidence.

Three questions the main pipeline left unanswered:

  1. "Multi-Omics"  -- does methylation ADD anything over expression alone?
     The main run only showed methylation features get SELECTED, which is not
     the same claim. Answered here by an ablation over feature blocks on the
     SAME outer folds.

  2. "Biomarker Signature" -- (a) does omics add INCREMENTAL value over the
     clinical variables a clinician already has? Model B lost to Model A, but
     that could be overfitting rather than omics being worthless, so the test
     is repeated with a strongly regularised LINEAR model, where useless
     features shrink toward zero and adding them should at worst tie.
     (b) does a LOCKED threshold transfer? Every risk group so far came from a
     median split computed within each cohort, which is not how a real test
     works. Here a cutpoint is frozen in TCGA and applied unchanged to METABRIC.

  3. "for Breast Cancer Relapse" -- already supported; not retested.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, paired_fold_test, summarise
from folds import build_or_load_folds
from utils import get_logger, load_config, repo_root, save_json, save_table, set_seed

warnings.filterwarnings("ignore")

BLOCKS = {
    "clinical":                     ["clin"],
    "expression":                   ["expr"],
    "methylation":                  ["meth"],
    "expression+methylation":       ["expr", "meth"],
    "clinical+expression":          ["clin", "expr"],
    "clinical+methylation":         ["clin", "meth"],
    "clinical+expression+methylation": ["clin", "expr", "meth"],
}


def make_model(kind, y_tr, seed, k):
    if kind == "xgboost":
        pos = int(y_tr.sum()); neg = len(y_tr) - pos
        clf = XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                            tree_method="hist", scale_pos_weight=neg / max(pos, 1),
                            max_depth=3, learning_rate=0.05, n_estimators=400,
                            subsample=0.8, colsample_bytree=0.6,
                            random_state=seed, n_jobs=8, verbosity=0)
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("select", SelectKBest(f_classif, k=k)), ("clf", clf)]
    else:  # regularised linear
        clf = LogisticRegressionCV(Cs=[0.01, 0.1, 1.0], cv=3, scoring="roc_auc",
                                   max_iter=4000, class_weight="balanced",
                                   random_state=seed, n_jobs=4)
        steps = [("impute", SimpleImputer(strategy="median")),
                 ("scale", StandardScaler()),
                 ("select", SelectKBest(f_classif, k=k)), ("clf", clf)]
    return Pipeline(steps)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("10_ablation_increment", cfg)
    root = repo_root()

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)
    Xc, cnames = clinical_matrix(clin, log)

    parts = {"clin": (Xc, [f"clin:{c}" for c in cnames]),
             "expr": (expr.values.astype(np.float32), [f"expr:{g}" for g in expr.columns]),
             "meth": (met.values.astype(np.float32), [f"meth:{g}" for g in met.columns])}
    folds = build_or_load_folds(y, cfg, log)

    # ---------------------------------------------------------------- ablation
    log.info("=" * 66)
    log.info("PART 1 -- ABLATION: does each omics block earn its place?")
    log.info("=" * 66)
    per_fold_store, rows = {}, []
    for name, keys in BLOCKS.items():
        X = np.hstack([parts[k][0] for k in keys]).astype(np.float32)
        for kind in ("logreg_l2", "xgboost"):
            k = min(1000 if kind == "logreg_l2" else 2000, X.shape[1])
            recs = []
            for i, (tr, te) in enumerate(folds):
                m = make_model(kind, y[tr], cfg["seed"] + i, k)
                m.fit(X[tr], y[tr])
                r = fold_metrics(y[te], m.predict_proba(X[te])[:, 1])
                r["fold_id"] = i
                recs.append(r)
            df = pd.DataFrame(recs)
            per_fold_store[(name, kind)] = df
            s = summarise(df, f"{name}|{kind}")
            s.update({"block": name, "model": kind, "n_features": int(X.shape[1])})
            rows.append(s)
            log.info("  %-34s %-10s AUC=%.4f [%.4f,%.4f]  AP=%.4f  (%d features)",
                     name, kind, s["auc_mean"], s["auc_ci_lo"], s["auc_ci_hi"],
                     s["ap_mean"], X.shape[1])
    abl = pd.DataFrame(rows)
    save_table(abl, cfg, "phase9_ablation.csv", log)

    # ---------------------------------------------- key contrasts, paired
    log.info("-" * 66)
    log.info("KEY CONTRASTS (paired on identical folds)")
    contrasts = [
        ("expression", "expression+methylation", "does methylation add to expression?"),
        ("clinical", "clinical+expression", "does expression add to clinical?"),
        ("clinical", "clinical+methylation", "does methylation add to clinical?"),
        ("clinical", "clinical+expression+methylation", "does ANY omics add to clinical?"),
    ]
    crows = []
    for a, b, q in contrasts:
        for kind in ("logreg_l2", "xgboost"):
            if (a, kind) in per_fold_store and (b, kind) in per_fold_store:
                t = paired_fold_test(per_fold_store[(a, kind)], per_fold_store[(b, kind)], "auc")
                t.update({"baseline": a, "augmented": b, "model": kind, "question": q})
                crows.append(t)
                verdict = ("ADDS" if t["mean_diff"] > 0 and t["p_value"] < 0.05
                           else "HURTS" if t["mean_diff"] < 0 and t["p_value"] < 0.05
                           else "no effect")
                log.info("  [%-9s] %-46s dAUC=%+.4f p=%.3g  -> %s",
                         kind, q, t["mean_diff"], t["p_value"], verdict)
    save_table(pd.DataFrame(crows), cfg, "phase9_incremental_value.csv", log)

    # ------------------------------------------------------- locked threshold
    log.info("=" * 66)
    log.info("PART 2 -- LOCKED THRESHOLD: freeze a cutpoint in TCGA, apply to METABRIC")
    log.info("=" * 66)
    panels = pd.read_csv(root / cfg["paths"]["tables"] / "signature_panels.csv")
    p50 = panels[panels["panel_size"] == 50]
    sig_genes = sorted(set(p50[p50["block"] == "expr"]["gene"]))

    mb_km = pd.read_csv(root / cfg["paths"]["tables"] / "gate7_metabric_km_data.csv")
    mbe = pd.read_csv(root / cfg["files"]["metabric_expr"], sep="\t", low_memory=False)
    mbe = mbe.rename(columns={mbe.columns[0]: "Hugo_Symbol"}).dropna(subset=["Hugo_Symbol"])
    mbe = mbe.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in mbe.columns])
    mbe = mbe.groupby("Hugo_Symbol").mean(numeric_only=True).T
    mbe = mbe.loc[[s for s in mb_km["patient"] if s in mbe.index]]
    mapped = [g for g in sig_genes if g in mbe.columns and g in expr.columns]
    log.info("locked panel: %d expression markers, %d map to METABRIC", len(sig_genes), len(mapped))

    Xtr = StandardScaler().fit_transform(expr[mapped].values)
    Xte = StandardScaler().fit_transform(mbe[mapped].values)
    final = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1,
                              random_state=cfg["seed"]).fit(Xtr, y)

    s_tcga = final.predict_proba(Xtr)[:, 1]
    s_mb = final.predict_proba(Xte)[:, 1]
    CUT = float(np.median(s_tcga))            # <-- frozen in TCGA, a single number
    log.info("threshold LOCKED at TCGA median risk = %.6f", CUT)

    mb_lab = mb_km.set_index("patient").loc[mbe.index]
    y_mb = mb_lab["label"].values.astype(int)
    res = {}
    for tag, grp in (("locked_tcga_cutpoint", (s_mb > CUT).astype(int)),
                     ("metabric_own_median", (s_mb > np.median(s_mb)).astype(int))):
        n_hi = int(grp.sum()); n_lo = int((1 - grp).sum())
        if n_hi < 5 or n_lo < 5:
            log.warning("  %s: degenerate split (%d high / %d low)", tag, n_hi, n_lo)
            res[tag] = {"n_high": n_hi, "n_low": n_lo, "degenerate": True}
            continue
        lr = logrank_test(mb_lab["time_days"][grp == 0], mb_lab["time_days"][grp == 1],
                          mb_lab["event"][grp == 0], mb_lab["event"][grp == 1])
        rr_hi = float(y_mb[grp == 1].mean()); rr_lo = float(y_mb[grp == 0].mean())
        res[tag] = {"n_high": n_hi, "n_low": n_lo,
                    "relapse_rate_high": round(rr_hi, 4), "relapse_rate_low": round(rr_lo, 4),
                    "risk_ratio": round(rr_hi / rr_lo, 2) if rr_lo else None,
                    "logrank_p": float(lr.p_value)}
        log.info("  %-22s high=%4d (%.1f%% relapse)  low=%4d (%.1f%% relapse)  "
                 "RR=%.2f  log-rank p=%.3g", tag, n_hi, 100 * rr_hi, n_lo, 100 * rr_lo,
                 res[tag]["risk_ratio"] or float("nan"), lr.p_value)

    res["locked_cutpoint_value"] = CUT
    res["n_markers_locked"] = len(mapped)
    res["tcga_score_median"] = float(np.median(s_tcga))
    res["metabric_score_median"] = float(np.median(s_mb))
    res["score_shift"] = round(float(np.median(s_mb) - np.median(s_tcga)), 4)
    log.info("score distribution shift TCGA->METABRIC: median %.4f -> %.4f (delta %+.4f)",
             np.median(s_tcga), np.median(s_mb), res["score_shift"])
    save_json(res, cfg, "phase9_locked_threshold.json", log)

    best = abl.sort_values("auc_mean", ascending=False).iloc[0]
    log.info("=" * 66)
    log.info("PHASE 9 DONE. Best block overall: %s (%s) AUC=%.4f",
             best["block"], best["model"], best["auc_mean"])


if __name__ == "__main__":
    main()
