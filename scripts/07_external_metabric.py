"""PHASE 7 / GATE 7 -- external validation in METABRIC.

KNOWN CONSTRAINT, not worked around: METABRIC has no matched Illumina 450k
methylation. Only the EXPRESSION component of the signature can be tested
externally. (cBioPortal does carry a 'methylation_promoters_rrbs' profile for
METABRIC, but RRBS on a subset is a different assay from 450k and is not a
substitute -- it is not used here.)

Cross-platform transfer: TCGA is RNA-seq log2(count+1), METABRIC is Illumina
HT-12 microarray intensity. Genes are z-scored WITHIN each cohort so the model
sees comparable scales. Degradation relative to internal CV is expected and the
actual number is reported, not smoothed.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import load_processed
from utils import (FilterLog, flag, get_logger, load_config, repo_root,
                   save_json, save_table, set_seed)

warnings.filterwarnings("ignore")
DAYS_PER_MONTH = 30.4375


def load_metabric(cfg, log, flog):
    root = repo_root()
    log.info("reading METABRIC expression...")
    ex = pd.read_csv(root / cfg["files"]["metabric_expr"], sep="\t", low_memory=False)
    ex = ex.rename(columns={ex.columns[0]: "Hugo_Symbol"})
    ex = ex.dropna(subset=["Hugo_Symbol"])
    drop = [c for c in ("Entrez_Gene_Id",) if c in ex.columns]
    ex = ex.drop(columns=drop)
    flog.step("METABRIC expression rows", len(ex), len(ex), "probes/genes")
    ex = ex.groupby("Hugo_Symbol").mean(numeric_only=True)
    flog.step("METABRIC: collapse duplicate symbols (mean)", len(ex), len(ex), "genes")
    ex = ex.T                                        # samples x genes
    log.info("METABRIC expression: %d samples x %d genes", *ex.shape)

    pat = pd.read_csv(root / cfg["files"]["metabric_patient"], sep="\t", comment="#",
                      low_memory=False)
    pat = pat.rename(columns={pat.columns[0]: "PATIENT_ID"}).set_index("PATIENT_ID")
    flog.step("METABRIC clinical rows", len(pat), len(pat), "patients")
    return ex, pat


def metabric_labels(pat, cfg, log, flog):
    h = cfg["label"]["horizon_days"]
    need = ["RFS_STATUS", "RFS_MONTHS"]
    miss = [c for c in need if c not in pat.columns]
    if miss:
        raise SystemExit(f"METABRIC clinical missing {miss}")

    df = pat[need].copy()
    n0 = len(df)
    # "0:Not Recurred" / "1:Recurred"
    df["event"] = pd.to_numeric(df["RFS_STATUS"].astype(str).str.split(":").str[0],
                                errors="coerce")
    df["months"] = pd.to_numeric(df["RFS_MONTHS"], errors="coerce")
    df = df.dropna(subset=["event", "months"])
    flog.step("METABRIC: drop missing RFS status/months", n0, len(df), "patients")
    df["time_days"] = df["months"] * DAYS_PER_MONTH

    ev, t = df["event"].astype(int), df["time_days"]
    pos = (ev == 1) & (t <= h)
    neg = ((ev == 0) & (t >= h)) | ((ev == 1) & (t > h))
    exc = (ev == 0) & (t < h)
    df["label"] = np.where(pos, 1, np.where(neg, 0, -1))
    log.info("METABRIC label at %dd: pos=%d neg=%d excluded_early_censor=%d",
             h, int(pos.sum()), int(neg.sum()), int(exc.sum()))
    lab = df[df["label"] >= 0].copy()
    flog.step(f"METABRIC: EXCLUDE censored before {h}d", len(df), len(lab), "patients")
    flog.note("METABRIC positives", int((lab["label"] == 1).sum()), "patients")
    flog.note("METABRIC negatives", int((lab["label"] == 0).sum()), "patients")
    return lab


def km_curve(time, event, group, label, log):
    kmf = KaplanMeierFitter()
    res = {}
    for g in (0, 1):
        m = group == g
        if m.sum() < 5:
            continue
        kmf.fit(time[m], event[m], label=f"{label}_{'high' if g else 'low'}_risk")
        res[g] = {"n": int(m.sum()), "events": int(event[m].sum()),
                  "median_survival": float(kmf.median_survival_time_)}
    lr = logrank_test(time[group == 0], time[group == 1],
                      event[group == 0], event[group == 1])
    return res, float(lr.p_value), float(lr.test_statistic)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("07_external_metabric", cfg)
    root = repo_root()
    flog = FilterLog(log, "metabric")

    # ---------- TCGA side ----------
    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)

    panels = pd.read_csv(root / cfg["paths"]["tables"] / "signature_panels.csv")
    sizes = sorted(panels["panel_size"].unique())
    log.info("signature panels available: %s", sizes)

    mb_expr, mb_pat = load_metabric(cfg, log, flog)
    mb_lab = metabric_labels(mb_pat, cfg, log, flog)

    common_samples = [s for s in mb_expr.index if s in set(mb_lab.index)]
    flog.step("METABRIC: patients with expression AND a usable label",
              len(mb_lab), len(common_samples), "patients")
    mb_expr = mb_expr.loc[common_samples]
    mb_lab = mb_lab.loc[common_samples]
    y_mb = mb_lab["label"].values.astype(int)
    log.info("METABRIC validation cohort: n=%d, positives=%d (%.1f%%)",
             len(y_mb), y_mb.sum(), 100 * y_mb.mean())

    results, mapping_rows = [], []
    best_scores = None
    for N in sizes:
        p = panels[panels["panel_size"] == N]
        expr_genes = sorted(set(p[p["block"] == "expr"]["gene"]))
        n_meth = int((p["block"] == "meth").sum())
        n_clin = int((p["block"] == "clin").sum())
        mapped = [g for g in expr_genes if g in mb_expr.columns and g in expr.columns]
        mapping_rows.append({
            "panel_size": N, "expression_markers": len(expr_genes),
            "methylation_markers_dropped": n_meth, "clinical_markers": n_clin,
            "expression_markers_mapped_to_METABRIC": len(mapped),
            "mapping_rate": round(len(mapped) / len(expr_genes), 3) if expr_genes else np.nan,
        })
        log.info("panel %4d: %d expression markers, %d map to METABRIC (%.0f%%); "
                 "%d methylation markers CANNOT be tested externally",
                 N, len(expr_genes), len(mapped),
                 100 * len(mapped) / max(len(expr_genes), 1), n_meth)
        if len(mapped) < 5:
            log.warning("  panel %d: too few mapped genes to model", N)
            continue

        # z-score within each cohort, independently
        Xtr = StandardScaler().fit_transform(expr[mapped].values)
        Xte = StandardScaler().fit_transform(mb_expr[mapped].values)

        pos, neg = int(y.sum()), int(len(y) - y.sum())
        for mname, model in (
            ("xgboost", XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                                      tree_method="hist", scale_pos_weight=neg / max(pos, 1),
                                      max_depth=3, learning_rate=0.05, n_estimators=400,
                                      subsample=0.8, colsample_bytree=0.6,
                                      random_state=cfg["seed"], n_jobs=8, verbosity=0)),
            ("logreg_l2", LogisticRegression(max_iter=5000, class_weight="balanced",
                                             C=0.1, random_state=cfg["seed"])),
        ):
            model.fit(Xtr, y)
            s_mb = model.predict_proba(Xte)[:, 1]
            s_tcga_insample = model.predict_proba(Xtr)[:, 1]
            row = {
                "panel_size": N, "model": mname, "n_genes_used": len(mapped),
                "metabric_auc": round(roc_auc_score(y_mb, s_mb), 4),
                "metabric_ap": round(average_precision_score(y_mb, s_mb), 4),
                "tcga_in_sample_auc": round(roc_auc_score(y, s_tcga_insample), 4),
            }
            results.append(row)
            log.info("   %-10s METABRIC AUC=%.4f AP=%.4f  (TCGA in-sample %.4f, "
                     "optimistic by construction)",
                     mname, row["metabric_auc"], row["metabric_ap"],
                     row["tcga_in_sample_auc"])
            if N == 50 and mname == "xgboost":
                best_scores = s_mb

    res = pd.DataFrame(results)
    save_table(res, cfg, "gate7_metabric_performance.csv", log)
    save_table(pd.DataFrame(mapping_rows), cfg, "gate7_gene_mapping.csv", log)
    save_table(flog.to_frame(), cfg, "gate7_metabric_filtering.csv", log)

    # ---------- Kaplan-Meier ----------
    km_out = {}
    if best_scores is not None:
        grp = (best_scores > np.median(best_scores)).astype(int)
        curves, p, stat = km_curve(mb_lab["time_days"].values,
                                   mb_lab["event"].astype(int).values, grp, "METABRIC", log)
        km_out["metabric"] = {"groups": curves, "logrank_p": p, "logrank_stat": stat}
        log.info("METABRIC KM (50-marker panel, median split): log-rank p=%.3g", p)
        pd.DataFrame({"patient": mb_lab.index, "time_days": mb_lab["time_days"],
                      "event": mb_lab["event"].astype(int), "risk_score": best_scores,
                      "risk_group": grp, "label": y_mb}) \
            .to_csv(root / cfg["paths"]["tables"] / "gate7_metabric_km_data.csv", index=False)

    oof_p = root / cfg["paths"]["tables"] / "model_b_oof_predictions.csv"
    if oof_p.exists():
        oof = pd.read_csv(oof_p).set_index("patient").reindex(lab.index)
        grp = (oof["risk_oof"].values > np.nanmedian(oof["risk_oof"].values)).astype(int)
        curves, p, stat = km_curve(lab["PFI.time"].values.astype(float),
                                   lab["PFI"].values.astype(int), grp, "TCGA", log)
        km_out["tcga"] = {"groups": curves, "logrank_p": p, "logrank_stat": stat}
        log.info("TCGA KM (out-of-fold Model B risk, median split): log-rank p=%.3g", p)
        pd.DataFrame({"patient": lab.index, "time_days": lab["PFI.time"].values,
                      "event": lab["PFI"].values, "risk_score": oof["risk_oof"].values,
                      "risk_group": grp, "label": lab["label"].values}) \
            .to_csv(root / cfg["paths"]["tables"] / "gate7_tcga_km_data.csv", index=False)

    best = res.sort_values("metabric_auc", ascending=False).iloc[0] if len(res) else None
    save_json({"metabric_n": int(len(y_mb)), "metabric_positives": int(y_mb.sum()),
               "metabric_positive_rate": round(float(y_mb.mean()), 4),
               "best_metabric_auc": float(best["metabric_auc"]) if best is not None else None,
               "best_panel_size": int(best["panel_size"]) if best is not None else None,
               "best_model": str(best["model"]) if best is not None else None,
               "km": km_out,
               "methylation_limitation": "METABRIC has no matched Illumina 450k methylation; "
                                         "only the expression component of the signature was "
                                         "externally validated."},
              cfg, "gate7_summary.json", log)

    if best is not None:
        log.info("GATE 7: best METABRIC AUC = %.4f (panel %d, %s)",
                 best["metabric_auc"], best["panel_size"], best["model"])
        if best["metabric_auc"] < 0.55:
            flag(log, f"METABRIC AUC {best['metabric_auc']:.3f} is near chance. "
                      "Cross-platform transfer essentially failed -- report it as such.")


if __name__ == "__main__":
    main()
