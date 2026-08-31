"""Locked-threshold transfer, tested across EVERY panel size.

Phase 9 tested threshold-locking with the 50-marker panel only -- which Gate 7
had already shown to be one of the worst-transferring configurations. That
conflates two different questions:

    (a) does this PANEL carry signal into METABRIC?
    (b) does a FIXED CUTPOINT survive the cohort change?

Testing (b) on a panel that fails (a) cannot answer (b). So every panel size is
run here, and for each the frozen-cutpoint split is compared against METABRIC's
own median. If the two agree, the cutpoint transferred and only panel quality
varies -- which is the deployable-test question that actually matters.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from data import load_processed
from utils import get_logger, load_config, repo_root, save_json, save_table, set_seed

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("12_locked_threshold", cfg)
    root = repo_root()

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)

    mb_km = pd.read_csv(root / cfg["paths"]["tables"] / "gate7_metabric_km_data.csv")
    mbe = pd.read_csv(root / cfg["files"]["metabric_expr"], sep="\t", low_memory=False)
    mbe = mbe.rename(columns={mbe.columns[0]: "Hugo_Symbol"}).dropna(subset=["Hugo_Symbol"])
    mbe = mbe.drop(columns=[c for c in ("Entrez_Gene_Id",) if c in mbe.columns])
    mbe = mbe.groupby("Hugo_Symbol").mean(numeric_only=True).T
    mbe = mbe.loc[[s for s in mb_km["patient"] if s in mbe.index]]
    mb_lab = mb_km.set_index("patient").loc[mbe.index]
    y_mb = mb_lab["label"].values.astype(int)
    log.info("METABRIC: n=%d, %d positive (%.1f%%)", len(y_mb), y_mb.sum(), 100 * y_mb.mean())

    panels = pd.read_csv(root / cfg["paths"]["tables"] / "signature_panels.csv")
    rows = []
    for N in sorted(panels["panel_size"].unique()):
        p = panels[panels["panel_size"] == N]
        genes = sorted(set(p[p["block"] == "expr"]["gene"]))
        mapped = [g for g in genes if g in mbe.columns and g in expr.columns]
        if len(mapped) < 5:
            log.warning("panel %d: only %d mapped genes, skipped", N, len(mapped))
            continue

        Xtr = StandardScaler().fit_transform(expr[mapped].values)
        Xte = StandardScaler().fit_transform(mbe[mapped].values)
        model = LogisticRegression(max_iter=5000, class_weight="balanced", C=0.1,
                                   random_state=cfg["seed"]).fit(Xtr, y)
        s_tcga = model.predict_proba(Xtr)[:, 1]
        s_mb = model.predict_proba(Xte)[:, 1]
        CUT = float(np.median(s_tcga))            # frozen in TCGA: one number
        auc = roc_auc_score(y_mb, s_mb)

        rec = {"panel_size": int(N), "n_genes": len(mapped),
               "metabric_auc": round(float(auc), 4),
               "locked_cutpoint": round(CUT, 5),
               "tcga_median_score": round(float(np.median(s_tcga)), 4),
               "metabric_median_score": round(float(np.median(s_mb)), 4),
               "score_shift": round(float(np.median(s_mb) - np.median(s_tcga)), 4)}

        for tag, grp in (("locked", (s_mb > CUT).astype(int)),
                         ("own_median", (s_mb > np.median(s_mb)).astype(int))):
            n_hi, n_lo = int(grp.sum()), int((1 - grp).sum())
            if n_hi < 5 or n_lo < 5:
                rec[f"{tag}_logrank_p"] = np.nan
                rec[f"{tag}_note"] = f"degenerate ({n_hi}/{n_lo})"
                continue
            lr = logrank_test(mb_lab["time_days"][grp == 0], mb_lab["time_days"][grp == 1],
                              mb_lab["event"][grp == 0], mb_lab["event"][grp == 1])
            rr_hi = float(y_mb[grp == 1].mean()); rr_lo = float(y_mb[grp == 0].mean())
            rec[f"{tag}_n_high"] = n_hi
            rec[f"{tag}_relapse_high"] = round(rr_hi, 4)
            rec[f"{tag}_relapse_low"] = round(rr_lo, 4)
            rec[f"{tag}_risk_ratio"] = round(rr_hi / rr_lo, 2) if rr_lo else None
            rec[f"{tag}_logrank_p"] = float(lr.p_value)
        rows.append(rec)
        log.info("panel %4d (%3d genes) AUC=%.4f | LOCKED cut=%.4f p=%-9.3g RR=%-5s | "
                 "own-median p=%-9.3g | score shift %+.4f",
                 N, len(mapped), auc, CUT, rec.get("locked_logrank_p", np.nan),
                 rec.get("locked_risk_ratio"), rec.get("own_median_logrank_p", np.nan),
                 rec["score_shift"])

    df = pd.DataFrame(rows)
    save_table(df, cfg, "phase12_locked_threshold_all_panels.csv", log)

    ok = df.dropna(subset=["locked_logrank_p"])
    best = ok.sort_values("metabric_auc", ascending=False).iloc[0] if len(ok) else None
    if best is not None:
        log.info("=" * 70)
        log.info("BEST-TRANSFERRING panel %d (%d genes, AUC %.4f):",
                 best["panel_size"], best["n_genes"], best["metabric_auc"])
        log.info("  LOCKED TCGA cutpoint  -> log-rank p=%.3g, high-risk relapse %.1f%% vs %.1f%% (RR %.2f)",
                 best["locked_logrank_p"], 100 * best["locked_relapse_high"],
                 100 * best["locked_relapse_low"], best["locked_risk_ratio"])
        log.info("  METABRIC own median   -> log-rank p=%.3g", best["own_median_logrank_p"])
        agree = (best["locked_logrank_p"] < 0.05) == (best["own_median_logrank_p"] < 0.05)
        log.info("  cutpoint transferred? %s (score shift %+.4f)",
                 "YES - locked and data-driven splits agree" if agree
                 else "NO - locking changes the conclusion", best["score_shift"])
        save_json({"best_panel": int(best["panel_size"]),
                   "best_metabric_auc": float(best["metabric_auc"]),
                   "locked_logrank_p": float(best["locked_logrank_p"]),
                   "own_median_logrank_p": float(best["own_median_logrank_p"]),
                   "locked_risk_ratio": best["locked_risk_ratio"],
                   "cutpoint_transferred": bool(agree),
                   "score_shift": float(best["score_shift"])},
                  cfg, "phase12_summary.json", log)


if __name__ == "__main__":
    main()
