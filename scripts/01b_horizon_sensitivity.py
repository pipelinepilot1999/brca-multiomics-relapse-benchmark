"""GATE 1 follow-up -- horizon sensitivity.

Gate 1 failed its own viability check at the 5-year horizon (n below 400).
The spec anticipates this: "We may need to move to a 3-year window."

Rather than pick a horizon by intuition, this enumerates candidate horizons and
reports what each costs and yields, over the SAME patient universe (patients
with expression AND methylation AND a PFI record). The chosen horizon is then
a documented trade-off, not an arbitrary knob.
"""
from __future__ import annotations

import argparse
import gzip

import numpy as np
import pandas as pd

from utils import (get_logger, load_config, repo_root, save_table, set_seed,
                   tcga_is_primary_tumour, tcga_patient_id)

HORIZONS = [730, 1095, 1460, 1825]      # 2, 3, 4, 5 years


def header_patients(path, sep="\t") -> set[str]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        cols = fh.readline().rstrip("\n").split(sep)[1:]
    return {tcga_patient_id(c) for c in cols if tcga_is_primary_tumour(c)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("01b_horizon_sensitivity", cfg)
    root = repo_root()

    expr_p = header_patients(root / cfg["files"]["tcga_expression"])
    met_p = header_patients(root / cfg["files"]["tcga_methylation"])
    log.info("patient universe: expression=%d methylation=%d both=%d",
             len(expr_p), len(met_p), len(expr_p & met_p))

    cdr = pd.read_csv(root / cfg["files"]["tcga_cdr"], sep="\t", low_memory=False)
    brca = cdr[cdr["cancer type abbreviation"] == "BRCA"].copy()
    if "Redaction" in brca.columns:
        brca = brca[brca["Redaction"].isna() | (brca["Redaction"].astype(str).str.strip() == "")]
    brca["patient"] = brca["_PATIENT"].astype(str)
    brca = brca.drop_duplicates("patient")
    brca["PFI"] = pd.to_numeric(brca["PFI"], errors="coerce")
    brca["PFI.time"] = pd.to_numeric(brca["PFI.time"], errors="coerce")
    brca = brca.dropna(subset=["PFI", "PFI.time"])

    both = expr_p & met_p
    sub = brca[brca["patient"].isin(both)].copy()
    log.info("patients with PFI AND expression AND methylation: %d", len(sub))

    ev, t = sub["PFI"].astype(int).values, sub["PFI.time"].astype(float).values

    rows = []
    for h in HORIZONS:
        pos = int(((ev == 1) & (t <= h)).sum())
        neg = int((((ev == 0) & (t >= h)) | ((ev == 1) & (t > h))).sum())
        exc = int(((ev == 0) & (t < h)).sum())
        n = pos + neg
        rows.append({
            "horizon_days": h,
            "horizon_years": round(h / 365.25, 2),
            "n_usable": n,
            "n_positive": pos,
            "n_negative": neg,
            "positive_rate": round(pos / n, 4) if n else np.nan,
            "n_excluded_early_censor": exc,
            "pct_excluded": round(100 * exc / len(sub), 1),
            "meets_n_ge_400": n >= 400,
            "meets_pos_ge_60": pos >= 60,
            "prevalence_in_15_25_band": bool(n and 0.15 <= pos / n <= 0.25),
        })
        log.info("horizon %4dd (%.1fy): n=%3d pos=%3d (%.1f%%) neg=%3d excluded=%3d (%.0f%%)",
                 h, h / 365.25, n, pos, 100 * pos / n if n else 0, neg, exc,
                 100 * exc / len(sub))

    df = pd.DataFrame(rows)
    save_table(df, cfg, "gate1_horizon_sensitivity.csv", log)

    ok = df[df["meets_n_ge_400"] & df["meets_pos_ge_60"]]
    if len(ok):
        best = ok.sort_values("horizon_days", ascending=False).iloc[0]
        log.info("RECOMMENDATION: longest horizon meeting the spec floors is %d days (%.1f y): "
                 "n=%d, pos=%d (%.1f%%)", best["horizon_days"], best["horizon_years"],
                 best["n_usable"], best["n_positive"], 100 * best["positive_rate"])
    else:
        log.warning("NO horizon meets both spec floors (n>=400, pos>=60) with matched "
                    "expression+methylation. The multi-omics intersection is the binding "
                    "constraint, not the horizon.")


if __name__ == "__main__":
    main()
