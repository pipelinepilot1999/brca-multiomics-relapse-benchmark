"""PHASE 6 / GATE 6 -- immune/stromal deconvolution vs predicted risk.

Question: is the signature largely an immune-infiltration readout? If so, that
is a real finding and is reported as such.

IMPLEMENTATION NOTE (deviation from spec, stated plainly):
The spec asks for immunedeconv (R) with quanTIseq or MCP-counter. The R and
Bioconductor dependency install for immunedeconv did not complete in the time
available on this machine, so MCP-counter is reimplemented directly in Python
here, from the published marker set (Becht et al., Genome Biology 2016 -- the
same 111-transcript, 10-population signature immunedeconv wraps).

MCP-counter is a marker-gene aggregate, not a fitted deconvolution: the score
for a population is the mean log2 expression of its markers. The reimplementation
is therefore faithful rather than approximate. It yields ABUNDANCE SCORES that
are comparable BETWEEN samples for a given population, and NOT cell fractions
comparable between populations -- the same interpretive limit the R package has.
quanTIseq was not run: it needs the TIL10 signature matrix that ships with the R
package, which is unavailable here.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from data import load_processed
from utils import get_logger, load_config, repo_root, save_json, save_table, set_seed


def load_markers(path) -> pd.DataFrame:
    m = pd.read_csv(path, sep="\t")
    m.columns = [c.strip().strip('"') for c in m.columns]
    m = m.rename(columns={"HUGO symbols": "gene", "Cell population": "population"})
    for c in ("gene", "population"):
        m[c] = m[c].astype(str).str.strip().str.strip('"')
    return m[["gene", "population"]].dropna()


def mcp_counter(expr: pd.DataFrame, markers: pd.DataFrame, log) -> pd.DataFrame:
    """Mean log2 expression of each population's markers. Input already log2."""
    scores, cover = {}, []
    for pop, grp in markers.groupby("population"):
        want = sorted(set(grp["gene"]))
        have = [g for g in want if g in expr.columns]
        cover.append({"population": pop, "markers_in_signature": len(want),
                      "markers_found": len(have),
                      "coverage": round(len(have) / len(want), 3),
                      "missing": ";".join(sorted(set(want) - set(have)))})
        if not have:
            log.warning("population %s has NO markers present -- skipped", pop)
            continue
        scores[pop] = expr[have].mean(axis=1)
    cov = pd.DataFrame(cover).sort_values("coverage")
    log.info("MCP-counter marker coverage:\n%s",
             cov[["population", "markers_in_signature", "markers_found", "coverage"]]
             .to_string(index=False))
    return pd.DataFrame(scores), cov


def pick_risk_source(cfg, log):
    tdir = repo_root() / cfg["paths"]["tables"]
    best, best_auc, best_f = None, -np.inf, None
    for tag, s_f, p_f in (("B", "model_b_summary.json", "model_b_oof_predictions.csv"),
                          ("C", "model_c_summary.json", "model_c_oof_predictions.csv")):
        sp, pp = tdir / s_f, tdir / p_f
        if sp.exists() and pp.exists():
            auc = json.load(open(sp))["auc_mean"]
            if auc > best_auc:
                best, best_auc, best_f = tag, auc, pp
    if best is None:
        raise SystemExit("no model OOF predictions found -- run Phase 3 and/or 4 first")
    log.info("risk groups taken from Model %s (outer-fold AUC %.4f)", best, best_auc)
    return best, best_auc, pd.read_csv(best_f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("06_immune_deconv", cfg)

    expr, met, clin, lab = load_processed(cfg)
    markers = load_markers(repo_root() / cfg["files"]["mcp_genes"])
    log.info("MCP-counter signature: %d markers, %d populations",
             len(markers), markers["population"].nunique())

    scores, cov = mcp_counter(expr, markers, log)
    save_table(cov, cfg, "gate6_mcp_marker_coverage.csv", log)
    out = scores.copy()
    out.insert(0, "patient", expr.index)
    save_table(out, cfg, "gate6_mcp_scores.csv", log)

    tag, auc, risk = pick_risk_source(cfg, log)
    risk = risk.set_index("patient").reindex(expr.index)
    r = risk["risk_oof"].values
    y = lab["label"].values.astype(int)

    med = np.nanmedian(r)
    high = r > med
    log.info("median split at risk=%.4f: high-risk n=%d, low-risk n=%d",
             med, int(high.sum()), int((~high).sum()))
    log.info("observed relapse rate: high-risk %.1f%%, low-risk %.1f%%",
             100 * y[high].mean(), 100 * y[~high].mean())

    rows = []
    for pop in scores.columns:
        v = scores[pop].values
        a, b = v[high], v[~high]
        try:
            u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            u, p = np.nan, np.nan
        # rank-biserial correlation as effect size
        eff = (2 * u / (len(a) * len(b)) - 1) if np.isfinite(u) else np.nan
        rows.append({"population": pop,
                     "mean_high_risk": round(float(np.mean(a)), 4),
                     "mean_low_risk": round(float(np.mean(b)), 4),
                     "delta": round(float(np.mean(a) - np.mean(b)), 4),
                     "rank_biserial": round(float(eff), 4),
                     "u_stat": float(u), "p_value": float(p)})
    res = pd.DataFrame(rows)
    ok = res["p_value"].notna()
    res.loc[ok, "p_fdr"] = multipletests(res.loc[ok, "p_value"], method="fdr_bh")[1]
    res["significant_fdr_0.05"] = res["p_fdr"] < 0.05
    res = res.sort_values("p_fdr")
    save_table(res, cfg, "gate6_risk_vs_immune.csv", log)

    log.info("GATE 6 -- MCP-counter scores, high vs low predicted risk "
             "(Mann-Whitney U, BH-FDR):")
    for _, x in res.iterrows():
        log.info("   %-24s delta=%+.3f  rbc=%+.3f  p=%.3g  FDR=%.3g %s",
                 x["population"], x["delta"], x["rank_biserial"],
                 x["p_value"], x["p_fdr"], "*" if x["significant_fdr_0.05"] else "")

    n_sig = int(res["significant_fdr_0.05"].sum())
    immune_pops = {"T cells", "CD8 T cells", "Cytotoxic lymphocytes", "B lineage",
                   "NK cells", "Monocytic lineage", "Myeloid dendritic cells", "Neutrophils"}
    sig_immune = res[res["significant_fdr_0.05"] & res["population"].isin(immune_pops)]
    verdict = ("The signature separates risk groups that ALSO differ in immune "
               "infiltration -- it is at least partly an immune readout."
               if len(sig_immune) else
               "No immune population differs significantly between risk groups at "
               "FDR 0.05: the signature is NOT primarily an immune-infiltration readout.")
    log.info("GATE 6 VERDICT: %d/%d populations significant at FDR 0.05. %s",
             n_sig, len(res), verdict)

    save_json({"risk_model": tag, "risk_model_auc": auc,
               "median_split": float(med),
               "n_high_risk": int(high.sum()), "n_low_risk": int((~high).sum()),
               "relapse_rate_high": round(float(y[high].mean()), 4),
               "relapse_rate_low": round(float(y[~high].mean()), 4),
               "n_significant_fdr05": n_sig,
               "n_significant_immune": int(len(sig_immune)),
               "verdict": verdict,
               "method": "MCP-counter (Becht 2016) reimplemented in Python; "
                         "see module docstring for why immunedeconv/R was not used"},
              cfg, "gate6_summary.json", log)


if __name__ == "__main__":
    main()
