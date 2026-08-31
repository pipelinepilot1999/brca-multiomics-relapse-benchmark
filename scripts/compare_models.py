"""Paired comparison of Models A, B and C on identical outer folds.

The pairing is only valid because all three models were scored on the same fold
indices (loaded from cv_folds.json), so a paired test is the right test.
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from evaluate import paired_fold_test
from utils import get_logger, load_config, repo_root, save_table, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("compare_models", cfg)
    t = repo_root() / cfg["paths"]["tables"]

    per = {}
    for tag, f in (("A", "model_a_per_fold.csv"), ("B", "model_b_per_fold.csv"),
                   ("C", "model_c_per_fold.csv")):
        p = t / f
        if p.exists():
            per[tag] = pd.read_csv(p)

    summ = []
    for tag, f in (("A", "model_a_summary.json"), ("B", "model_b_summary.json"),
                   ("C", "model_c_summary.json")):
        p = t / f
        if p.exists():
            summ.append(json.load(open(p)))
    if summ:
        save_table(pd.DataFrame(summ), cfg, "model_summary_all.csv", log)

    rows = []
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        if a in per and b in per:
            r = paired_fold_test(per[a], per[b], "auc")
            r["comparison"] = f"{a} vs {b}"
            r["metric"] = "auc"
            rows.append(r)
            log.info("%s vs %s : mean AUC diff %+0.4f (%s better on %d/%d folds), "
                     "paired t p=%.3g, wilcoxon p=%.3g",
                     a, b, r["mean_diff"], b, r["b_wins_folds"], r["n_folds"],
                     r["p_value"], r["wilcoxon_p"])
    if rows:
        save_table(pd.DataFrame(rows), cfg, "model_comparison_tests.csv", log)

    if "A" in per and "B" in per:
        aA = pd.read_json(t / "model_a_summary.json", typ="series")["auc_mean"]
        aB = pd.read_json(t / "model_b_summary.json", typ="series")["auc_mean"]
        if aA > aB:
            log.warning("HEADLINE: the clinical baseline (A, AUC %.4f) OUTPERFORMS the "
                        "multi-omics model (B, AUC %.4f). Reported as found.", aA, aB)
    log.info("comparison complete")


if __name__ == "__main__":
    main()
