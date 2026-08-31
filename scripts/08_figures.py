"""PHASE 8 -- figures.

Analysis scripts write tables; this script turns tables into figures. No model
is fitted here and no number is recomputed -- everything is read from
results/tables/, so a figure can never disagree with the table it came from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from utils import get_logger, load_config, repo_root, set_seed

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
C = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "acc": "#C44E52", "grey": "#8C8C8C"}


def _save(fig, cfg, name, log):
    out = repo_root() / cfg["paths"]["figures"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out.relative_to(repo_root()))


def T(cfg, name) -> Path:
    return repo_root() / cfg["paths"]["tables"] / name


def fig_horizon(cfg, log):
    p = T(cfg, "gate1_horizon_sensitivity.csv")
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(d["horizon_years"], d["n_usable"], color=C["A"], width=0.6, label="usable n")
    ax[0].bar(d["horizon_years"], d["n_positive"], color=C["acc"], width=0.6, label="positives")
    ax[0].axhline(400, ls="--", c="k", lw=1, label="spec floor n=400")
    ax[0].set_xlabel("relapse horizon (years)"); ax[0].set_ylabel("patients")
    ax[0].set_title("Cohort size vs horizon\n(expression + methylation matched)")
    ax[0].legend(fontsize=7)
    ax[1].plot(d["horizon_years"], 100 * d["positive_rate"], "o-", color=C["B"])
    ax[1].axhspan(15, 25, color=C["C"], alpha=0.18, label="spec-expected 15-25%")
    ax[1].set_xlabel("relapse horizon (years)"); ax[1].set_ylabel("positive rate (%)")
    ax[1].set_title("Class balance vs horizon"); ax[1].legend(fontsize=7)
    _save(fig, cfg, "fig1_horizon_sensitivity.png", log)


def fig_model_comparison(cfg, log):
    data, labels, colors = [], [], []
    for tag, f, nm in (("A", "model_a_per_fold.csv", "A: clinical"),
                       ("B", "model_b_per_fold.csv", "B: XGBoost multi-omics"),
                       ("C", "model_c_per_fold.csv", "C: pathway NN")):
        p = T(cfg, f)
        if p.exists():
            d = pd.read_csv(p)
            data.append(d["auc"].dropna().values); labels.append(nm); colors.append(C[tag])
    if not data:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (v, c) in enumerate(zip(data, colors)):
        bp = ax[0].boxplot(v, positions=[i], widths=0.55, patch_artist=True,
                           medianprops=dict(color="k"))
        bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.7)
        ax[0].scatter(np.random.normal(i, 0.06, len(v)), v, s=7, c="k", alpha=0.35, zorder=3)
    ax[0].axhline(0.5, ls="--", c=C["grey"], lw=1)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax[0].set_ylabel("AUC (outer fold)")
    ax[0].set_title("Model comparison on identical outer folds")

    for tag, f, nm in (("A", "model_a_per_fold.csv", "A"), ("B", "model_b_per_fold.csv", "B"),
                       ("C", "model_c_per_fold.csv", "C")):
        p = T(cfg, f)
        if p.exists():
            d = pd.read_csv(p)
            ax[1].scatter(d["auc"], d["ap"], s=14, alpha=0.65, c=C[tag], label=f"Model {nm}")
    ax[1].set_xlabel("AUC"); ax[1].set_ylabel("Average precision")
    ax[1].set_title("AUC vs AP, per fold"); ax[1].legend(fontsize=8)
    _save(fig, cfg, "fig2_model_comparison.png", log)


def fig_roc(cfg, log):
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    any_ = False
    for tag, f, nm in (("B", "model_b_oof_predictions.csv", "B: XGBoost multi-omics"),
                       ("C", "model_c_oof_predictions.csv", "C: pathway NN")):
        p = T(cfg, f)
        if p.exists():
            d = pd.read_csv(p).dropna(subset=["risk_oof"])
            if d["label"].nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(d["label"], d["risk_oof"])
            from sklearn.metrics import auc as _auc
            ax.plot(fpr, tpr, color=C[tag], lw=1.8, label=f"{nm} (AUC={_auc(fpr,tpr):.3f})")
            any_ = True
    if not any_:
        plt.close(fig); return
    ax.plot([0, 1], [0, 1], ls="--", c=C["grey"], lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Pooled out-of-fold ROC"); ax.legend(fontsize=7, loc="lower right")
    _save(fig, cfg, "fig3_roc_oof.png", log)


def fig_shap(cfg, log):
    p = T(cfg, "model_b_shap_ranking.csv")
    if not p.exists():
        return
    d = pd.read_csv(p).head(30).iloc[::-1]
    cmap = {"clinical": C["acc"], "expression": C["A"], "methylation": C["C"]}
    fig, ax = plt.subplots(figsize=(6.4, 7))
    ax.barh(range(len(d)), d["mean_abs_shap"],
            color=[cmap.get(b, C["grey"]) for b in d["block"]])
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f.split(":", 1)[-1] for f in d["feature"]], fontsize=7)
    ax.set_xlabel("mean |SHAP| (out-of-fold)")
    ax.set_title("Model B: top 30 features")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=v, label=k) for k, v in cmap.items()], fontsize=7)
    _save(fig, cfg, "fig4_shap_top30.png", log)


def fig_panel_curve(cfg, log):
    p = T(cfg, "signature_panel_curve.csv")
    if not p.exists():
        return
    d = pd.read_csv(p).sort_values("panel_size")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.errorbar(d["panel_size"], d["auc_mean"],
                yerr=[d["auc_mean"] - d["auc_ci_lo"], d["auc_ci_hi"] - d["auc_mean"]],
                fmt="o-", color=C["B"], capsize=3, label="panel AUC")
    for f, tag, nm in (("model_a_summary.json", "A", "Model A (clinical)"),
                       ("model_b_summary.json", "B", "Model B (all features)")):
        q = T(cfg, f)
        if q.exists():
            ax.axhline(json.load(open(q))["auc_mean"], ls="--", lw=1.2,
                       color=C[tag], label=nm)
    pj = T(cfg, "signature_plateau.json")
    if pj.exists():
        ax.axvline(json.load(open(pj))["plateau_panel_size"], ls=":", c="k", lw=1.2,
                   label="plateau")
    ax.set_xscale("log"); ax.set_xticks(d["panel_size"])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("panel size (markers)"); ax.set_ylabel("AUC")
    ax.set_title("Performance vs panel size"); ax.legend(fontsize=7)
    _save(fig, cfg, "fig5_panel_size_curve.png", log)


def fig_immune(cfg, log):
    p = T(cfg, "gate6_risk_vs_immune.csv")
    if not p.exists():
        return
    d = pd.read_csv(p).sort_values("delta")
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = [C["acc"] if s else C["grey"] for s in d["significant_fdr_0.05"]]
    ax.barh(range(len(d)), d["delta"], color=cols)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["population"], fontsize=8)
    ax.axvline(0, c="k", lw=1)
    ax.set_xlabel("MCP-counter score: high-risk minus low-risk")
    ax.set_title("Immune/stromal composition by predicted risk\n(red = FDR < 0.05)")
    _save(fig, cfg, "fig6_immune_deconvolution.png", log)


def fig_km(cfg, log):
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    panels = [("gate7_tcga_km_data.csv", "TCGA (out-of-fold risk)"),
              ("gate7_metabric_km_data.csv", "METABRIC (external)")]
    avail = [(f, t) for f, t in panels if T(cfg, f).exists()]
    if not avail:
        return
    fig, axes = plt.subplots(1, len(avail), figsize=(5 * len(avail), 3.9), squeeze=False)
    for ax, (f, title) in zip(axes[0], avail):
        d = pd.read_csv(T(cfg, f))
        kmf = KaplanMeierFitter()
        for g, c, nm in ((0, C["A"], "low risk"), (1, C["acc"], "high risk")):
            m = d["risk_group"] == g
            if m.sum() < 5:
                continue
            kmf.fit(d.loc[m, "time_days"] / 365.25, d.loc[m, "event"],
                    label=f"{nm} (n={int(m.sum())})")
            kmf.plot_survival_function(ax=ax, color=c, ci_alpha=0.12)
        lr = logrank_test(d.loc[d.risk_group == 0, "time_days"],
                          d.loc[d.risk_group == 1, "time_days"],
                          d.loc[d.risk_group == 0, "event"],
                          d.loc[d.risk_group == 1, "event"])
        ax.set_title(f"{title}\nlog-rank p = {lr.p_value:.3g}", fontsize=9)
        ax.set_xlabel("years"); ax.set_ylabel("relapse-free probability")
        ax.set_ylim(0, 1.02); ax.legend(fontsize=7)
    _save(fig, cfg, "fig7_kaplan_meier.png", log)


def fig_metabric(cfg, log):
    p = T(cfg, "gate7_metabric_performance.csv")
    if not p.exists():
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for mname, c in (("xgboost", C["B"]), ("logreg_l2", C["A"])):
        s = d[d["model"] == mname].sort_values("panel_size")
        if len(s):
            ax.plot(s["panel_size"], s["metabric_auc"], "o-", color=c, label=f"METABRIC, {mname}")
    q = T(cfg, "model_b_summary.json")
    if q.exists():
        ax.axhline(json.load(open(q))["auc_mean"], ls="--", c=C["C"], lw=1.2,
                   label="TCGA internal CV (Model B)")
    ax.axhline(0.5, ls=":", c=C["grey"], lw=1, label="chance")
    ax.set_xscale("log"); ax.set_xticks(sorted(d["panel_size"].unique()))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("panel size (markers)"); ax.set_ylabel("AUC")
    ax.set_title("External validation: cross-platform transfer"); ax.legend(fontsize=7)
    _save(fig, cfg, "fig8_metabric_validation.png", log)



def fig_representation(cfg, log):
    """Phase 10: learned latent factors vs raw features, and the transduction check."""
    p = T(cfg, "phase10_ae_performance.csv")
    if not p.exists():
        return
    d = pd.read_csv(p)
    abl = T(cfg, "phase9_ablation.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # left: dimensionality vs AUC
    pts = []
    if abl.exists():
        a = pd.read_csv(abl)
        for _, x in a[a["block"] == "expression+methylation"].iterrows():
            pts.append((x["n_features"], x["auc_mean"], f"raw ({x['model']})", C["grey"]))
    row = d[d["model"] == "AE latent (transductive)"]
    if len(row):
        pts.append((30, float(row["auc_mean"].iloc[0]), "AE latent (30)", C["C"]))
    for nf, auc, lbl, c in pts:
        ax[0].scatter(nf, auc, s=90, color=c, zorder=3, edgecolor="k", linewidth=0.5)
        ax[0].annotate(lbl, (nf, auc), textcoords="offset points", xytext=(6, 6), fontsize=7.5)
    cl = d[d["model"] == "clinical only"]
    if len(cl):
        ax[0].axhline(float(cl["auc_mean"].iloc[0]), ls="--", c=C["A"], lw=1.2,
                      label="clinical only")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("number of features (log scale)")
    ax[0].set_ylabel("AUC")
    ax[0].set_title("Learned representation vs raw features")
    ax[0].legend(fontsize=7)

    # right: model comparison bars
    order = ["clinical only", "AE latent (transductive)", "AE latent + clinical (transductive)",
             "AE latent (fold-safe)", "AE latent + clinical (fold-safe)"]
    sub = d[d["model"].isin(order)].set_index("model").reindex(order).dropna(subset=["auc_mean"])
    cols = [C["A"] if "clinical only" in m else (C["C"] if "fold-safe" in m else C["B"])
            for m in sub.index]
    yerr = [sub["auc_mean"] - sub["auc_ci_lo"], sub["auc_ci_hi"] - sub["auc_mean"]]
    ax[1].barh(range(len(sub)), sub["auc_mean"], xerr=yerr, color=cols,
               error_kw=dict(lw=1, capsize=3))
    ax[1].set_yticks(range(len(sub)))
    ax[1].set_yticklabels([m.replace(" (transductive)", "\n(transductive)")
                           .replace(" (fold-safe)", "\n(fold-safe)") for m in sub.index],
                          fontsize=7)
    ax[1].axvline(0.5, ls=":", c=C["grey"], lw=1)
    ax[1].set_xlim(0.45, 0.9)
    ax[1].set_xlabel("AUC")
    ax[1].set_title("Autoencoder factors vs clinical baseline")
    _save(fig, cfg, "fig9_representation_learning.png", log)


def fig_latent_immune(cfg, log):
    """What the unsupervised factors actually encode."""
    p = T(cfg, "phase10_latent_interpretation.csv")
    if not p.exists():
        return
    d = pd.read_csv(p)
    if "corr_immune" not in d.columns:
        return
    fig, ax = plt.subplots(figsize=(5.6, 4))
    ax.scatter(d["corr_immune"].abs(), d["corr_stage"].abs(), s=55,
               c=d["auc_vs_label"], cmap="viridis", edgecolor="k", linewidth=0.4)
    lim = max(0.55, float(d["corr_immune"].abs().max()) + 0.05)
    ax.plot([0, lim], [0, lim], ls="--", c=C["grey"], lw=1)
    ax.set_xlabel("|correlation| with strongest immune population")
    ax.set_ylabel("|correlation| with pathologic stage")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Latent factors encode immune biology,\nnot stage")
    cb = fig.colorbar(ax.collections[0], ax=ax)
    cb.set_label("univariate AUC vs relapse", fontsize=8)
    _save(fig, cfg, "fig10_latent_factors.png", log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("08_figures", cfg)
    for fn in (fig_horizon, fig_model_comparison, fig_roc, fig_shap,
               fig_panel_curve, fig_immune, fig_km, fig_metabric,
               fig_representation, fig_latent_immune):
        try:
            fn(cfg, log)
        except Exception as exc:  # noqa: BLE001
            log.warning("%s failed: %s", fn.__name__, exc)
    log.info("figures complete")


if __name__ == "__main__":
    main()
