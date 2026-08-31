"""PHASE 10 -- unsupervised multi-omics representation learning.

Motivation. The supervised pipeline was crippled by n=367: the landmark rule
discards every patient censored before the horizon. But an autoencoder needs NO
LABELS, so it can train on every patient with both assays -- 782 rather than 367.
The label bottleneck simply does not bind on the unsupervised half.

Architecture (denoising multi-modal autoencoder):

    expression [N, G_e] --enc_e--> h_e [256] --\
                                                concat --> z [n_latent]
    methylation[N, G_m] --enc_m--> h_m [256] --/
                                                |
                        expr_hat <--dec_e-------+-------dec_m--> meth_hat

    loss = MSE(expr_hat, expr) + MSE(meth_hat, meth)

LEAKAGE, and why this script trains the encoder twice.
Training one encoder on all 782 patients and then cross-validating a classifier
on its embeddings is TRANSDUCTIVE: the encoder has seen the FEATURES (never the
labels) of patients who later appear in test folds. That is accepted practice in
the semi-supervised literature but it is not the same as a clean holdout, and
reporting only that number would overstate the result. So both are run:

    A. transductive -- encoder fit once on all 782 patients
    B. fold-safe    -- encoder refit inside each fold on TRAINING patients only,
                       plus unlabelled patients not in that test fold

The gap between A and B is itself the finding: it quantifies how much
transduction inflates performance.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from data import clinical_matrix, load_processed
from evaluate import fold_metrics, paired_fold_test, summarise
from folds import build_or_load_folds
from utils import get_logger, load_config, repo_root, save_json, save_table, set_seed

warnings.filterwarnings("ignore")

N_TOP = 4000          # per-modality variance filter, keeps the model CPU-tractable
N_LATENT = 30
HIDDEN = 256
EPOCHS = 300
PATIENCE = 25
NOISE = 0.1
BATCH = 32


class MultiOmicsAE(nn.Module):
    def __init__(self, g_e: int, g_m: int, hidden: int, n_latent: int, dropout: float = 0.2):
        super().__init__()
        self.enc_e = nn.Sequential(nn.Linear(g_e, hidden), nn.BatchNorm1d(hidden),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.enc_m = nn.Sequential(nn.Linear(g_m, hidden), nn.BatchNorm1d(hidden),
                                   nn.ReLU(), nn.Dropout(dropout))
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, n_latent))
        self.dec_e = nn.Sequential(nn.Linear(n_latent, hidden), nn.ReLU(), nn.Linear(hidden, g_e))
        self.dec_m = nn.Sequential(nn.Linear(n_latent, hidden), nn.ReLU(), nn.Linear(hidden, g_m))

    def encode(self, xe, xm):
        return self.fuse(torch.cat([self.enc_e(xe), self.enc_m(xm)], dim=1))

    def forward(self, xe, xm):
        z = self.encode(xe, xm)
        return self.dec_e(z), self.dec_m(z), z


def train_ae(Xe, Xm, seed, log=None, tag=""):
    """Denoising AE. Early stopping on a held-out 15% RECONSTRUCTION split
    (still unsupervised -- no labels involved anywhere)."""
    torch.manual_seed(seed)
    n = len(Xe)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(8, int(0.15 * n))
    va, tr = perm[:n_val], perm[n_val:]

    model = MultiOmicsAE(Xe.shape[1], Xm.shape[1], HIDDEN, N_LATENT)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = nn.MSELoss()

    Xe_t, Xm_t = torch.tensor(Xe), torch.tensor(Xm)
    best, best_state, wait = np.inf, None, 0
    for ep in range(EPOCHS):
        model.train()
        idx = torch.tensor(rng.permutation(tr))
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            if len(b) < 2:            # BatchNorm needs >1 sample
                continue
            xe, xm = Xe_t[b], Xm_t[b]
            # denoising: corrupt the input, reconstruct the clean signal
            xe_n = xe + NOISE * torch.randn_like(xe)
            xm_n = xm + NOISE * torch.randn_like(xm)
            opt.zero_grad()
            re, rm, _ = model(xe_n, xm_n)
            loss = crit(re, xe) + crit(rm, xm)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            re, rm, _ = model(Xe_t[va], Xm_t[va])
            vl = (crit(re, Xe_t[va]) + crit(rm, Xm_t[va])).item()
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
        if log and ep % 50 == 0:
            log.info("    %s epoch %3d  val recon loss %.5f", tag, ep, vl)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    if log:
        log.info("    %s stopped at epoch %d, best val recon loss %.5f", tag, ep + 1, best)
    return model, ep + 1, best


def embed(model, Xe, Xm):
    model.eval()
    with torch.no_grad():
        return model.encode(torch.tensor(Xe), torch.tensor(Xm)).numpy()


def top_var(X, k):
    v = X.var(axis=0)
    return np.argsort(v)[::-1][:min(k, X.shape[1])]


def run_clf(Z, y, folds, cfg, name, log, kind="logreg"):
    rows = []
    for i, (tr, te) in enumerate(folds):
        if kind == "logreg":
            m = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler()),
                          ("clf", LogisticRegressionCV(Cs=[0.01, 0.1, 1.0], cv=3,
                                                       scoring="roc_auc", max_iter=4000,
                                                       class_weight="balanced",
                                                       random_state=cfg["seed"], n_jobs=4))])
        else:
            pos = int(y[tr].sum()); neg = len(tr) - pos
            m = Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("clf", XGBClassifier(objective="binary:logistic",
                                                eval_metric="logloss", tree_method="hist",
                                                scale_pos_weight=neg / max(pos, 1),
                                                max_depth=3, learning_rate=0.05,
                                                n_estimators=400, subsample=0.8,
                                                colsample_bytree=0.6, n_jobs=4,
                                                random_state=cfg["seed"], verbosity=0))])
        Zi = Z[i] if isinstance(Z, dict) else Z      # dict => per-fold embeddings
        m.fit(Zi[tr], y[tr])
        r = fold_metrics(y[te], m.predict_proba(Zi[te])[:, 1])
        r["fold_id"] = i
        rows.append(r)
    df = pd.DataFrame(rows)
    s = summarise(df, name)
    log.info("  %-46s AUC=%.4f [%.4f,%.4f]  AP=%.4f",
             name, s["auc_mean"], s["auc_ci_lo"], s["auc_ci_hi"], s["ap_mean"])
    return df, s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="labelled cohort config")
    ap.add_argument("--full-config", required=True, help="all-patients cohort config")
    ap.add_argument("--strict-folds", type=int, default=10,
                    help="folds for the fold-safe encoder (expensive: refits per fold)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_full = load_config(args.full_config)
    set_seed(cfg["seed"])
    torch.set_num_threads(4)
    log = get_logger("11_multiomics_autoencoder", cfg)

    expr, met, clin, lab = load_processed(cfg)          # labelled: n=367
    y = lab["label"].values.astype(int)
    fexpr, fmet, fclin, flab = load_processed(cfg_full)  # all: n=782
    log.info("labelled cohort n=%d (%d positive) | full cohort n=%d (%d extra unlabelled)",
             len(y), int(y.sum()), len(fexpr), len(fexpr) - len(expr))

    genes_e, genes_m = list(expr.columns), list(met.columns)
    fexpr, fmet = fexpr[genes_e], fmet[genes_m]

    # variance filter chosen on the FULL cohort (unsupervised, no labels involved)
    ie = top_var(fexpr.values, N_TOP)
    im = top_var(fmet.values, N_TOP)
    log.info("variance filter: expression %d->%d, methylation %d->%d",
             len(genes_e), len(ie), len(genes_m), len(im))
    ge = [genes_e[i] for i in ie]
    gm = [genes_m[i] for i in im]

    sc_e = StandardScaler().fit(fexpr[ge].values)
    sc_m = StandardScaler().fit(fmet[gm].values)
    FXe = sc_e.transform(fexpr[ge].values).astype(np.float32)
    FXm = sc_m.transform(fmet[gm].values).astype(np.float32)
    LXe = sc_e.transform(expr[ge].values).astype(np.float32)
    LXm = sc_m.transform(met[gm].values).astype(np.float32)

    folds = build_or_load_folds(y, cfg, log)
    Xc, cnames = clinical_matrix(clin, log)

    # ------------------------------------------------ A. transductive encoder
    log.info("=" * 70)
    log.info("A. TRANSDUCTIVE encoder -- fit once on all %d patients (no labels)", len(FXe))
    log.info("=" * 70)
    model, n_ep, vl = train_ae(FXe, FXm, cfg["seed"], log, "[transductive]")
    Z_all = embed(model, LXe, LXm)
    log.info("latent embedding: %s", Z_all.shape)

    results, store = [], {}
    for nm, Z, kind in (("clinical only", Xc, "logreg"),
                        ("AE latent (transductive)", Z_all, "logreg"),
                        ("AE latent + clinical (transductive)",
                         np.hstack([Xc, Z_all]), "logreg"),
                        ("AE latent + clinical (transductive, xgb)",
                         np.hstack([Xc, Z_all]), "xgb")):
        df, s = run_clf(Z, y, folds, cfg, nm, log, kind)
        s["encoder"] = "transductive" if "transductive" in nm else "n/a"
        results.append(s); store[nm] = df

    # ------------------------------------------------ B. fold-safe encoder
    sub = folds[: args.strict_folds]
    log.info("=" * 70)
    log.info("B. FOLD-SAFE encoder -- refit inside each of %d folds, training patients only",
             len(sub))
    log.info("=" * 70)
    lab_index = list(lab.index)
    full_index = list(fexpr.index)
    pos_in_full = {p: i for i, p in enumerate(full_index)}

    Z_fold = {}
    for i, (tr, te) in enumerate(sub):
        test_patients = {lab_index[j] for j in te}
        # unsupervised training set: everyone in the full cohort EXCEPT this fold's test patients
        keep = [pos_in_full[p] for p in full_index if p not in test_patients]
        m_i, ep_i, vl_i = train_ae(FXe[keep], FXm[keep], cfg["seed"] + i)
        Z_fold[i] = embed(m_i, LXe, LXm)
        log.info("  fold %2d/%d: encoder trained on %d patients (%d test held out), "
                 "%d epochs, val loss %.5f", i + 1, len(sub), len(keep), len(test_patients),
                 ep_i, vl_i)

    for nm, build in (("AE latent (fold-safe)", lambda i: Z_fold[i]),
                      ("AE latent + clinical (fold-safe)",
                       lambda i: np.hstack([Xc, Z_fold[i]]))):
        Zd = {i: build(i) for i in range(len(sub))}
        df, s = run_clf(Zd, y, sub, cfg, nm, log, "logreg")
        s["encoder"] = "fold-safe"; s["n_folds_used"] = len(sub)
        results.append(s); store[nm] = df

    # same-fold comparison so transductive vs fold-safe is apples to apples
    log.info("-" * 70)
    trans_sub, s_ts = run_clf(Z_all, y, sub, cfg,
                              f"AE latent (transductive, first {len(sub)} folds)", log)
    gap = s_ts["auc_mean"] - [r for r in results if r["model"] == "AE latent (fold-safe)"][0]["auc_mean"]
    log.info("TRANSDUCTION GAP on identical folds: %+.4f AUC "
             "(transductive minus fold-safe)", gap)

    res = pd.DataFrame(results)
    save_table(res, cfg, "phase10_ae_performance.csv", log)

    # paired contrasts
    crows = []
    for a, b in (("clinical only", "AE latent + clinical (transductive)"),
                 ("clinical only", "AE latent (transductive)")):
        if a in store and b in store:
            t = paired_fold_test(store[a], store[b], "auc")
            t.update({"baseline": a, "augmented": b}); crows.append(t)
            log.info("  %-38s -> %-42s dAUC=%+.4f p=%.3g",
                     a, b, t["mean_diff"], t["p_value"])
    if crows:
        save_table(pd.DataFrame(crows), cfg, "phase10_contrasts.csv", log)

    # ------------------------------------------------ interpretation
    log.info("=" * 70)
    log.info("LATENT FACTOR INTERPRETATION")
    log.info("=" * 70)
    interp = []
    mcp_p = repo_root() / cfg["paths"]["tables"] / "gate6_mcp_scores.csv"
    mcp = pd.read_csv(mcp_p).set_index("patient").reindex(lab.index) if mcp_p.exists() else None
    stage = pd.to_numeric(clin["stage"], errors="coerce").values
    for f in range(Z_all.shape[1]):
        z = Z_all[:, f]
        row = {"factor": f}
        ok = ~np.isnan(stage)
        row["corr_stage"] = round(float(np.corrcoef(z[ok], stage[ok])[0, 1]), 3)
        row["auc_vs_label"] = round(float(roc_auc_score(y, z)), 3)
        if mcp is not None:
            best_pop, best_r = None, 0.0
            for pop in mcp.columns:
                r = float(np.corrcoef(z, mcp[pop].values)[0, 1])
                if abs(r) > abs(best_r):
                    best_pop, best_r = pop, r
            row["top_immune_population"] = best_pop
            row["corr_immune"] = round(best_r, 3)
        interp.append(row)
    idf = pd.DataFrame(interp)
    idf["auc_vs_label"] = idf["auc_vs_label"].apply(lambda a: max(a, 1 - a))
    idf = idf.sort_values("auc_vs_label", ascending=False)
    save_table(idf, cfg, "phase10_latent_interpretation.csv", log)
    log.info("top latent factors by univariate association with relapse:\n%s",
             idf.head(10).to_string(index=False))

    save_json({"n_labelled": int(len(y)), "n_full_cohort": int(len(FXe)),
               "extra_unlabelled_patients": int(len(FXe) - len(y)),
               "n_latent": N_LATENT, "n_top_features_per_modality": N_TOP,
               "transductive_epochs": int(n_ep), "transductive_val_loss": float(vl),
               "transduction_gap_auc": round(float(gap), 4),
               "strict_folds": len(sub)},
              cfg, "phase10_summary.json", log)
    log.info("PHASE 10 COMPLETE")


if __name__ == "__main__":
    main()
