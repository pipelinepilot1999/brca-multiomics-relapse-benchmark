"""PHASE 4 / GATE 4 -- Model C: pathway-informed neural network (P-NET style).

Architecture (Elmarakeby et al., Nature 2021):

    per-gene inputs (expression, methylation)   [N, G, 2]
            |  gene layer: per-gene 2 -> 1
            v                                    [N, G]
    pathway layer: nn.Linear with a FIXED BINARY MASK from MSigDB
            v                                    [N, P]
    dense (64) -> dropout -> single logit

Evaluated on the SAME outer fold indices as Models A and B (loaded from
cv_folds.json, not merely the same seed), so the comparison is paired.

Note on the expected outcome: XGBoost beating this network at n~370 is a
likely and acceptable result. It is reported as found, not tuned away.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import load_processed
from evaluate import fold_metrics, summarise
from folds import build_or_load_folds
from pathways import build_mask, load_gene_sets
from utils import (flag, get_logger, load_config, repo_root, save_json,
                   save_table, set_seed)

warnings.filterwarnings("ignore")


class MaskedLinear(nn.Module):
    """Linear layer whose connectivity is fixed by a binary mask."""

    def __init__(self, mask: torch.Tensor):
        super().__init__()
        self.register_buffer("mask", mask)                       # [out, in], 0/1
        self.linear = nn.Linear(mask.shape[1], mask.shape[0])

    def forward(self, x):
        return F.linear(x, self.linear.weight * self.mask, self.linear.bias)


class PNet(nn.Module):
    def __init__(self, mask: torch.Tensor, hidden: int, dropout: float):
        super().__init__()
        n_genes = mask.shape[1]
        # gene layer: one 2->1 combination per gene (own weights, not shared)
        self.gene_w = nn.Parameter(torch.randn(n_genes, 2) * 0.1)
        self.gene_b = nn.Parameter(torch.zeros(n_genes))
        self.gene_act = nn.Tanh()
        self.pathway = MaskedLinear(mask)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mask.shape[0], hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):                     # x: [N, G, 2]
        g = self.gene_act((x * self.gene_w).sum(-1) + self.gene_b)   # [N, G]
        p = self.pathway(g)                                          # [N, P]
        return self.head(p).squeeze(-1)                               # [N]


def train_fold(Xtr, ytr, Xte, mask_t, cfg, seed):
    """Train with weighted BCE and early stopping on an inner validation split."""
    torch.manual_seed(seed)
    dev = torch.device("cpu")

    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.2,
                                stratify=ytr, random_state=seed)
    model = PNet(mask_t, cfg["model_c"]["hidden"], cfg["model_c"]["dropout"]).to(dev)
    pos = float(ytr[itr].sum())
    neg = float(len(itr) - pos)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["model_c"]["lr"],
                           weight_decay=cfg["model_c"]["weight_decay"])

    Xt = torch.tensor(Xtr[itr]); yt = torch.tensor(ytr[itr], dtype=torch.float32)
    Xv = torch.tensor(Xtr[iva]); yv = torch.tensor(ytr[iva], dtype=torch.float32)

    best, best_state, wait = np.inf, None, 0
    bs = cfg["model_c"]["batch_size"]
    for epoch in range(cfg["model_c"]["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xv), yv).item()
        if vl < best - 1e-5:
            best, wait = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["model_c"]["patience"]:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        s = torch.sigmoid(model(torch.tensor(Xte))).numpy()
    return s, epoch + 1, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    torch.set_num_threads(8)
    log = get_logger("04_model_c_pnet", cfg)

    expr, met, clin, lab = load_processed(cfg)
    y = lab["label"].values.astype(int)

    # Model C operates on genes measured by BOTH modalities.
    shared = sorted(set(expr.columns) & set(met.columns))
    log.info("Model C gene space: %d genes with expression AND methylation", len(shared))
    E = expr[shared].values.astype(np.float32)
    M = met[shared].values.astype(np.float32)

    sets = load_gene_sets(cfg, log)
    if not sets:
        raise SystemExit("no gene sets found -- run scripts/fetch_genesets.py first")

    folds = build_or_load_folds(y, cfg, log)
    if args.fast:
        folds = folds[: cfg["cv"]["n_splits"]]
        log.info("FAST mode: %d folds", len(folds))

    k = min(cfg["model_c"]["select_k"], len(shared))
    oof = np.full((len(y), len(folds)), np.nan, dtype=np.float32)
    rows = []

    for i, (tr, te) in enumerate(folds):
        # ---- feature selection INSIDE the fold (training half only) ----
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fe, _ = f_classif(E[tr], y[tr])
            fm, _ = f_classif(M[tr], y[tr])
        score = np.nanmax(np.vstack([np.nan_to_num(fe), np.nan_to_num(fm)]), axis=0)
        sel = np.argsort(score)[::-1][:k]
        genes_sel = [shared[j] for j in sel]

        mask, kept, connected = build_mask(genes_sel, sets, cfg,
                                           log if i == 0 else None)
        sel = sel[connected]
        mask = mask[:, connected]
        mask_t = torch.tensor(mask, dtype=torch.float32)

        # ---- scale on the training half only ----
        se, sm = StandardScaler().fit(E[tr][:, sel]), StandardScaler().fit(M[tr][:, sel])
        def pack(idx):
            return np.stack([se.transform(E[idx][:, sel]),
                             sm.transform(M[idx][:, sel])], axis=-1).astype(np.float32)

        s, n_ep, vl = train_fold(pack(tr), y[tr], pack(te), mask_t, cfg, cfg["seed"] + i)
        oof[te, i] = s
        m = fold_metrics(y[te], s)
        m.update({"fold_id": i, "repeat": i // cfg["cv"]["n_splits"],
                  "fold": i % cfg["cv"]["n_splits"], "epochs": n_ep,
                  "val_loss": round(float(vl), 4), "n_pathways": mask.shape[0],
                  "n_genes_connected": mask.shape[1]})
        rows.append(m)
        if i % 5 == 0 or i == len(folds) - 1:
            log.info("  fold %2d/%d auc=%.3f ap=%.3f epochs=%d pathways=%d genes=%d",
                     i + 1, len(folds), m["auc"], m["ap"], n_ep, mask.shape[0], mask.shape[1])

    per_fold = pd.DataFrame(rows)
    save_table(per_fold, cfg, "model_c_per_fold.csv", log)
    summ = summarise(per_fold, "C_pnet_pathway_nn")
    summ["mean_epochs"] = round(float(per_fold["epochs"].mean()), 1)
    summ["mean_pathways"] = int(per_fold["n_pathways"].mean())
    save_json(summ, cfg, "model_c_summary.json", log)

    oof_mean = np.nanmean(oof, axis=1)
    pd.DataFrame({"patient": lab.index, "label": y, "risk_oof": oof_mean}).to_csv(
        repo_root() / cfg["paths"]["tables"] / "model_c_oof_predictions.csv", index=False)

    log.info("GATE 4: Model C AUC = %.4f [%.4f, %.4f]  AP = %.4f [%.4f, %.4f]  "
             "(mean %.0f epochs to early stop)",
             summ["auc_mean"], summ["auc_ci_lo"], summ["auc_ci_hi"],
             summ["ap_mean"], summ["ap_ci_lo"], summ["ap_ci_hi"], summ["mean_epochs"])
    if summ["auc_mean"] > 0.85:
        flag(log, f"Model C AUC {summ['auc_mean']:.3f} > 0.85 -- check for leakage.")


if __name__ == "__main__":
    main()
