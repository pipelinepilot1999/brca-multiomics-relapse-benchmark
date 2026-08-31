"""Outer cross-validation folds, generated ONCE and shared by Models A, B and C.

The spec requires identical fold indices across models, not merely the same
seed. So the folds are built here, written to results/tables/cv_folds.json,
and every model loads that file. If the file exists it is reused verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from utils import repo_root

FOLD_FILE = "cv_folds.json"


def build_or_load_folds(y: np.ndarray, cfg: dict, logger=None) -> list[tuple[np.ndarray, np.ndarray]]:
    path = repo_root() / cfg["paths"]["tables"] / FOLD_FILE
    if path.exists():
        with open(path) as fh:
            blob = json.load(fh)
        if blob["n_samples"] == len(y) and blob["seed"] == cfg["seed"]:
            folds = [(np.array(f["train"]), np.array(f["test"])) for f in blob["folds"]]
            if logger:
                logger.info("reusing %d existing outer folds from %s", len(folds), FOLD_FILE)
            return folds
        if logger:
            logger.warning("fold file exists but n_samples/seed changed -- regenerating")

    rskf = RepeatedStratifiedKFold(n_splits=cfg["cv"]["n_splits"],
                                   n_repeats=cfg["cv"]["n_repeats"],
                                   random_state=cfg["seed"])
    folds = [(tr, te) for tr, te in rskf.split(np.zeros(len(y)), y)]
    blob = {
        "seed": cfg["seed"],
        "n_samples": int(len(y)),
        "n_splits": cfg["cv"]["n_splits"],
        "n_repeats": cfg["cv"]["n_repeats"],
        "class_balance": {"n_pos": int(y.sum()), "n_neg": int(len(y) - y.sum())},
        "folds": [{"repeat": i // cfg["cv"]["n_splits"], "fold": i % cfg["cv"]["n_splits"],
                   "train": tr.tolist(), "test": te.tolist()}
                  for i, (tr, te) in enumerate(folds)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(blob, fh)
    if logger:
        logger.info("generated %d outer folds (%d-fold x %d repeats), wrote %s",
                    len(folds), cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"], FOLD_FILE)
    return folds
