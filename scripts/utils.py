"""Shared utilities: config, seeding, logging, and filter accounting.

Standing rules enforced here:
  * every script takes a config path and logs what it did
  * a single seed is set and recorded everywhere
  * no filter drops samples silently -- FilterLog forces in/out counts
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def set_seed(seed: int) -> int:
    """Set every RNG we touch. Returns the seed so callers can record it."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # some ops lack determinism on CPU
    except ImportError:
        pass
    return seed


def get_logger(name: str, cfg: dict) -> logging.Logger:
    logdir = repo_root() / cfg["paths"]["logs"]
    logdir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    fh = logging.FileHandler(logdir / f"{name}.log", mode="w")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.info("=" * 70)
    logger.info("start %s | config=%s | seed=%s", name, cfg.get("_config_path"), cfg.get("seed"))
    return logger


class FilterLog:
    """Accounting for every filtering step.

    Never drop rows without calling .step(). The resulting table is a
    deliverable (Gate 1), not a debugging aid.
    """

    def __init__(self, logger: logging.Logger, name: str):
        self.logger = logger
        self.name = name
        self.rows: list[dict] = []

    def step(self, description: str, n_in: int, n_out: int, unit: str = "samples") -> None:
        rec = {
            "stage": self.name,
            "step": description,
            "unit": unit,
            "n_in": int(n_in),
            "n_out": int(n_out),
            "n_dropped": int(n_in - n_out),
        }
        self.rows.append(rec)
        self.logger.info(
            "FILTER | %-52s | %6d -> %6d (%s %d) [%s]",
            description, n_in, n_out, "dropped", n_in - n_out, unit,
        )

    def note(self, description: str, n: int, unit: str = "samples") -> None:
        """Record a count that is not a drop (e.g. 'positives')."""
        self.rows.append(
            {"stage": self.name, "step": description, "unit": unit,
             "n_in": int(n), "n_out": int(n), "n_dropped": 0}
        )
        self.logger.info("COUNT  | %-52s | %6d [%s]", description, n, unit)

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame(self.rows)


def flag(logger: logging.Logger, message: str) -> None:
    """Surface a surprising number loudly instead of smoothing it over."""
    bar = "!" * 70
    logger.warning(bar)
    logger.warning("FLAG: %s", message)
    logger.warning(bar)


def save_table(df, cfg: dict, filename: str, logger: logging.Logger | None = None):
    out = repo_root() / cfg["paths"]["tables"] / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    if logger:
        logger.info("wrote %s  shape=%s", out.relative_to(repo_root()), getattr(df, "shape", "?"))
    return out


def save_json(obj, cfg: dict, filename: str, logger: logging.Logger | None = None):
    out = repo_root() / cfg["paths"]["tables"] / filename
    out.parent.mkdir(parents=True, exist_ok=True)

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not serialisable: {type(o)}")

    with open(out, "w") as fh:
        json.dump(obj, fh, indent=2, default=default)
    if logger:
        logger.info("wrote %s", out.relative_to(repo_root()))
    return out


def tcga_patient_id(barcode: str) -> str:
    """TCGA barcodes vary in length across files (12/15/16 chars).

    Truncate to the patient level: TCGA-XX-YYYY. Collisions are expected
    (multiple aliquots per patient) and are logged by the caller.
    """
    parts = str(barcode).split("-")
    return "-".join(parts[:3])


def tcga_is_primary_tumour(barcode: str) -> bool:
    """Sample type code 01 = primary solid tumour. Position 4 of the barcode."""
    parts = str(barcode).split("-")
    if len(parts) < 4:
        return False
    return parts[3][:2] == "01"


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
