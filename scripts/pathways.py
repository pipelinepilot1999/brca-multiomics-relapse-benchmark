"""Pathway mask construction for the P-NET style model."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from utils import repo_root


def read_gmt(path: Path) -> dict[str, list[str]]:
    sets = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > 2:
                sets[f[0]] = [g for g in f[2:] if g]
    return sets


def load_gene_sets(cfg, logger=None) -> dict[str, list[str]]:
    d = repo_root() / cfg["paths"]["raw"] / "genesets"
    sets = {}
    for f in ("hallmark.gmt", "reactome.gmt"):
        p = d / f
        if p.exists():
            s = read_gmt(p)
            sets.update({f"{f.split('.')[0]}:{k}": v for k, v in s.items()})
            if logger:
                logger.info("loaded %d gene sets from %s", len(s), f)
    return sets


def build_mask(genes: list[str], sets: dict[str, list[str]], cfg, logger=None):
    """mask[p, g] = 1 if gene g belongs to pathway p.

    Pathways are kept only if their overlap with the supplied gene list falls
    inside [min_pathway_genes, max_pathway_genes]; genes in no surviving pathway
    are dropped, since they cannot reach the pathway layer.
    """
    gi = {g: i for i, g in enumerate(genes)}
    rows, kept = [], []
    lo = cfg["model_c"]["min_pathway_genes"]
    hi = cfg["model_c"]["max_pathway_genes"]
    for name, members in sets.items():
        idx = [gi[g] for g in set(members) if g in gi]
        if lo <= len(idx) <= hi:
            r = np.zeros(len(genes), dtype=np.float32)
            r[idx] = 1.0
            rows.append(r)
            kept.append(name)
    if not rows:
        raise SystemExit("no pathway survived the size filter")
    mask = np.vstack(rows)
    connected = mask.sum(axis=0) > 0
    if logger:
        logger.info("pathway mask: %d pathways x %d genes; %d genes connected "
                    "(%d dropped as unannotated), density %.4f",
                    mask.shape[0], mask.shape[1], int(connected.sum()),
                    int((~connected).sum()), float(mask.mean()))
    return mask, kept, connected
