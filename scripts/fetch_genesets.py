"""Fetch MSigDB Hallmark + Reactome gene sets as GMT files.

Written to data/raw/genesets/ so the pathway mask for Model C is reproducible
from disk and the pipeline does not depend on network access at model time.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from utils import get_logger, load_config, repo_root, set_seed

HALLMARK_CANDIDATES = ["MSigDB_Hallmark_2020", "MSigDB_Hallmark_2019"]
REACTOME_CANDIDATES = ["Reactome_Pathways_2024", "Reactome_2022", "Reactome_2016"]


def write_gmt(lib: dict, path: Path) -> int:
    with open(path, "w") as fh:
        for name, genes in lib.items():
            fh.write("\t".join([name, "na"] + list(genes)) + "\n")
    return len(lib)


def fetch(candidates, outpath: Path, log):
    if outpath.exists() and outpath.stat().st_size > 0:
        n = sum(1 for _ in open(outpath))
        log.info("%s already present (%d sets)", outpath.name, n)
        return
    import gseapy
    for name in candidates:
        try:
            log.info("fetching Enrichr library %s ...", name)
            lib = gseapy.get_library(name=name, organism="Human")
            n = write_gmt(lib, outpath)
            log.info("wrote %s: %d gene sets from %s", outpath.name, n, name)
            return
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s failed: %s", name, exc)
    raise SystemExit(f"could not fetch any of {candidates}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("fetch_genesets", cfg)

    outdir = repo_root() / cfg["paths"]["raw"] / "genesets"
    outdir.mkdir(parents=True, exist_ok=True)
    fetch(HALLMARK_CANDIDATES, outdir / "hallmark.gmt", log)
    fetch(REACTOME_CANDIDATES, outdir / "reactome.gmt", log)
    log.info("gene sets ready in %s", outdir)


if __name__ == "__main__":
    main()
