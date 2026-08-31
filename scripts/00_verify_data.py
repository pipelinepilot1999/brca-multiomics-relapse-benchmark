"""PHASE 0 / GATE 0 -- verify raw data acquisition.

Reports every file downloaded: exact URL, size on disk, md5, and shape.
Writes results/tables/gate0_data_manifest.csv.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
from pathlib import Path

import pandas as pd

from utils import (flag, get_logger, load_config, repo_root, save_table, set_seed)

# Exact URLs used, recorded as a deliverable.
SOURCES = {
    "tcga_expression": ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.star_counts.tsv.gz",
                        "UCSC Xena, GDC TCGA-BRCA hub. STAR counts, log2(count+1)."),
    "tcga_methylation": ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.methylation450.tsv.gz",
                         "UCSC Xena, GDC TCGA-BRCA hub. Illumina HumanMethylation450 beta values."),
    "tcga_clinical": ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.clinical.tsv.gz",
                      "UCSC Xena, GDC TCGA-BRCA hub. Clinical/phenotype."),
    "tcga_survival": ("https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-BRCA.survival.tsv.gz",
                      "UCSC Xena, GDC TCGA-BRCA hub. OS only -- PFI comes from TCGA-CDR."),
    "tcga_cdr": ("https://pancanatlas.xenahubs.net/download/Survival_SupplementalTable_S1_20171025_xena_sp",
                 "TCGA-CDR curated endpoints (Liu et al. 2018, Cell). Source of PFI / PFI.time."),
    "tcga_subtype": ("https://pancanatlas.xenahubs.net/download/TCGASubtype.20170308.tsv.gz",
                     "TCGA PanCanAtlas molecular subtype calls (PAM50 for BRCA)."),
    "methyl_manifest": ("https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/"
                        "GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz",
                        "Illumina 450k manifest via NCBI GEO GPL13534. Carries UCSC_RefGene_Group."),
    "mcp_genes": ("https://raw.githubusercontent.com/ebecht/MCPcounter/master/Signatures/genes.txt",
                  "MCP-counter marker genes (Becht et al. 2016)."),
    "metabric_expr": ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
                      "brca_metabric/data_mrna_illumina_microarray.txt",
                      "METABRIC expression, Illumina HT-12 v3 microarray (cBioPortal datahub)."),
    "metabric_patient": ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
                         "brca_metabric/data_clinical_patient.txt",
                         "METABRIC patient clinical. Carries RFS_STATUS / RFS_MONTHS."),
    "metabric_sample": ("https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
                        "brca_metabric/data_clinical_sample.txt",
                        "METABRIC sample clinical (ER/HER2/grade/stage)."),
}


def md5(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def peek_shape(path: Path, logger) -> tuple[int | None, int | None]:
    """Rows x cols without loading the whole file into memory."""
    opener = gzip.open if path.suffix == ".gz" else open
    sep = "," if "manifest" in path.name else "\t"
    try:
        with opener(path, "rt", errors="replace") as fh:
            # The Illumina manifest has a 7-line preamble before the header.
            if "manifest" in path.name:
                for _ in range(7):
                    fh.readline()
            header = fh.readline().rstrip("\n").split(sep)
            n_rows = sum(1 for _ in fh)
        return n_rows, len(header)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not read %s: %s", path.name, exc)
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-md5", action="store_true", help="skip md5 on multi-GB files")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("00_verify_data", cfg)
    root = repo_root()

    rows = []
    missing = []
    for key, relpath in cfg["files"].items():
        path = root / relpath
        url, desc = SOURCES.get(key, ("(not recorded)", ""))
        if not path.exists():
            missing.append(key)
            log.error("MISSING %s -> %s", key, relpath)
            rows.append({"key": key, "file": relpath, "url": url, "description": desc,
                         "exists": False, "size_bytes": 0, "size_human": "-",
                         "n_rows": None, "n_cols": None, "md5": ""})
            continue
        size = path.stat().st_size
        n_rows, n_cols = peek_shape(path, log)
        digest = "" if (args.skip_md5 and size > 5e8) else md5(path)
        human = f"{size/1e6:.1f} MB" if size < 1e9 else f"{size/1e9:.2f} GB"
        log.info("%-18s %-10s rows=%-9s cols=%-6s %s", key, human, n_rows, n_cols, path.name)
        rows.append({"key": key, "file": relpath, "url": url, "description": desc,
                     "exists": True, "size_bytes": size, "size_human": human,
                     "n_rows": n_rows, "n_cols": n_cols, "md5": digest})

    manifest = pd.DataFrame(rows)
    save_table(manifest, cfg, "gate0_data_manifest.csv", log)

    if missing:
        flag(log, f"{len(missing)} raw file(s) missing: {missing}. Run scripts/download_data.sh first.")
        raise SystemExit(1)

    log.info("GATE 0 PASS: %d files present, %.2f GB total",
             len(manifest), manifest["size_bytes"].sum() / 1e9)


if __name__ == "__main__":
    main()
