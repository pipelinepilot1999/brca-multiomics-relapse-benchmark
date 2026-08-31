"""Build a small, committable fixture so the FULL pipeline runs without the 4 GB download.

Subsamples the real raw files into data/smoke/ keeping the same formats, so the
smoke test exercises Phase 0 -> 8 for real (assembly, labelling, CV, models,
signature, deconvolution, external validation) rather than starting from
pre-baked matrices.

Run once on a machine that has the full data; commit data/smoke/ to the repo.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (get_logger, load_config, repo_root, set_seed,
                   tcga_is_primary_tumour, tcga_patient_id)

N_PATIENTS = 130          # before labelling; yields ~100 usable
N_GENES = 3000
N_PROBES = 6000
N_METABRIC = 400


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(set_seed(cfg["seed"]))
    log = get_logger("make_smoke_fixture", cfg)
    root = repo_root()
    out = root / "data/smoke"
    out.mkdir(parents=True, exist_ok=True)

    # ---- choose the patient subset from the assembled cohort ----
    lab = pd.read_parquet(root / cfg["paths"]["processed"] / "labels.parquet")
    pos = lab[lab.label == 1]["patient"].tolist()
    neg = lab[lab.label == 0]["patient"].tolist()
    keep_pat = set(list(rng.choice(pos, min(45, len(pos)), replace=False))
                   + list(rng.choice(neg, min(N_PATIENTS - 45, len(neg)), replace=False)))
    log.info("smoke cohort: %d patients (%d pos, %d neg)", len(keep_pat),
             sum(p in set(pos) for p in keep_pat), sum(p in set(neg) for p in keep_pat))

    # ---- CDR ----
    cdr = pd.read_csv(root / cfg["files"]["tcga_cdr"], sep="\t", low_memory=False)
    cdr_s = cdr[cdr["_PATIENT"].astype(str).isin(keep_pat)]
    cdr_s.to_csv(out / "TCGA-CDR_Survival_S1.tsv", sep="\t", index=False)
    log.info("CDR subset: %d rows", len(cdr_s))

    # ---- expression ----
    ex = pd.read_csv(root / cfg["files"]["tcga_expression"], sep="\t", index_col=0,
                     low_memory=False)
    cols = [c for c in ex.columns if tcga_is_primary_tumour(c) and tcga_patient_id(c) in keep_pat]
    ex = ex[cols]
    var = ex.var(axis=1)
    ex = ex.loc[var.sort_values(ascending=False).index[:N_GENES]]
    ex.round(4).to_csv(out / "expression.tsv.gz", sep="\t", compression="gzip")
    log.info("expression subset: %s", ex.shape)
    keep_ens = {i.split(".")[0] for i in ex.index}

    # ---- gencode subset (only the genes retained) ----
    with gzip.open(root / cfg["files"]["gencode_gtf"], "rt") as fh, \
         gzip.open(out / "gencode.subset.gtf.gz", "wt") as fo:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene":
                continue
            i = f[8].find('gene_id "')
            gid = f[8][i + 9: f[8].find('"', i + 9)].split(".")[0]
            if gid in keep_ens:
                fo.write(line)
    log.info("gencode subset written")

    # ---- methylation ----
    man = pd.read_csv(root / cfg["files"]["methyl_manifest"], skiprows=7, low_memory=False,
                      usecols=["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "CHR"],
                      encoding="latin-1").dropna(subset=["IlmnID"])
    sym = {s for s in pd.read_parquet(root / cfg["paths"]["processed"]
                                      / "methylation.parquet").columns}
    man["first_gene"] = man["UCSC_RefGene_Name"].astype(str).str.split(";").str[0]
    man_s = man[man["first_gene"].isin(sym)].head(N_PROBES * 3)
    wanted = set(man_s["IlmnID"])

    chunks = []
    for ch in pd.read_csv(root / cfg["files"]["tcga_methylation"], sep="\t", index_col=0,
                          chunksize=50000, low_memory=False):
        sub = ch[ch.index.isin(wanted)]
        if len(sub):
            mcols = [c for c in sub.columns
                     if tcga_is_primary_tumour(c) and tcga_patient_id(c) in keep_pat]
            chunks.append(sub[mcols])
        if sum(len(c) for c in chunks) >= N_PROBES:
            break
    met = pd.concat(chunks).head(N_PROBES)
    met.round(4).to_csv(out / "methylation.tsv.gz", sep="\t", compression="gzip")
    log.info("methylation subset: %s", met.shape)

    man_s2 = man_s[man_s["IlmnID"].isin(met.index)]
    with gzip.open(out / "manifest.subset.csv.gz", "wt") as fo:
        for _ in range(7):
            fo.write("# smoke fixture preamble line\n")
        man_s2[["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "CHR"]] \
            .to_csv(fo, index=False)
    log.info("manifest subset: %d probes", len(man_s2))

    # ---- METABRIC ----
    mb = pd.read_csv(root / cfg["files"]["metabric_expr"], sep="\t", low_memory=False)
    mb = mb.rename(columns={mb.columns[0]: "Hugo_Symbol"})
    genes_needed = set(pd.read_parquet(root / cfg["paths"]["processed"]
                                       / "expression.parquet").columns)
    mb_s = mb[mb["Hugo_Symbol"].isin(genes_needed)].head(N_GENES)
    samp = [c for c in mb_s.columns if c.startswith("MB-")][:N_METABRIC]
    keep_cols = ["Hugo_Symbol"] + samp
    mb_s[keep_cols].to_csv(out / "metabric_expr.txt.gz", sep="\t", index=False,
                           compression="gzip")
    log.info("METABRIC subset: %d genes x %d samples", len(mb_s), len(samp))

    pat = pd.read_csv(root / cfg["files"]["metabric_patient"], sep="\t", comment="#",
                      low_memory=False)
    idc = pat.columns[0]
    pat[pat[idc].isin(samp)].to_csv(out / "metabric_patient.txt", sep="\t", index=False)

    sub = pd.read_csv(root / cfg["files"]["tcga_subtype"], sep="\t", low_memory=False)
    sub[sub[sub.columns[0]].astype(str).map(tcga_patient_id).isin(keep_pat)] \
        .to_csv(out / "TCGASubtype.subset.tsv.gz", sep="\t", index=False, compression="gzip")

    import shutil
    shutil.copy(root / cfg["files"]["mcp_genes"], out / "MCPcounter_genes.txt")
    gsd = root / "data/raw/genesets"
    if gsd.exists():
        (out / "genesets").mkdir(exist_ok=True)
        for f in gsd.glob("*.gmt"):
            shutil.copy(f, out / "genesets" / f.name)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    log.info("smoke fixture complete: %.1f MB in %s", total / 1e6, out.relative_to(root))
    for f in sorted(out.rglob("*")):
        if f.is_file():
            log.info("   %-34s %8.2f MB", f.relative_to(out), f.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
