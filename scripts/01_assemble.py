"""PHASE 1 / GATE 1 -- assemble matrices and define the 5-year relapse label.

Label rule (from PFI, TCGA-CDR):
    PFI==1 and PFI.time <= 1825  -> positive (relapsed within 5y)
    PFI==0 and PFI.time >= 1825  -> negative (relapse-free through 5y)
    PFI==0 and PFI.time <  1825  -> EXCLUDED (censored early; status unknown)
    PFI==1 and PFI.time >  1825  -> negative (relapsed, but after 5y)

The third row is the one that matters: a patient censored at 2 years is not a
negative, they are unknown. Excluding them is correct and costly.
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (FilterLog, flag, get_logger, load_config, repo_root,
                   save_json, save_table, set_seed, tcga_is_primary_tumour,
                   tcga_patient_id)

PROMOTER_GROUPS = {"TSS1500", "TSS200", "5'UTR", "1stExon"}


# --------------------------------------------------------------------------- #
# gene annotation
# --------------------------------------------------------------------------- #
def load_gencode(path: Path, log) -> pd.DataFrame:
    """Parse gene-level records from the GENCODE GTF: id, symbol, biotype, chrom."""
    recs = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if f[2] != "gene":
                continue
            attr = f[8]

            def grab(key: str) -> str:
                i = attr.find(key + ' "')
                if i < 0:
                    return ""
                j = attr.find('"', i + len(key) + 2)
                return attr[i + len(key) + 2: j]

            recs.append((grab("gene_id"), grab("gene_name"), grab("gene_type"), f[0]))
    ann = pd.DataFrame(recs, columns=["gene_id", "symbol", "biotype", "chrom"])
    ann["ensembl"] = ann["gene_id"].str.split(".").str[0]
    log.info("GENCODE: %d genes, %d protein_coding",
             len(ann), (ann["biotype"] == "protein_coding").sum())
    return ann


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def build_labels(cfg, log, flog: FilterLog) -> pd.DataFrame:
    root = repo_root()
    cdr = pd.read_csv(root / cfg["files"]["tcga_cdr"], sep="\t", low_memory=False)
    flog.step("TCGA-CDR rows (all cancer types)", len(cdr), len(cdr))

    brca = cdr[cdr["cancer type abbreviation"] == cfg["label"]["cancer_type"]].copy()
    flog.step("restrict to BRCA", len(cdr), len(brca))

    # Redacted patients are excluded by TCGA guidance.
    if "Redaction" in brca.columns:
        n0 = len(brca)
        brca = brca[brca["Redaction"].isna() | (brca["Redaction"].astype(str).str.strip() == "")]
        flog.step("drop TCGA-redacted patients", n0, len(brca))

    brca["patient"] = brca["_PATIENT"].astype(str)
    n0 = len(brca)
    brca = brca.drop_duplicates(subset="patient", keep="first")
    flog.step("de-duplicate to one row per patient", n0, len(brca))

    brca["PFI"] = pd.to_numeric(brca["PFI"], errors="coerce")
    brca["PFI.time"] = pd.to_numeric(brca["PFI.time"], errors="coerce")
    n0 = len(brca)
    brca = brca.dropna(subset=["PFI", "PFI.time"])
    flog.step("drop patients with missing PFI or PFI.time", n0, len(brca))

    h = cfg["label"]["horizon_days"]
    ev, t = brca["PFI"].astype(int), brca["PFI.time"].astype(float)

    positive = (ev == 1) & (t <= h)
    negative = ((ev == 0) & (t >= h)) | ((ev == 1) & (t > h))
    excluded = (ev == 0) & (t < h)                       # censored before horizon

    brca["label"] = np.where(positive, 1, np.where(negative, 0, -1))
    n_excl = int(excluded.sum())
    log.info("label rule at %d days: pos=%d neg=%d excluded_early_censor=%d",
             h, int(positive.sum()), int(negative.sum()), n_excl)

    lab = brca[brca["label"] >= 0].copy()
    flog.step(f"EXCLUDE censored before {h}d (status unknown)", len(brca), len(lab))
    flog.note("positives (relapse <= 5y)", int((lab["label"] == 1).sum()), "patients")
    flog.note("negatives (relapse-free through 5y)", int((lab["label"] == 0).sum()), "patients")

    keep = ["patient", "label", "PFI", "PFI.time",
            "age_at_initial_pathologic_diagnosis", "ajcc_pathologic_tumor_stage",
            "histological_grade", "gender"]
    keep = [c for c in keep if c in lab.columns]
    return lab[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# expression
# --------------------------------------------------------------------------- #
def build_expression(cfg, ann, log, flog: FilterLog) -> pd.DataFrame:
    root = repo_root()
    log.info("reading expression matrix (this takes a minute)...")
    expr = pd.read_csv(root / cfg["files"]["tcga_expression"], sep="\t",
                       index_col=0, low_memory=False)
    expr = expr.astype(np.float32)
    flog.step("expression: raw genes", expr.shape[0], expr.shape[0], "genes")
    flog.step("expression: raw samples (aliquots)", expr.shape[1], expr.shape[1], "aliquots")

    cols = list(expr.columns)
    prim = [c for c in cols if tcga_is_primary_tumour(c)]
    flog.step("expression: keep primary tumour (sample type 01)", len(cols), len(prim), "aliquots")
    expr = expr[prim]

    # collapse multiple aliquots per patient -> keep the first by sorted barcode
    pat = pd.Series([tcga_patient_id(c) for c in expr.columns], index=expr.columns)
    dup_patients = pat.value_counts()
    n_multi = int((dup_patients > 1).sum())
    if n_multi:
        log.info("expression: %d patients have >1 aliquot; keeping first by sorted barcode", n_multi)
        for p in dup_patients[dup_patients > 1].index[:10]:
            log.info("   multi-aliquot example %s -> %s", p, sorted(pat[pat == p].index))
    order = sorted(expr.columns)
    seen, keep_cols = set(), []
    for c in order:
        p = tcga_patient_id(c)
        if p not in seen:
            seen.add(p)
            keep_cols.append(c)
    flog.step("expression: collapse aliquots to one per patient",
              expr.shape[1], len(keep_cols), "aliquots")
    expr = expr[keep_cols]
    expr.columns = [tcga_patient_id(c) for c in expr.columns]

    # annotate genes
    # PAR_Y copies share an Ensembl accession with their chrX counterpart once the
    # version suffix is stripped, which would create duplicate index entries.
    raw_ids = expr.index.astype(str)
    par_y = raw_ids.str.contains("_PAR_Y")
    if par_y.any():
        flog.step("expression: drop chrY PAR_Y duplicate accessions",
                  len(expr), int((~par_y).sum()), "genes")
        expr = expr[~par_y]
    expr.index = expr.index.astype(str).str.split(".").str[0]
    n0 = len(expr)
    expr = expr[~expr.index.duplicated(keep="first")]
    if len(expr) != n0:
        flog.step("expression: drop residual duplicate Ensembl accessions", n0, len(expr), "genes")

    amap = ann.drop_duplicates("ensembl").set_index("ensembl")
    common = expr.index.intersection(amap.index)
    flog.step("expression: genes mappable to GENCODE v36", expr.shape[0], len(common), "genes")
    expr = expr.loc[common]
    meta = amap.loc[common]
    assert len(expr) == len(meta), (len(expr), len(meta))

    if cfg["assembly"]["expr_protein_coding_only"]:
        pc = meta["biotype"] == "protein_coding"
        flog.step("expression: keep protein_coding", len(expr), int(pc.sum()), "genes")
        expr, meta = expr[pc.values], meta[pc.values]

    expr.index = meta["symbol"].values
    n0 = len(expr)
    expr = expr.groupby(level=0).mean()          # duplicate symbols -> mean
    flog.step("expression: collapse duplicate gene symbols (mean)", n0, len(expr), "genes")

    v = expr.var(axis=1)
    n0 = len(expr)
    expr = expr[v > 0]
    flog.step("expression: drop zero-variance genes", n0, len(expr), "genes")

    q = cfg["assembly"]["expr_min_variance_quantile"]
    thr = expr.var(axis=1).quantile(q)
    n0 = len(expr)
    expr = expr[expr.var(axis=1) > thr]
    flog.step(f"expression: drop bottom {q:.0%} by variance", n0, len(expr), "genes")

    log.info("expression matrix ready: %d genes x %d patients", *expr.shape)
    log.info("expression value range: min=%.3f max=%.3f (expect log2(count+1))",
             float(expr.values.min()), float(expr.values.max()))
    return expr.T                                 # patients x genes


# --------------------------------------------------------------------------- #
# methylation
# --------------------------------------------------------------------------- #
def load_promoter_map(cfg, log, flog: FilterLog) -> pd.DataFrame:
    """probe -> gene, restricted to promoter-associated probes."""
    root = repo_root()
    man = pd.read_csv(root / cfg["files"]["methyl_manifest"], skiprows=7, low_memory=False,
                      usecols=["IlmnID", "UCSC_RefGene_Name", "UCSC_RefGene_Group", "CHR"],
                      encoding="latin-1")
    man = man.dropna(subset=["IlmnID"])
    flog.step("450k manifest probes", len(man), len(man), "probes")

    if cfg["assembly"]["drop_sex_chromosomes"]:
        n0 = len(man)
        man = man[~man["CHR"].astype(str).isin(["X", "Y"])]
        flog.step("manifest: drop chrX/chrY probes", n0, len(man), "probes")

    man = man.dropna(subset=["UCSC_RefGene_Name", "UCSC_RefGene_Group"])
    flog.step("manifest: probes with a RefGene annotation", len(man), len(man), "probes")

    groups = set(cfg["assembly"]["methyl_promoter_groups"])
    pairs = []
    for probe, names, grps in zip(man["IlmnID"].values,
                                  man["UCSC_RefGene_Name"].values,
                                  man["UCSC_RefGene_Group"].values):
        nl, gl = str(names).split(";"), str(grps).split(";")
        if len(nl) != len(gl):
            continue
        for gene, grp in zip(nl, gl):
            if grp in groups and gene:
                pairs.append((probe, gene))
    pmap = pd.DataFrame(pairs, columns=["probe", "gene"]).drop_duplicates()
    log.info("promoter probe->gene pairs: %d (%d unique probes, %d unique genes)",
             len(pmap), pmap["probe"].nunique(), pmap["gene"].nunique())
    flog.step("manifest: promoter-associated probes (%s)" % ",".join(sorted(groups)),
              man["IlmnID"].nunique(), pmap["probe"].nunique(), "probes")
    return pmap


def build_methylation(cfg, pmap, log, flog: FilterLog) -> pd.DataFrame:
    root = repo_root()
    wanted = set(pmap["probe"].unique())
    log.info("streaming methylation matrix, keeping %d promoter probes...", len(wanted))

    chunks, n_seen = [], 0
    reader = pd.read_csv(root / cfg["files"]["tcga_methylation"], sep="\t",
                         index_col=0, chunksize=20000, low_memory=False)
    for i, ch in enumerate(reader):
        n_seen += len(ch)
        sub = ch[ch.index.isin(wanted)]
        if len(sub):
            chunks.append(sub.astype(np.float32))
        if i % 5 == 0:
            log.info("  ... %d probes scanned, %d kept", n_seen,
                     sum(len(c) for c in chunks))
    met = pd.concat(chunks)
    flog.step("methylation: probes in file", n_seen, len(met), "probes")

    cols = list(met.columns)
    prim = [c for c in cols if tcga_is_primary_tumour(c)]
    flog.step("methylation: keep primary tumour (sample type 01)", len(cols), len(prim), "aliquots")
    met = met[prim]

    order = sorted(met.columns)
    seen, keep_cols = set(), []
    for c in order:
        p = tcga_patient_id(c)
        if p not in seen:
            seen.add(p)
            keep_cols.append(c)
    flog.step("methylation: collapse aliquots to one per patient",
              met.shape[1], len(keep_cols), "aliquots")
    met = met[keep_cols]
    met.columns = [tcga_patient_id(c) for c in met.columns]

    # missingness
    thr_p = cfg["assembly"]["methyl_max_na_frac_probe"]
    na_probe = met.isna().mean(axis=1)
    n0 = len(met)
    met = met[na_probe <= thr_p]
    flog.step(f"methylation: drop probes with >{thr_p:.0%} NA", n0, len(met), "probes")

    thr_s = cfg["assembly"]["methyl_max_na_frac_sample"]
    na_samp = met.isna().mean(axis=0)
    n0 = met.shape[1]
    met = met.loc[:, na_samp <= thr_s]
    flog.step(f"methylation: drop samples with >{thr_s:.0%} NA", n0, met.shape[1], "patients")

    n_na = int(met.isna().sum().sum())
    if n_na:
        log.info("methylation: imputing %d residual NA cells (%.4f%%) with probe mean",
                 n_na, 100 * n_na / met.size)
        met = met.T.fillna(met.mean(axis=1)).T

    # aggregate probes -> gene (mean beta)
    pm = pmap[pmap["probe"].isin(met.index)]
    joined = met.loc[pm["probe"].values].copy()
    joined.index = pm["gene"].values
    gene_met = joined.groupby(level=0).mean()
    log.info("methylation: %d promoter probes aggregated to %d genes",
             len(pm), len(gene_met))
    flog.step("methylation: aggregate promoter probes to gene mean beta",
              len(pm), len(gene_met), "genes")

    q = cfg["assembly"]["methyl_min_variance_quantile"]
    thr = gene_met.var(axis=1).quantile(q)
    n0 = len(gene_met)
    gene_met = gene_met[gene_met.var(axis=1) > thr]
    flog.step(f"methylation: drop bottom {q:.0%} by variance", n0, len(gene_met), "genes")

    log.info("methylation beta range: min=%.3f max=%.3f (expect 0-1)",
             float(gene_met.values.min()), float(gene_met.values.max()))
    return gene_met.T                              # patients x genes


# --------------------------------------------------------------------------- #
def build_clinical(cfg, labels, log, flog) -> pd.DataFrame:
    root = repo_root()
    cl = labels.copy().set_index("patient")

    ren = {"age_at_initial_pathologic_diagnosis": "age",
           "ajcc_pathologic_tumor_stage": "stage_raw",
           "histological_grade": "grade_raw"}
    cl = cl.rename(columns=ren)

    cl["age"] = pd.to_numeric(cl.get("age"), errors="coerce")

    def stage_num(s):
        s = str(s).upper().replace("STAGE", "").strip()
        for pat, val in [("IV", 4), ("III", 3), ("II", 2), ("I", 1)]:
            if s.startswith(pat):
                return val
        return np.nan

    cl["stage"] = cl["stage_raw"].map(stage_num) if "stage_raw" in cl else np.nan

    def grade_num(s):
        s = str(s).upper().strip()
        return {"G1": 1, "G2": 2, "G3": 3, "G4": 4}.get(s, np.nan)

    cl["grade"] = cl["grade_raw"].map(grade_num) if "grade_raw" in cl else np.nan

    # PAM50 from PanCanAtlas subtype calls
    sub = pd.read_csv(root / cfg["files"]["tcga_subtype"], sep="\t", low_memory=False)
    scol = next((c for c in sub.columns if "Subtype_Selected" in c or c == "Subtype_mRNA"), None)
    idcol = sub.columns[0]
    sub["patient"] = sub[idcol].astype(str).map(tcga_patient_id)
    sub = sub.drop_duplicates("patient").set_index("patient")
    if scol:
        pam = sub[scol].astype(str).str.replace("BRCA.", "", regex=False)
        cl["pam50"] = pam.reindex(cl.index).values
    else:
        cl["pam50"] = np.nan
    log.info("PAM50 available for %d / %d patients", cl["pam50"].notna().sum(), len(cl))

    for c in ["age", "stage", "grade"]:
        log.info("clinical %-6s: %d/%d present (%.1f%% missing)",
                 c, cl[c].notna().sum(), len(cl), 100 * cl[c].isna().mean())
    return cl.reset_index()


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset-patients", type=int, default=0,
                    help="smoke-test mode: cap the cohort at N patients")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    log = get_logger("01_assemble", cfg)
    root = repo_root()
    flog = FilterLog(log, "assembly")

    labels = build_labels(cfg, log, flog)
    ann = load_gencode(root / cfg["files"]["gencode_gtf"], log)
    expr = build_expression(cfg, ann, log, flog)
    pmap = load_promoter_map(cfg, log, flog)
    met = build_methylation(cfg, pmap, log, flog)

    # ---- intersect ----
    lab_p = set(labels["patient"])
    e_p, m_p = set(expr.index), set(met.index)
    log.info("patients: labels=%d expression=%d methylation=%d", len(lab_p), len(e_p), len(m_p))
    flog.note("patients with a usable label", len(lab_p), "patients")
    flog.note("patients with expression", len(e_p), "patients")
    flog.note("patients with methylation", len(m_p), "patients")

    final = sorted(lab_p & e_p & m_p)
    flog.step("INTERSECT label AND expression AND methylation",
              len(lab_p), len(final), "patients")

    if args.subset_patients:
        rng = np.random.default_rng(cfg["seed"])
        y_all = labels.set_index("patient").loc[final, "label"].values
        pos = [p for p, y in zip(final, y_all) if y == 1]
        neg = [p for p, y in zip(final, y_all) if y == 0]
        k_pos = max(1, int(round(args.subset_patients * len(pos) / len(final))))
        k_neg = args.subset_patients - k_pos
        final = sorted(list(rng.choice(pos, min(k_pos, len(pos)), replace=False)) +
                       list(rng.choice(neg, min(k_neg, len(neg)), replace=False)))
        log.info("SMOKE-TEST subset: %d patients", len(final))

    labels = labels[labels["patient"].isin(final)].set_index("patient").loc[final].reset_index()
    expr, met = expr.loc[final], met.loc[final]

    # align gene space between modalities
    shared_genes = sorted(set(expr.columns) & set(met.columns))
    log.info("gene space: expression=%d methylation=%d shared=%d",
             expr.shape[1], met.shape[1], len(shared_genes))
    flog.note("genes shared by both modalities", len(shared_genes), "genes")

    clinical = build_clinical(cfg, labels, log, flog)

    n_pos = int((labels["label"] == 1).sum())
    n_neg = int((labels["label"] == 0).sum())
    n = len(labels)
    prev = n_pos / n if n else 0.0

    # ---- write ----
    pdir = root / cfg["paths"]["processed"]
    pdir.mkdir(parents=True, exist_ok=True)
    expr.to_parquet(pdir / "expression.parquet")
    met.to_parquet(pdir / "methylation.parquet")
    clinical.to_parquet(pdir / "clinical.parquet")
    labels.to_parquet(pdir / "labels.parquet")
    pd.Series(shared_genes, name="gene").to_frame().to_parquet(pdir / "shared_genes.parquet")
    log.info("wrote processed matrices to %s", pdir.relative_to(root))

    save_table(flog.to_frame(), cfg, "gate1_filtering.csv", log)
    summary = {
        "n_final": n, "n_positive": n_pos, "n_negative": n_neg,
        "positive_rate": round(prev, 4),
        "n_excluded_early_censoring": int(
            flog.to_frame().query("step.str.contains('EXCLUDE censored')", engine="python")["n_dropped"].sum()),
        "n_expression_features": int(expr.shape[1]),
        "n_methylation_features": int(met.shape[1]),
        "n_shared_genes": len(shared_genes),
        "horizon_days": cfg["label"]["horizon_days"],
        "seed": cfg["seed"],
    }
    save_json(summary, cfg, "gate1_summary.json", log)

    log.info("GATE 1 SUMMARY: n=%d  pos=%d (%.1f%%)  neg=%d  expr_feats=%d  meth_feats=%d",
             n, n_pos, 100 * prev, n_neg, expr.shape[1], met.shape[1])

    if not args.subset_patients:
        if n < 400:
            flag(log, f"final n={n} is under 400. Spec says stop: consider a 3-year horizon.")
        if n_pos < 60:
            flag(log, f"only {n_pos} positives (<60). Spec says stop: task may not be viable.")
        if not (550 <= n <= 750):
            flag(log, f"final n={n} is outside the expected 550-750 band. Explain before modelling.")
        if not (0.15 <= prev <= 0.25):
            flag(log, f"positive rate {prev:.1%} is outside the expected 15-25% band.")


if __name__ == "__main__":
    main()
