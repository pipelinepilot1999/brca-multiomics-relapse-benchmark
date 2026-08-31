# Multi-Omics Predictive Modelling for Breast Cancer Relapse

### Benchmarking transcriptomic and epigenomic signatures against clinical staging, with external validation and unsupervised representation learning

**TCGA-BRCA discovery (n = 367 labelled / 777 with matched assays) → METABRIC external validation (n = 1,902)**

A pre-specified, leakage-controlled benchmark asking whether RNA-seq and DNA
methylation add prognostic value over the clinical variables already recorded at
diagnosis. **They do not.** The pipeline, the negative result, and the
diagnostics that establish it are all reproducible from this repository — the
full analysis runs offline on a bundled fixture in under a minute.

> **A note on the title.** An earlier version of this project was called
> *"Multi-Omics Biomarker Signature for Breast Cancer Relapse."* The ablation in
> Phase 9 showed methylation contributes +0.0015 AUC over expression alone, and
> that no omics combination adds incremental value over clinical staging. The
> title was changed to match the evidence rather than the ambition.

---

## Abstract

Using 367 TCGA-BRCA patients with matched RNA-seq, Illumina 450k methylation and a
usable 3-year relapse label derived from PFI, three models were compared on
**identical** outer cross-validation folds (5-fold × 10 repeats = 50 folds). A
clinical-only logistic regression (age, stage, PAM50) reached **AUC 0.779**
[0.762–0.796]. An XGBoost model on all 19,338 clinical + expression + methylation
features reached **AUC 0.669** [0.648–0.690], and a pathway-informed neural network
(P-NET architecture, MSigDB-masked layer) reached **AUC 0.613** [0.593–0.633].
**The clinical baseline beat both omics models** — by 0.110 AUC over XGBoost
(paired t p = 3.0e-11; XGBoost won only 8 of 50 folds) and 0.166 over the neural
network (p = 9.2e-18; won 2 of 50 folds). A reduced 250-marker panel recovered
AUC 0.687, still below the clinical baseline. In external validation the
expression component transferred weakly but real: best METABRIC AUC 0.632
(most configurations 0.48–0.59), and the best-transferring model **did**
separate risk groups in METABRIC (log-rank p = 4.7e-06, against p = 1.8e-4 in
TCGA). The honest headline is therefore mixed: **at this sample size, adding
~19,000 omics features did not improve on a two-variable clinical model** — but
the reduced expression panel does carry genuine, externally reproducible
prognostic signal, just weaker than the clinical variables already available.

---

## The headline result, stated plainly

| Model | Description | AUC (95% CI) | AP (95% CI) |
|---|---|---|---|
| **A** | Logistic regression, clinical only (age, stage, PAM50) | **0.779 [0.762, 0.796]** | 0.546 [0.516, 0.576] |
| **B** | XGBoost, clinical + expression + methylation (19,338 features) | 0.669 [0.648, 0.690] | 0.364 [0.338, 0.389] |
| **C** | Pathway-informed neural network (P-NET style, MSigDB mask) | 0.613 [0.593, 0.633] | 0.308 [0.288, 0.329] |

Paired comparisons on identical folds:

| Comparison | Mean AUC difference | Paired t p | Wilcoxon p | Folds won by 2nd model |
|---|---|---|---|---|
| A vs B | −0.110 | 3.0e-11 | 1.0e-10 | 8 / 50 |
| A vs C | −0.166 | 9.2e-18 | 1.2e-14 | 2 / 50 |
| B vs C | −0.056 | 8.2e-11 | 2.8e-10 | 4 / 50 |

XGBoost beating the neural network is the outcome the build spec anticipated at
this sample size, and it is reported as found. The network was **not** retuned
until it won.

---

## Deviation from spec: the relapse horizon

The spec targeted a **5-year** horizon and expected n = 550–750 with 15–25%
positives. That is not achievable in TCGA-BRCA once matched methylation and a
correct early-censoring rule are both enforced. Measured over the 782-patient
universe with both modalities:

| Horizon | n | Positives | Prevalence | Excluded (early censoring) | n ≥ 400 | pos ≥ 60 |
|---|---|---|---|---|---|---|
| 2 y | 460 | 49 | 10.7% | 317 (41%) | yes | **no** |
| **3 y (primary)** | **367** | **68** | **18.5%** | **410 (53%)** | **no** | **yes** |
| 4 y | 297 | 80 | 26.9% | 480 (62%) | no | yes |
| 5 y (spec target) | 254 | 94 | 37.0% | 523 (67%) | **no** | yes |

**No horizon meets both floors.** The binding constraint is the multi-omics
intersection (only 782 TCGA-BRCA patients have both assays), not the horizon.
3 years was chosen as primary because it clears the positives floor and is the
only horizon whose prevalence falls inside the expected 15–25% band; it misses
the n ≥ 400 floor by 33 patients. **The full 5-year analysis is retained as a
sensitivity run** in `results/tables_5y/` and shows the same ordering
(A 0.689 > B 0.626 > C 0.619).

## The early-censoring exclusion, and what it cost

| PFI event | PFI time | Label |
|---|---|---|
| 1 | ≤ 1095 d | positive (relapsed) |
| 0 | ≥ 1095 d | negative (relapse-free) |
| 0 | < 1095 d | **EXCLUDED — censored early, status unknown** |
| 1 | > 1095 d | negative |

This exclusion removed **586 of 953 labelled patients (61%)** at the 3-year
horizon (744 of 1,091 at 5 years). A patient censored at 18 months is not a
negative; their 3-year status is unknown. Dropping them is correct and expensive.

It also has a consequence that must be stated: because negatives are required to
have full follow-up, the surviving cohort is **selected**. Lower-stage patients
are over-represented among negatives, which inflates both the prevalence and the
apparent strength of stage as a predictor. This is the main reason the clinical
baseline is stronger here (0.779) than the 0.60–0.70 the spec expected.

---

## Was the strong clinical baseline leakage? No.

Model A exceeded the expected band, so it was investigated before any omics model
was fit (`results/tables/gate2_baseline_variants.csv`):

| Variant | AUC |
|---|---|
| A_full (age + stage + PAM50) | 0.779 |
| A_no_PAM50 (age + stage only — *true* clinical) | 0.769 |
| A_PAM50_only | 0.571 |
| A_stage_only | 0.718 |
| A_age_only | 0.637 |
| A_full excluding stage IV | 0.744 |

PAM50 — which is expression-derived and arguably should not sit in a "clinical"
baseline at all — adds only **+0.009**. The signal is carried by pathologic stage,
whose gradient in this cohort is steep: stage I relapses at 3.2%, stage II 13.2%,
stage III 31.8%, stage IV 90% (9 of 10 patients). This is a real property of the
selected cohort, not a code defect.

---

## The signature

Performance versus panel size (selection re-done inside every fold; the reported
panel is the cross-fold consensus):

| Panel size | AUC (95% CI) |
|---|---|
| 20 | 0.662 [0.643, 0.682] |
| 50 | 0.668 [0.646, 0.690] |
| 100 | 0.683 [0.662, 0.705] |
| **250 (best)** | **0.687 [0.666, 0.709]** |
| 500 | 0.682 [0.661, 0.704] |

Performance plateaus at **100 markers**. The 50-marker panel is 31 expression,
17 methylation and 2 clinical features. Notably, reducing from 19,338 features to
250 *improved* AUC (0.669 → 0.687), which is consistent with most of the omics
block being noise at this sample size.

**Stability caveat, and it is a serious one.** Only four features are selected in
a majority of folds: `clin:stage` (100%), `meth:MIR1284` (100%), `expr:TNFSF11`
(98%) and `expr:SMN2` (80%). Most highly-ranked features by SHAP appear in only
2–16% of folds. The panel is therefore **not a stable signature** — it is a
different set of markers on almost every resample. Reporting a fixed marker list
without this caveat would misrepresent the result.

---

## External validation (METABRIC)

**METABRIC has no matched Illumina 450k methylation.** Only the expression
component could be tested; 32–153 methylation markers per panel are untestable
externally. (cBioPortal hosts a promoter-RRBS profile for METABRIC, but RRBS on a
sample subset is a different assay and is not a substitute — it was not used.)

Validation cohort: n = 1,902, 359 positives (18.9% — closely matching TCGA's
18.5%). Gene symbol mapping was good: 92–100% of expression markers mapped.

| Panel | Model | Genes | METABRIC AUC | TCGA in-sample AUC |
|---|---|---|---|---|
| 20 | logreg L2 | 9 | **0.632** | 0.756 |
| 20 | XGBoost | 9 | 0.558 | 1.000 |
| 50 | XGBoost | 29 | 0.558 | 1.000 |
| 100 | XGBoost | 62 | 0.579 | 1.000 |
| 250 | XGBoost | 162 | 0.585 | 1.000 |
| 500 | logreg L2 | 319 | 0.481 | 1.000 |

Two things to read here. First, XGBoost's TCGA in-sample AUC is **exactly 1.000** —
it memorises the training cohort completely, and transfers at ~0.56. Second, the
best external result (0.632) comes from the *smallest* panel with the *linear*
model, which is what one expects when most of the signal is noise.

Kaplan–Meier by predicted risk group. Log-rank is computed for **every**
panel/model configuration rather than one chosen in advance — tying the survival
conclusion to a single pre-picked model would make it an artifact of that choice:

| Panel | Model | METABRIC AUC | log-rank p |
|---|---|---|---|
| **20** | **logreg L2** | **0.632** | **4.7e-06** |
| 250 | XGBoost | 0.585 | 3.5e-03 |
| 500 | XGBoost | 0.568 | 2.0e-03 |
| 100 | XGBoost | 0.579 | 4.6e-02 |
| 50 | XGBoost | 0.558 | 0.69 |
| 20 | XGBoost | 0.558 | 0.27 |
| 100 | logreg L2 | 0.533 | 0.63 |
| 50 | logreg L2 | 0.531 | 0.82 |
| 250 | logreg L2 | 0.522 | 0.42 |
| 500 | logreg L2 | 0.481 | 0.82 |

- **TCGA** (out-of-fold risk): log-rank **p = 1.8e-4**.
- **METABRIC** (external, best-transferring model): log-rank **p = 4.7e-06** —
  **the separation does replicate.** The 5-year sensitivity run agrees
  (p = 1.3e-05).

Note the tight coupling between transfer AUC and log-rank significance: the four
configurations that clear AUC 0.567 all separate survival, and every
configuration at or below AUC 0.558 does not. That coherence is reassuring — it
means the surviving signal is a property of the model's discrimination, not a
lucky split. Cross-platform RNA-seq → microarray transfer still degrades
performance independently of biology, so AUC 0.63 externally versus 0.67
internally confounds signature quality with platform shift.

---

## Immune deconvolution

**Method deviation, stated plainly.** The spec asks for `immunedeconv` (R) with
quanTIseq or MCP-counter. The immunedeconv install failed on this machine —
Bioconductor dependencies `limSolve`, `biomaRt`, `sva`, `xCell`, `ConsensusTME`,
`ComICS`, `quantiseqr` were unavailable. MCP-counter was therefore reimplemented
directly in Python from the published 111-transcript, 10-population marker set
(Becht et al., *Genome Biology* 2016) — the same signature the R package wraps.
MCP-counter is a marker-gene aggregate rather than a fitted deconvolution, so the
reimplementation is faithful, not approximate. **quanTIseq was not run**: it needs
the TIL10 signature matrix shipped with the R package.

Splitting at median predicted risk (high-risk relapse rate 27.9% vs low-risk 9.2%),
**6 of 10 populations differ at FDR < 0.05**, all in the same direction — high-risk
tumours are immune- and stroma-**depleted**:

| Population | High risk | Low risk | Δ | FDR |
|---|---|---|---|---|
| Endothelial cells | 8.82 | 9.25 | −0.43 | 7.5e-5 |
| Myeloid dendritic cells | 5.73 | 6.38 | −0.65 | 1.3e-3 |
| Neutrophils | 5.54 | 5.84 | −0.29 | 2.4e-3 |
| B lineage | 5.82 | 6.37 | −0.55 | 1.9e-2 |
| NK cells | 2.77 | 3.04 | −0.27 | 2.9e-2 |
| Fibroblasts | 13.93 | 14.26 | −0.33 | 4.0e-2 |

**Gate 6 verdict:** the signature is at least partly an immune-infiltration
readout. Immune-cold tumours are predicted high-risk. That is consistent with
known breast cancer biology and is a real finding — but it also means the model is
substantially re-deriving infiltration, not discovering novel relapse biology.

---

## Unsupervised multi-omics representation learning (Phase 10)

The supervised pipeline is bottlenecked by n=367, because the landmark rule
discards every patient censored before the horizon. **An autoencoder needs no
labels**, so it trains on every patient with both assays — **777 instead of 367,
a 2.1× increase**. The label bottleneck simply does not bind on the
unsupervised half.

**Architecture** — denoising multi-modal autoencoder: separate encoders for
expression and methylation (4,000 highest-variance features each) → 256-unit
hidden layers → concatenated → **30 shared latent factors** → two decoders
reconstructing each modality. Loss is summed reconstruction MSE; inputs are
corrupted with Gaussian noise; early stopping on a held-out *reconstruction*
split (no labels anywhere in training).

### Result 1 — learned factors beat raw features

| Representation | Features | AUC |
|---|---|---|
| Raw expression + methylation (logistic) | 19,331 | 0.616 |
| Raw expression + methylation (XGBoost) | 19,331 | 0.637 |
| **AE latent factors (logistic)** | **30** | **0.664** |

Compressing 19,331 measurements into 30 learned factors **improved** AUC. This
is the clearest evidence in the project that the raw feature space is dominated
by noise, and that unsupervised integration extracts real structure from it.

### Result 2 — transduction did not inflate anything

Training one encoder on all 777 patients and then cross-validating on its
embeddings is *transductive*: the encoder saw the **features** (never the labels)
of patients who later appear in test folds. Reporting only that number would
overstate the result, so the encoder was **also refit inside each fold** on
training patients only, excluding that fold's test patients entirely:

| Encoder | AUC (identical 10 folds) |
|---|---|
| Transductive (fit once on all 777) | 0.6651 |
| Fold-safe (refit per fold, test patients excluded) | 0.6697 |
| **Transduction gap** | **−0.005** |

The gap is essentially zero — the transductive result was not inflated. This is
the check that makes the unsupervised numbers trustworthy.

### Result 3 — the factors are immune biology, rediscovered

Latent factors correlate strongly with MCP-counter cell populations (|r| up to
0.46: endothelial cells, CD8 T cells, fibroblasts, monocytic lineage) but
**barely at all with stage** (|r| ≤ 0.12). The autoencoder, given no labels and
no immune annotation, independently recovered the same immune/stromal axis that
Gate 6 found by an entirely different route. That convergence is strong evidence
the axis is real — and it also explains why the factors do not add to stage:
they encode a *different*, weaker signal.

### Result 4 — it still does not beat clinical

| Model | AUC | vs clinical |
|---|---|---|
| Clinical only | 0.7776 | — |
| AE latent only | 0.6640 | −0.114 (p = 4.1e-11) |
| **AE latent + clinical** | **0.7810** | **+0.003 (p = 0.68, n.s.)** |
| AE latent + clinical (fold-safe, 10 folds) | 0.7914 | wide CI [0.732, 0.851] |

Adding 30 learned multi-omics factors to age + stage + PAM50 improves AUC by
**+0.003, which is not significant**. The fold-safe variant reaches 0.7914 — the
highest number in the project — but on only 10 folds with a CI wide enough to
include no effect, so it is reported, not claimed.

**Conclusion.** Representation learning substantially improves the *omics* story
(0.637 → 0.664 with 640× fewer features) and is methodologically clean, but it
does not overturn the headline: clinical stage remains the dominant predictor,
and multi-omics adds no significant increment on top of it.

---

## Does each omics block earn its place? (Phase 9 ablation)

Seven feature blocks, two model families, all on the **same 50 outer folds**.
Clinical covariates are *forced into* every combined model (a custom selector
retains them before univariate selection touches the omics), so a
"clinical + omics" contrast cannot silently drop stage and answer a different
question than the one asked.

| Feature block | Logistic L2 | XGBoost |
|---|---|---|
| **Clinical only** | **0.778** | 0.742 |
| Expression only | 0.614 | 0.638 |
| Methylation only | 0.610 | 0.595 |
| Expression + methylation | 0.616 | 0.637 |
| Clinical + expression | 0.647 | 0.676 |
| Clinical + methylation | 0.659 | 0.643 |
| Clinical + expression + methylation | 0.645 | 0.670 |

### Paired contrasts

| Question | Model | ΔAUC | p | Omics wins |
|---|---|---|---|---|
| Does methylation add to expression? | logistic | +0.0015 | 0.82 | 27/50 |
| Does methylation add to expression? | XGBoost | −0.0013 | 0.85 | 26/50 |
| Does expression add to clinical? | logistic | −0.130 | 7.8e-16 | 3/50 |
| Does methylation add to clinical? | logistic | −0.118 | 3.5e-14 | 3/50 |
| **Does ANY omics add to clinical?** | logistic | **−0.133** | **3.1e-16** | **2/50** |
| **Does ANY omics add to clinical?** | XGBoost | **−0.073** | **2.5e-07** | **8/50** |

Two clean answers:

1. **Methylation adds nothing over expression** — +0.0015 / −0.0013, opposite
   signs across model families, both p > 0.8. The second omics layer does not
   pay for itself. "Multi-omics" is honest as a description of the inputs, but
   is *not* supported as a performance claim.
2. **No omics combination adds incremental value over clinical.** Even with
   stage forced into the model and strong regularisation applied, omics *hurts*
   significantly, winning on 2–8 of 50 folds. This is the version of the test
   that matters clinically, and it fails.

## Does a locked threshold transfer? (Phase 12)

Every risk group elsewhere in this report came from a median split computed
*within* each cohort. That is not how a deployed test works: a real signature
carries a fixed cutpoint. So the model is fit once on all of TCGA, the
**coefficients and a single threshold number are frozen**, and that exact number
is applied to METABRIC.

| Panel | Genes | METABRIC AUC | Locked cutpoint p | Own-median p | Agree? |
|---|---|---|---|---|---|
| **20** | **9** | **0.632** | **1.2e-05** | 4.7e-06 | **yes** |
| 50 | 29 | 0.531 | 0.745 | 0.818 | yes (both fail) |
| 100 | 62 | 0.533 | 0.883 | 0.628 | yes (both fail) |
| 250 | 162 | 0.522 | 0.55 | 0.42 | yes (both fail) |
| 500 | 319 | 0.481 | 0.942 | 0.824 | yes (both fail) |

**The 9-gene panel with a frozen cutpoint (0.4101) works.** In 1,902 independent
METABRIC patients it separates **23.9% relapse (high risk) from 12.2% (low risk)
— a risk ratio of 1.95**, log-rank p = 1.2e-05.

And the locked and data-driven splits **agree at every panel size**, with score
shifts of only +0.02 to +0.04. So cutpoint transfer is *not* the failure mode —
panel quality is. An earlier version of this analysis tested threshold-locking
using the 50-marker panel alone, which Gate 7 had already shown does not
transfer; that conflated "does this panel work" with "does a fixed threshold
work", and testing every panel size is what separates them.

---

## Reproduce it

### Smoke test — the fast path (no download, ~50 s)

The repo ships a 13 MB fixture (`data/smoke/`) subsampled from the real data in
the same file formats, so the **entire** pipeline — Gate 0 through the PDF —
runs offline:

```bash
bash scripts/smoke_test.sh
```

### Docker one-liner

```bash
docker build -f docker/Dockerfile -t biomarker-brca:1.0 . && docker run --rm -u $(id -u):$(id -g) -v "$PWD":/work -w /work --entrypoint bash biomarker-brca:1.0 scripts/smoke_test.sh
```

On Apple Silicon add `--platform linux/amd64` to the build. The container
reproduces the host numbers exactly (A 0.7428 / B 0.6889 / C 0.5908 on the fixture).

### Nextflow

```bash
cd nextflow && nextflow run main.nf -profile smoke
```

Profiles: `standard` (local), `docker`, `conda`, `smoke`. Execution report,
timeline, trace and DAG are written to `results/nextflow/`.

Full run on real data:

```bash
bash scripts/download_data.sh && cd nextflow && nextflow run main.nf -profile standard
```

### Full analysis directly

```bash
python scripts/00_verify_data.py --config config/config.yaml --skip-md5
python scripts/01_assemble.py --config config/config.yaml
# ... phases 02-09, or just use the Nextflow pipeline
```

---

## Repository layout

```
config/         config.yaml (3y primary), config_5y.yaml (sensitivity), config_smoke.yaml
data/raw/       downloaded, never edited (3.9 GB, gitignored)
data/smoke/     13 MB fixture, committed, runs the full pipeline offline
scripts/        one script per phase; each takes --config, writes results/, logs every filter
nextflow/       main.nf (DSL2) + nextflow.config
docker/         pinned Dockerfile
results/        tables/ figures/ logs/ + PDF reports (3-year and 5-year)
env/            requirements.txt (pip freeze), requirements-docker.txt, environment.yml
```

**Reports:** `results/biomarker_signature_report_3-year.pdf` (primary) and
`results/biomarker_signature_report_5-year.pdf` (sensitivity).

## Data sources

| Dataset | Source |
|---|---|
| TCGA-BRCA STAR counts | UCSC Xena GDC hub |
| TCGA-BRCA methylation 450k | UCSC Xena GDC hub (2.99 GB, 486,427 probes × 893) |
| TCGA-CDR endpoints (PFI) | PanCanAtlas, Liu et al. *Cell* 2018 |
| PAM50 subtypes | TCGA PanCanAtlas subtype calls |
| 450k manifest | Illumina via NCBI GEO GPL13534 (`UCSC_RefGene_Group`) |
| GENCODE v36 | EBI — matches the GDC annotation |
| MSigDB Hallmark + Reactome | Enrichr via gseapy (50 + 2,100 sets) |
| METABRIC | cBioPortal datahub (expression + RFS) |

Exact URLs, sizes, shapes and md5s: `results/tables/gate0_data_manifest.csv`.

---

## How leakage was prevented

This is the single most likely way a project like this produces a fake AUC.

- **All** feature selection, imputation and scaling happen inside `sklearn`
  Pipelines, refit on the training half of every split — outer *and* inner.
- Models A, B and C are scored on **identical fold indices**, written once to
  `results/tables/cv_folds.json` and loaded by each model, not regenerated from a
  seed. This is what makes the paired tests valid.
- Signature panels are re-selected within each fold; the reported panel is the
  cross-fold consensus. Scoring a globally-chosen panel would leak the test folds
  into the panel definition.
- Nested CV: inner loop tunes hyper-parameters, outer loop measures performance.
- The reported METABRIC numbers come from a model that never saw METABRIC.

The one number in this repo that *is* optimistic by construction — XGBoost's
in-sample TCGA AUC of 1.000 — is labelled as such wherever it appears.

---

## Limitations

- **Sample size (n = 367) against ~19,000 features** is the dominant limitation and
  the most likely reason the omics models lose to a two-variable clinical model.
- **The early-censoring rule discards 61% of labelled patients** and leaves a
  selected cohort in which stage is unusually predictive.
- **The signature is unstable** — most top-ranked markers appear in <20% of folds.
- **External discrimination is weak** (METABRIC AUC 0.632 vs 0.669 internally),
  even though risk-group separation replicates with a locked cutpoint
  (RR 1.95, p = 1.2e-05). The signature is externally *reproducible* but not
  externally *strong*.
- **Omics adds no incremental value over clinical** (ΔAUC −0.073 to −0.133, all
  p < 1e-6, winning 2–8 of 50 folds). This is the decisive negative: the panel
  requires an RNA-seq assay to underperform a number already on the pathology
  report.
- **Only expression was externally testable**; a multi-omics signature validated on
  one modality is partially validated at best.
- **Cross-platform transfer** confounds signature quality with platform shift.
- **PAM50 is expression-derived** but sits in the clinical baseline because the
  spec lists it; its contribution is quantified (+0.009) rather than hidden.
- **Histologic grade is 100% missing** in TCGA-CDR for this cohort and was dropped,
  so the clinical baseline is thinner than specified.
- **immunedeconv/quanTIseq not run** — see the deconvolution section.
- **Single-cohort discovery**; no multi-cohort discovery or meta-analysis.

---

## Data provenance and citation

All data is **public, de-identified and open-access**. Raw files are downloaded
by `scripts/download_data.sh` and never committed; the 13 MB `data/smoke/`
fixture contains a subsample redistributed under the terms of its original
sources.

| Source | Reference |
|---|---|
| TCGA-BRCA expression, methylation, clinical | The Cancer Genome Atlas Network. *Nature* 490, 61–70 (2012). Accessed via UCSC Xena GDC hub. |
| TCGA-CDR curated endpoints (PFI) | Liu J. *et al.* An Integrated TCGA Pan-Cancer Clinical Data Resource. *Cell* 173, 400–416 (2018). |
| METABRIC expression + RFS | Curtis C. *et al.* *Nature* 486, 346–352 (2012); Pereira B. *et al.* *Nat Commun* 7, 11479 (2016). Accessed via cBioPortal. |
| Illumina 450k manifest | Illumina HumanMethylation450 v1.2, via NCBI GEO platform GPL13534. |
| GENCODE v36 | Frankish A. *et al.* *Nucleic Acids Res* 47, D766–D773 (2019). |
| MSigDB Hallmark + Reactome | Liberzon A. *et al.* *Cell Syst* 1, 417–425 (2015); via Enrichr/gseapy. |
| MCP-counter markers | Becht E. *et al.* *Genome Biol* 17, 218 (2016). |
| P-NET architecture | Elmarakeby H. *et al.* *Nature* 598, 348–352 (2021). |

Exact URLs, file sizes, shapes and md5 checksums for every input are recorded in
`results/tables/gate0_data_manifest.csv`.

Code is MIT licensed (see `LICENSE`). Data retains the licence of its source.

## Reproducibility

Seed 42, set and recorded in every script. Outer CV shared via a written fold
file. Environment pinned in `env/requirements.txt` (pip freeze) and
`env/requirements-docker.txt`. Analysis scripts write tables only; figures are
produced by a separate script that reads those tables, so a figure cannot
disagree with the number behind it.
