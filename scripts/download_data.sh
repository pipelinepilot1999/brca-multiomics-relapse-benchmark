#!/usr/bin/env bash
# Download every raw input to data/raw/. ~3.9 GB. Nothing here is ever edited
# afterwards. Re-running skips files that already exist.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw/genesets
cd data/raw

get () {  # url outfile
  if [ -s "$2" ]; then echo "SKIP   $2 (already present)"; return; fi
  echo "FETCH  $2"
  curl -fsSL --retry 5 --retry-delay 10 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

GDC="https://gdc-hub.s3.us-east-1.amazonaws.com/download"
PAN="https://pancanatlas.xenahubs.net/download"
CBIO="https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/brca_metabric"

get "$GDC/TCGA-BRCA.survival.tsv.gz"        TCGA-BRCA.survival.tsv.gz
get "$GDC/TCGA-BRCA.clinical.tsv.gz"        TCGA-BRCA.clinical.tsv.gz
get "$GDC/TCGA-BRCA.star_counts.tsv.gz"     TCGA-BRCA.star_counts.tsv.gz
get "$GDC/TCGA-BRCA.methylation450.tsv.gz"  TCGA-BRCA.methylation450.tsv.gz   # 2.99 GB
get "$PAN/Survival_SupplementalTable_S1_20171025_xena_sp" TCGA-CDR_Survival_S1.tsv
get "$PAN/TCGASubtype.20170308.tsv.gz"      TCGASubtype.20170308.tsv.gz
get "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPL13nnn/GPL13534/suppl/GPL13534_HumanMethylation450_15017482_v.1.1.csv.gz" \
    GPL13534_450k_manifest.csv.gz
get "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_36/gencode.v36.annotation.gtf.gz" \
    gencode.v36.annotation.gtf.gz
get "https://raw.githubusercontent.com/ebecht/MCPcounter/master/Signatures/genes.txt" \
    MCPcounter_genes.txt
get "$CBIO/data_mrna_illumina_microarray.txt" metabric_data_mrna_illumina_microarray.txt
get "$CBIO/data_clinical_patient.txt"         metabric_data_clinical_patient.txt
get "$CBIO/data_clinical_sample.txt"          metabric_data_clinical_sample.txt

echo ""
echo "Raw data:"; ls -lh; du -sh .
echo ""
echo "Next: python scripts/fetch_genesets.py --config config/config.yaml"
echo "Then: python scripts/00_verify_data.py --config config/config.yaml"
