#!/usr/bin/env bash
# Full pipeline on a 100-patient fixture. Target: under five minutes, no network,
# no 4 GB download. This is the entry point for anyone who wants to actually run
# the repo rather than read it.
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG="${1:-config/config_smoke.yaml}"
START=$(date +%s)

echo "=============================================================="
echo " biomarker-signature-brca :: SMOKE TEST"
echo " config: $CONFIG"
echo "=============================================================="

if [ ! -d data/smoke ]; then
  echo "ERROR: data/smoke/ fixture not found." >&2
  echo "Regenerate it from full data with:" >&2
  echo "  python scripts/make_smoke_fixture.py --config config/config.yaml" >&2
  exit 1
fi

run () {
  echo ""
  echo "---- $1 ----"
  shift
  python "$@"
}

export PYTHONPATH="$PWD/scripts:${PYTHONPATH:-}"

run "Gate 0: verify fixture"        scripts/00_verify_data.py         --config "$CONFIG" --skip-md5
run "Gate 1: assemble + label"      scripts/01_assemble.py            --config "$CONFIG"
run "Gate 1b: horizon sensitivity"  scripts/01b_horizon_sensitivity.py --config "$CONFIG"
run "Gate 2: Model A (clinical)"    scripts/02_model_a_clinical.py    --config "$CONFIG"
run "Gate 2b: baseline diagnostic"  scripts/02b_baseline_diagnostics.py --config "$CONFIG"
run "Gate 3: Model B (XGBoost)"     scripts/03_model_b_xgboost.py     --config "$CONFIG" --fast
run "Gate 4: Model C (pathway NN)"  scripts/04_model_c_pnet.py        --config "$CONFIG" --fast
run "Model comparison"              scripts/compare_models.py         --config "$CONFIG"
run "Gate 5: signature reduction"   scripts/05_signature_reduction.py --config "$CONFIG" --fast
run "Gate 6: immune deconvolution"  scripts/06_immune_deconv.py       --config "$CONFIG"
run "Gate 7: METABRIC validation"   scripts/07_external_metabric.py   --config "$CONFIG"
run "Figures"                       scripts/08_figures.py             --config "$CONFIG"
run "PDF report"                    scripts/09_report.py              --config "$CONFIG"

ELAPSED=$(( $(date +%s) - START ))
echo ""
echo "=============================================================="
echo " SMOKE TEST PASSED in ${ELAPSED}s"
echo " tables:  $(ls results_smoke/tables 2>/dev/null | wc -l) files"
echo " figures: $(ls results_smoke/figures 2>/dev/null | wc -l) files"
echo "=============================================================="
[ "$ELAPSED" -lt 300 ] || echo "WARNING: exceeded the 5-minute target (${ELAPSED}s)"
