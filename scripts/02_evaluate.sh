#!/bin/bash
# =============================================================================
# UPLOTS — Stage 2: Evaluate generation quality (C-FID / Discriminative / Predictive)
# Compares generated samples against the normalized ground-truth for all 14 configs.
# Run after 01_train_sample.sh:  bash scripts/02_evaluate.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../uplots"

EXP_NAME="mix14_etth_energy"
MILE=1000
SEQ=24
MASK=0.0
OUTDIR="OUTPUT"

CONFIGS=(
  morning_peak_etth   evening_peak_etth   morning_peak_energy evening_peak_energy
  workday_etth        weekend_etth        workday_energy      weekend_energy
  high_load_etth      low_load_etth       high_load_energy    low_load_energy
  volatile_etth       volatile_energy
)

# workday/weekend truth files are saved under different names
declare -A TRUTHNAME
TRUTHNAME[workday_etth]=workdays_etth
TRUTHNAME[weekend_etth]=non_workdays_etth
TRUTHNAME[workday_energy]=workdays_energy
TRUTHNAME[weekend_energy]=non_workdays_energy

echo "Config | C-FID | Discri | Predic"
echo "-------|-------|--------|-------"
for CFG in "${CONFIGS[@]}"; do
  TNAME="${TRUTHNAME[$CFG]:-$CFG}"
  ORI="${OUTDIR}/${EXP_NAME}/samples/${TNAME}_norm_truth_${SEQ}_train.npy"
  FAKE="${OUTDIR}/${EXP_NAME}/ddpm_fake_${CFG}_milestone_${MILE}_mask${MASK}_len${SEQ}.npy"

  python -u evaluate.py --ori "${ORI}" --fake "${FAKE}" \
    --tag "${EXP_NAME}_${CFG}" --out "eval_${CFG}.json" > /dev/null 2>&1

  python -c "
import json
d=json.load(open('eval_${CFG}.json'))
g=lambda k: f\"{d[k]:.4f}\" if isinstance(d.get(k),float) else 'N/A'
print(f'${CFG} | {g(\"cfid_mean\")} | {g(\"discri_mean\")} | {g(\"predic_mean\")}')"
done
