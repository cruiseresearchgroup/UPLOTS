#!/bin/bash
# =============================================================================
# UPLOTS — Stage 1: Train the unified backbone + sample all constraints
# Reproduces the main experiment (ETTh + Energy, 14 constraint configs).
# Run from anywhere:  bash scripts/01_train_sample.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../uplots"

EXP_NAME="mix14_etth_energy"
MILE=1000          # training epochs / checkpoint milestone
GPU=0

CONFIGS=(
  morning_peak_etth   evening_peak_etth   morning_peak_energy evening_peak_energy
  workday_etth        weekend_etth        workday_energy      weekend_energy
  high_load_etth      low_load_etth       high_load_energy    low_load_energy
  volatile_etth       volatile_energy
)

# ---- Train one unified model on ALL constraints at once -----------------------
python -u main.py --gpu ${GPU} --name ${EXP_NAME} \
  --config_file "${CONFIGS[@]}" \
  --sample 0 --train --epoch ${MILE} --batch 32

# ---- Sample / inference per constraint prompt --------------------------------
for CFG in "${CONFIGS[@]}"; do
  python -u main.py --gpu ${GPU} --name ${EXP_NAME} \
    --config_file ${CFG} --sample 0 --milestone ${MILE}
done

echo "Done. Generated samples are in uplots/OUTPUT/${EXP_NAME}/"
