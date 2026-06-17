#!/bin/bash
# =============================================================================
# UPLOTS — Generalization: zero-shot + few-shot on held-out constraints
# Trains a base model on seen prompts, then evaluates unseen prompt combinations
# zero-shot and after K-shot adaptation (K = 100/300/500).
# Prereq (once): cd uplots && python prepare_fewshot.py   # builds few-shot YAML configs
# Run:  bash scripts/03_fewshot.sh
# =============================================================================
set -e
cd "$(dirname "$0")/../uplots"

BASE_NAME="base_fewshot"
MILE=1000
FS_EPOCH=50
GPU=0
SEQ=24; MASK=0.0; OUTDIR="OUTPUT"
CKPT="Checkpoints_${BASE_NAME}_${SEQ}_maskrate${MASK}/checkpoint-${MILE}.pt"

eval_uplots() {  # <exp> <config> <milestone> <tag>
  local ORI="${OUTDIR}/$1/samples/$2_norm_truth_${SEQ}_train.npy"
  local FAKE="${OUTDIR}/$1/ddpm_fake_$2_milestone_$3_mask${MASK}_len${SEQ}.npy"
  python -u evaluate.py --ori "${ORI}" --fake "${FAKE}" --tag "$4" --out "eval_$4.json"
}

# ---- Phase 1: base model (seen: ETTh, Energy, ETTh-MP, Energy-EP) ------------
python -u main.py --gpu ${GPU} --name ${BASE_NAME} \
  --config_file etth energy morning_peak_etth evening_peak_energy \
  --sample 0 --train --epoch ${MILE} --batch 32

# ---- Phase 2: zero-shot on unseen prompts (ETTh-EP, Energy-MP) ---------------
python -u main.py --gpu ${GPU} --name ${BASE_NAME} --config_file evening_peak_etth   --sample 0 --milestone ${MILE}
python -u main.py --gpu ${GPU} --name ${BASE_NAME} --config_file morning_peak_energy --sample 0 --milestone ${MILE}
eval_uplots ${BASE_NAME} evening_peak_etth   ${MILE} zeroshot_ettep
eval_uplots ${BASE_NAME} morning_peak_energy ${MILE} zeroshot_enemp

# ---- Phase 3+4: few-shot adaptation -----------------------------------------
for TARGET in "ettep:evening_peak_etth" "enemp:morning_peak_energy"; do
  TAG="${TARGET%%:*}"; CFG="${TARGET##*:}"
  for K in 100 300 500; do
    python -u main.py --gpu ${GPU} --name fewshot_${TAG}_k${K} \
      --config_file fewshot${K}_${CFG} --pretrain_path ${CKPT} \
      --sample 0 --train --epoch ${FS_EPOCH} --batch 32
    python -u main.py --gpu ${GPU} --name fewshot_${TAG}_k${K} \
      --config_file ${CFG} --sample 0 --milestone ${FS_EPOCH}
    eval_uplots fewshot_${TAG}_k${K} ${CFG} ${FS_EPOCH} fewshot_${TAG}_k${K}
  done
done

echo "Done. See eval_*.json"
