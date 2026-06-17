#!/bin/bash
# SparseTSF peak forecasting: orig 10%-90% + 10%+Gen
set -e

DATA_DIR="../etthmpep_energympep_pems04mpep_pems08mpep_dwcl"
SEQ_LEN=12; PRED_LEN=12; LABEL_LEN=4; PERIOD_LEN=4
TRAIN_EPOCHS=100; BATCH_SIZE=32; LR=0.001; GPU=5

mkdir -p logs/peak_forecast

run_exp() {
    local MODEL_ID=$1 DATA_PATH=$2 TRAIN_RATIO=$3
    local AUG_PATH=${4:-""}
    echo ">>> SparseTSF | $MODEL_ID | ratio=$TRAIN_RATIO"
    python run_longExp.py \
        --is_training 1 \
        --model_id "$MODEL_ID" --model SparseTSF \
        --data npy --root_path "$DATA_DIR" --data_path "$DATA_PATH" \
        --aug_data_path "$AUG_PATH" --train_ratio $TRAIN_RATIO \
        --features S --target OT \
        --seq_len $SEQ_LEN --label_len $LABEL_LEN --pred_len $PRED_LEN \
        --period_len $PERIOD_LEN --model_type linear \
        --enc_in 1 --dec_in 1 --c_out 1 --d_model 64 \
        --train_epochs $TRAIN_EPOCHS --batch_size $BATCH_SIZE \
        --learning_rate $LR --patience 10 --num_workers 0 --gpu $GPU \
        --itr 1 --des "peak_forecast" \
        2>&1 | tee "logs/peak_forecast/${MODEL_ID}.log"
    echo "<<< Done: $MODEL_ID"
}

DATASETS=(
    "etth_morning|morning_peak_etth_uni.npy|ddpm_fake_morning_peak_etth_milestone_1000_mask0.0_len24.npy"
    "etth_evening|evening_peak_etth_uni.npy|ddpm_fake_evening_peak_etth_milestone_1000_mask0.0_len24.npy"
    "energy_morning|morning_peak_energy_uni.npy|ddpm_fake_morning_peak_energy_milestone_1000_mask0.0_len24.npy"
    "energy_evening|evening_peak_energy_uni.npy|ddpm_fake_evening_peak_energy_milestone_1000_mask0.0_len24.npy"
)

for ds in "${DATASETS[@]}"; do
    IFS='|' read -r DS_NAME ORI_FILE AUG_FILE <<< "$ds"

    # orig 10%–90%
    for R in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        RPCT=$(echo "$R" | awk '{printf "%d", $1*100}')
        run_exp "SparseTSF_${DS_NAME}_r${RPCT}_orig" "$ORI_FILE" "$R"
    done

    # 10% + Gen
    run_exp "SparseTSF_${DS_NAME}_r10_aug" "$ORI_FILE" "0.1" "$AUG_FILE"
done

echo "=== All SparseTSF experiments done ==="
