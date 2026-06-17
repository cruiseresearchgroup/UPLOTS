
CUDA_VISIBLE_DEVICES=2 python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path electricity.csv \
    --model_id Electricity_336_720 \
    --model SparseTSF \
    --data custom \
    --features M \
    --seq_len 336 \
    --pred_len 720 \
    --period_len 24 \
    --model_type 'mlp' \
    --d_model 128 \
    --enc_in 321 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 128 --learning_rate 0.02