if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

export CUDA_VISIBLE_DEVICES=1

model_name=SparseTSF

root_path_name=./dataset/
data_path_name=invoice_data_1hour_quantities_pivot.csv
model_id_name=InvoiceQuantitiesHourly_2
data_name=custom

seq_len=2016
for pred_len in  288 2016  8640
do
  python -u run_longExp.py \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --period_len 24 \
    --model_type 'mlp' \
    --d_model 128 \
    --enc_in 317 \
    --train_epochs 30 \
    --patience 5 \
    --itr 1 --batch_size 128 --learning_rate 0.02 > logs/invoice_quantity_hourly_${pred_len}.log 2>&1
done



CUDA_VISIBLE_DEVICES=1 python -u run_longExp.py \
  --is_training 0 \
  --root_path ./dataset/ \
  --data_path invoice_data_1hour_quantities_pivot.csv \
  --model_id InvoiceQuantitiesHourly_2_2016_2016 \
  --model SparseTSF \
  --data custom \
  --features M \
  --seq_len 2016 \
  --pred_len 2016 \
  --period_len 24 \
  --model_type 'mlp' \
  --d_model 128 \
  --enc_in 317 \
  --train_epochs 30 \
  --patience 5 \
  --do_predict \
  --itr 1 --batch_size 128 --learning_rate 0.02 > logs/invoice_quantity_hourly_2016.log 2>&1