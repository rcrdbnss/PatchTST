if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=PatchTST

root_path_name=./dataset/
data_path_name=exchange_rate.csv
model_id_name=Exchange
data_name=custom

random_seed=2021
for pred_len in 96
do
    python run_longExp.py \
        --random_seed $random_seed \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name_$seq_len_$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --enc_in 8 \
        --e_layers 3 \
        --n_heads 4 \
        --d_model 16 \
        --d_ff 128 \
        --dropout 0.3\
        --fc_dropout 0.3\
        --head_dropout 0\
        --patch_len 16\
        --stride 8\
        --des 'Exp' \
        --train_epochs 3\
        --itr 1 --batch_size 32 --learning_rate 0.0005 #>logs/LongForecasting/$model_name'_'Exchange_$seq_len'_'96.log
done