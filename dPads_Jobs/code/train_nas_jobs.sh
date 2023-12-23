train_jobs_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 -W ignore train_nas.py \
    --algorithm nas \
    --exp_name jobs \
    --trial 1 \
    --train_data data/fly_process/train_data.npy \
    --test_data data/fly_process/test_data.npy \
    --train_labels data/fly_process/train_label.npy \
    --test_labels data/fly_process/test_label.npy \
    --input_type "atom" \
    --output_type "atom" \
    --input_size 18 \
    --output_size 1 \
    --num_labels 0 \
    --lossfxn "mseloss" \
    --max_depth 3 \
    --learning_rate 0.00001 \
    --search_learning_rate 0.01 \
    --train_valid_split 0.8 \
    --symbolic_epochs 200 \
    --neural_epochs 200 \
    --batch_size 16 \
    --normalize \
    --random_seed 0 \
    --finetune_epoch 10 \
    --finetune_lr 0.00025
}


train_jobs_specific 0