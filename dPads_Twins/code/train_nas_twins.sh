train_twins_specific(){
    CUDA_VISIBLE_DEVICES=$1 python3 train_nas.py \
    --algorithm nas \
    --exp_name twins \
    --trial 1 \
    --train_data data/ \
    --test_data data/ \
    --train_labels data/ \
    --test_labels data/ \
    --input_type "atom" \
    --output_type "atom" \
    --input_size 22 \
    --output_size 1 \
    --num_labels 0 \
    --lossfxn "mseloss" \
    --max_depth 3 \
    --learning_rate 0.00001 \
    --search_learning_rate 0.00001 \
    --train_valid_split 0.8 \
    --symbolic_epochs 200 \
    --neural_epochs 200 \
    --batch_size 128 \
    --normalize \
    --random_seed 0 \
    --finetune_epoch 50 \
    --finetune_lr 0.00025
}


train_twins_specific 0