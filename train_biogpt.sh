train_data_path=""
output_dir=""
model_name_or_path="microsoft/biogpt"
CUDA_VISIBLE_DEVICES=0,2 accelerate  launch --main_process_port 10001  train_pco.py \
    --model_name_or_path $model_name_or_path \
    --data_path $train_data_path \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 50 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 5 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --fsdp_transformer_layer_cls_to_wrap "BioGptDecoderLayer" \
    --tf32 True \
    --model_max_length 1024 \
    --fsdp "full_shard auto_wrap" \
