export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_TOKEN=""

accelerate launch --config_file=deepspeed_zero3.yaml --num_processes 8 main_train_sft.py \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name hzy/Qwen2.5-3B-Instruct-nq-knowledge-probe-balanced-kb-ft-new \
    --learning_rate 1.0e-6 \
    --num_train_epochs 2 \
    --packing \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --logging_steps 5 \
    --eval_strategy no \
    --save_strategy epoch \
    --output_dir qwen2.5-3b-it-fb-sft \
    --max_length 650 \
    --bf16 \
    --push_to_hub