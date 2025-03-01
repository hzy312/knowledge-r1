export HF_TOKEN="xxx"
export WANDB_PROJECT="kr1_Qwen2.5-7B-Instruct"
export WANDB_API_KEY="xxx"


# for 8 x 40G node, using bsz=16 & zero2 & zero3 will oom

# Detect available port and set it to PORT
PORT=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d: -f2 | sort -u) | shuf -n 1)

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    --main_process_port $PORT --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_kr1.yaml