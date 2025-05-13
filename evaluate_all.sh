# datasets=("nq_easy" "nq_hard" "popqa_easy" "popqa_hard" "hotpotqa_easy" "hotpotqa_hard" "2wikimultihopqa_easy" "2wikimultihopqa_hard")

models=(
   2025-04-28-11:23:40-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-ablation-2
)
datasets=("hotpotqa_easy")
    # "2025-04-23-14:22:38-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-7b-it-em-encourage-ret"
    # "2025-04-24-02:32:27-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-7b-em-encourage-ret"
    # "2025-04-26-15:55:59-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-encourage-ret"
    # "2025-04-27-07:37:05-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-em-encourage-ret"
    # "2025-04-28-02:47:19-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-ablation-1"
    # "2025-04-28-11:23:40-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-ablation-2"
    # "2025-04-29-01:35:16-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-ablation-easy"
    # "2025-04-29-08:01:41-nq_hotpotqa-kb-search-r1-grpo-qwen2.5-3b-it-em-ablation-hard"


for model in "${models[@]}"; do
    DATASET_PREFIX="data/ikea"
    LOG_PREFIX="logs/${model}"
    # if [[ $model == ikea* ]]; then
    #     DATASET_PREFIX="formal_ikea_dataset"
    #     LOG_PREFIX="formal_ikea_${model}_log"
    # elif [[ $model == R1* ]]; then
    #     DATASET_PREFIX="formal_r1_dataset"
    #     LOG_PREFIX="formal_r1_${model}_log"
    # elif [[ $model == SearchR1* ]]; then
    #     DATASET_PREFIX="formal_search_r1_dataset"
    #     LOG_PREFIX="formal_search_r1_${model}_log"
    # else
    #     echo "Unknown model prefix for $model"
    #     continue
    # fi

    mkdir -p "$LOG_PREFIX"

    for data_name in "${datasets[@]}"; do
        export DATA_DIR=${DATASET_PREFIX}/${data_name}
        echo "Processing model: ${model}, dataset: ${data_name}"

        PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
            data.train_files=$DATA_DIR/test.parquet \
            data.val_files=$DATA_DIR/test.parquet \
            data.train_data_num=null \
            data.val_data_num=null \
            data.train_batch_size=512 \
            data.val_batch_size=512 \
            data.max_prompt_length=4096 \
            data.max_response_length=500 \
            data.max_start_length=2048 \
            data.max_obs_length=500 \
            data.shuffle_train_dataloader=True \
            algorithm.adv_estimator=gae \
            actor_rollout_ref.model.path=searchr1_verl_checkpoints/$model/actor/global_step_120 \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.enable_gradient_checkpointing=true \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
            actor_rollout_ref.actor.ppo_mini_batch_size=256 \
            actor_rollout_ref.actor.ppo_micro_batch_size=64 \
            actor_rollout_ref.actor.fsdp_config.param_offload=true \
            actor_rollout_ref.actor.fsdp_config.grad_offload=true \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
            actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.do_sample=False \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
            actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.rollout.n_agent=1 \
            actor_rollout_ref.rollout.temperature=0 \
            actor_rollout_ref.actor.state_masking=true \
            critic.optim.lr=1e-5 \
            critic.model.use_remove_padding=True \
            critic.optim.lr_warmup_steps_ratio=0.05 \
            critic.model.path=searchr1_verl_checkpoints/$model/actor/global_step_120 \
            critic.model.enable_gradient_checkpointing=true \
            critic.ppo_micro_batch_size=8 \
            critic.model.fsdp_config.param_offload=true \
            critic.model.fsdp_config.grad_offload=true \
            critic.model.fsdp_config.optimizer_offload=true \
            algorithm.kl_ctrl.kl_coef=0.001 \
            algorithm.no_think_rl=false \
            trainer.critic_warmup=0 \
            trainer.logger=[] \
            +trainer.val_only=true \
            +trainer.val_before_train=true \
            trainer.default_hdfs_dir=null \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1 \
            max_turns=3 \
            retriever.url="http://127.0.0.1:8000/retrieve" \
            retriever.topk=3 > ${LOG_PREFIX}/${data_name}_eval.log 2>&1
    done
done