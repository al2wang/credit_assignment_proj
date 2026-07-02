#!/bin/bash

module load cudatoolkit/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
echo "CUDA_HOME is set to: $CUDA_HOME"

# Force DeepSpeed to ignore the PyTorch 12.8 vs CUDA 11.8 mismatch
export DS_SKIP_CUDA_CHECK=1

CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file accelerate.yaml \
    --num_processes 1 \
    --gradient_accumulation_steps 8 \
    --main_process_port 19346 \
    grpo_cppo_train.py \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --generation_batch_size 8 \
    --model_name_or_path ./models/Qwen3-4B \
    --output_dir /network/scratch/g/guangyuan.wang/credit_assignment/OPSD/grpo_cppo/ \
    --run_config qwen3_cppo_test_run \
    --num_train_epochs 2 \
    --num_iterations 2 \
    --gradient_checkpointing \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --max_completion_length 1024 \
    --num_generations 8 \
    --temperature 1.2 \
    --use_vllm \
    --use_peft \
    --vllm_mode colocate \
    --vllm_max_model_len 4096 \
    --logging_steps 10 \
    --save_steps 20 \
    --beta 0.0 \
    --loss_type grpo \
    --scale_rewards group \
    --wandb_project GRPO_CPPO \
    --report_to wandb \
    --cppo_alpha 1.0 \
    --cppo_budget 2.5