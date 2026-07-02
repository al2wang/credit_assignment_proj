import subprocess
import os
import itertools

if __name__ == '__main__':

    # 1. Define the grid of CPPO hyperparameters to sweep
    alphas = [0.0, 1.0, 2.0]
    budgets = [1.5, 2.5, 5.0, 9999.0]
    
    # Note: alpha=0.0 and budget=9999.0 effectively recovers standard GRPO 
    # as a baseline within this sweep.

    # Update this path to where your OPSD repository lives
    base_dir = "/network/scratch/g/guangyuan.wang/credit_assignment/OPSD"  
    job_script_directory = os.path.join(base_dir, "temp_job_scripts")
    out_dir = "/network/scratch/g/guangyuan.wang/credit_assignment/OPSD/grpo_cppo/logs"
    os.makedirs(job_script_directory, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    submitted_jobs = 0
    base_port = 19346

    # Iterate through all combinations of alpha and budget
    for i, (alpha, budget) in enumerate(itertools.product(alphas, budgets)):
        
        # Create a clean, unique run name for wandb and output directories
        run_name = f"qwen4b_cppo_a{alpha}_b{budget}".replace(".", "p")
        
        # Increment port to avoid collisions if multiple jobs land on the same node
        current_port = base_port + i

        # 2. The Accelerate launch command adapted from your run_grpo.sh
        # We drop the manual CUDA_VISIBLE_DEVICES as SLURM will handle GPU masking
        bash_run_command = (
            f"accelerate launch "
            f"--config_file accelerate.yaml "
            f"--num_processes 8 "
            f"--main_process_port {current_port} "
            f"grpo_cppo_train.py "
            f"--learning_rate 5e-6 "
            f"--per_device_train_batch_size 1 "
            f"--gradient_accumulation_steps 4 "
            f"--model_name_or_path ./models/Qwen3.5-4B "
            f"--output_dir /network/scratch/g/guangyuan.wang/credit_assignment/OPSD/grpo_cppo/results "
            f"--run_config {run_name} "
            f"--num_train_epochs 2 "
            f"--num_iterations 2 "
            f"--gradient_checkpointing "
            f"--lora_r 64 "
            f"--lora_alpha 128 "
            f"--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "
            f"--max_prompt_length 2048 "
            f"--max_completion_length 16000 "
            f"--num_generations 8 "
            f"--temperature 1.2 "
            f"--use_vllm "
            f"--use_peft "
            f"--vllm_mode colocate "
            f"--logging_steps 10 "
            f"--save_steps 20 "
            f"--beta 0.0 "
            f"--loss_type grpo "
            f"--scale_rewards group "
            f"--wandb_project OPSD "
            # -- CPPO Specific Arguments --
            f"--cppo_alpha {alpha} "
            f"--cppo_budget {budget}"
            f"--report_to wandb "
        )

        job_script_content = f'''#!/usr/bin/bash
echo "Start time: $(date +%Y:%m:%d-%H:%M:%S)"
module unload python
module load anaconda

cd {base_dir}
# Update to your actual conda environment path
conda activate your_opsd_conda_env

echo "Running: {bash_run_command}"
{bash_run_command}

echo "Stop time: $(date +%Y:%m:%d-%H:%M:%S)"
'''
        job_script_filename = os.path.join(job_script_directory, f"{run_name}.sh")
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
            
        # 3. SLURM Launch Command
        # Requesting 8 GPUs natively via SLURM since accelerate requires 8 processes
        launch_command = (
            f"sbatch --job-name={run_name} "
            f"--time=12:00:00 "
            f"--nodes=1 "
            f"--gres=gpu:8 "           # IMPORTANT: Adjusted to 8 GPUs
            f"-c 32 "                  # Adjust CPU cores depending on your cluster
            f"--mem=256G "             # Adjust RAM based on vLLM/accelerate needs
            f"--output={out_dir}/{run_name}.out "
            f"{job_script_filename}"
        )
        
        subprocess.run(launch_command, shell=True, executable='/usr/bin/bash')
        submitted_jobs += 1

    print(f"Total CPPO sweeps submitted to SLURM: {submitted_jobs}")