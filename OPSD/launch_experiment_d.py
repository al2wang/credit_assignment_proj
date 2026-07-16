import subprocess
import os

if __name__ == '__main__':

    # The three modes defining how the CPPO dynamic trust region is calculated
    uncertainty_modes = ["time_heuristic", "uct_guided", "epinet_guided"]
    
    alpha = 1.0
    budget = 2.5 

    base_dir = "/network/scratch/g/guangyuan.wang/credit_assignment/OPSD"  
    job_script_directory = os.path.join(base_dir, "temp_job_scripts_exp_d")
    out_dir = os.path.join(base_dir, "grpo_cppo", "logs_exp_d")
    os.makedirs(job_script_directory, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    submitted_jobs = 0
    base_port = 49346

    for i, mode in enumerate(uncertainty_modes):
        
        run_name = f"exp_d_qwen4b_{mode}".replace(".", "p")
        current_port = base_port + i

        bash_run_command = (
            f"accelerate launch "
            f"--config_file accelerate.yaml "
            f"--num_processes 8 "
            f"--main_process_port {current_port} "
            f"--gradient_accumulation_steps 1 "  
            f"grpo_cppo_train_exp_d.py " 
            f"--learning_rate 5e-6 "
            f"--per_device_train_batch_size 1 "
            f"--gradient_accumulation_steps 1 "
            f"--model_name_or_path ./models/Qwen3-4B " 
            f"--output_dir {base_dir}/grpo_cppo/results_exp_d "
            f"--run_config {run_name} "
            f"--num_train_epochs 2 "
            f"--gradient_checkpointing "
            f"--lora_r 64 "
            f"--lora_alpha 128 "
            f"--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "
            f"--max_completion_length 1024 " 
            f"--num_generations 8 "
            f"--temperature 1.0 " 
            f"--use_vllm "
            f"--use_peft "
            f"--vllm_mode colocate "
            f"--vllm_max_model_len 4096 " 
            f"--logging_steps 5 " 
            f"--save_steps 50 "
            f"--beta 0.0 "
            f"--loss_type grpo "
            f"--scale_rewards group "
            f"--wandb_project GRPO_CPPO_Epistemic_Ablation "
            f"--cppo_alpha {alpha} "
            f"--cppo_budget {budget} "
            f"--uncertainty_mode {mode} " 
            f"--report_to wandb"
        )

        job_script_content = f'''#!/usr/bin/bash
echo "Start time: $(date +%Y:%m:%d-%H:%M:%S)"

module load cudatoolkit/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export DS_SKIP_CUDA_CHECK=1

eval "$(conda shell.bash hook)"
conda activate /network/scratch/g/guangyuan.wang/credit_assignment/condaenv

cd {base_dir}

echo "Running: {bash_run_command}"
{bash_run_command}

echo "Stop time: $(date +%Y:%m:%d-%H:%M:%S)"
'''
        job_script_filename = os.path.join(job_script_directory, f"{run_name}.sh")
        
        with open(job_script_filename, 'w') as job_script_file:
            job_script_file.write(job_script_content)
            
        launch_command = (
            f"sbatch --job-name={run_name} "
            f"--time=04:00:00 " 
            f"--nodes=1 "
            f"--gres=gpu:rtx8000:8 "           
            f"-c 32 "                  
            f"--mem=256G "             
            f"--output={out_dir}/{run_name}.out "
            f"{job_script_filename}"
        )
        
        subprocess.run(launch_command, shell=True, executable='/usr/bin/bash')
        submitted_jobs += 1

    print(f"Total Experiment D epistemic sweeps submitted to SLURM: {submitted_jobs}")