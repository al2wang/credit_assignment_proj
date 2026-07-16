import subprocess
import os

if __name__ == '__main__':

    # Curriculum phases:
    # 1: Subsequence RL (Isolated Lemmas)
    # 2: Compositional RL (A + B)
    # 3: Full Theorem (Hierarchical Advantage)
    curriculum_phases = [1, 2, 3]
    
    alpha = 1.0
    budget = 2.5 

    base_dir = "/network/scratch/g/guangyuan.wang/credit_assignment/OPSD"  
    job_script_directory = os.path.join(base_dir, "temp_job_scripts_curriculum")
    out_dir = os.path.join(base_dir, "grpo_cppo", "logs_curriculum")
    os.makedirs(job_script_directory, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    submitted_jobs = 0
    base_port = 89346

    for i, phase in enumerate(curriculum_phases):
        
        run_name = f"exp_curriculum_qwen4b_phase{phase}"
        current_port = base_port + i
        
        # Gradually expand the generation horizon as the curriculum advances
        max_len = 512 if phase == 1 else (1536 if phase == 2 else 4096)
        epochs = 1 if phase < 3 else 2

        bash_run_command = (
            f"accelerate launch "
            f"--config_file accelerate.yaml "
            f"--num_processes 8 "
            f"--main_process_port {current_port} "
            f"--gradient_accumulation_steps 1 "  
            f"grpo_cppo_train_curriculum.py " 
            f"--learning_rate 5e-6 "
            f"--per_device_train_batch_size 1 "
            f"--gradient_accumulation_steps 1 "
            f"--model_name_or_path ./models/Qwen3-4B " 
            f"--output_dir {base_dir}/grpo_cppo/results_curriculum "
            f"--run_config {run_name} "
            f"--num_train_epochs {epochs} "
            f"--gradient_checkpointing "
            f"--lora_r 64 "
            f"--lora_alpha 128 "
            f"--lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj "
            f"--max_completion_length {max_len} " 
            f"--num_generations 8 "
            f"--temperature 1.0 " 
            f"--use_vllm "
            f"--use_peft "
            f"--vllm_mode colocate "
            f"--vllm_max_model_len 8192 " 
            f"--logging_steps 5 " 
            f"--save_steps 50 "
            f"--beta 0.0 "
            f"--loss_type grpo "
            f"--scale_rewards group "
            f"--wandb_project GRPO_CPPO_Curriculum_Generation "
            f"--cppo_alpha {alpha} "
            f"--cppo_budget {budget} "
            f"--curriculum_phase {phase} " # NEW ARGUMENT
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

    print(f"Total Curriculum sweeps submitted to SLURM: {submitted_jobs}")