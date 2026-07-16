import os
import wandb
import re
import torch
from weight_learning_modules import PositionWeightLearner

# Assuming you have the PRM Oracle from the previous implementations
from prm_mcts_oracle import OptimisticPRMOracle 

from math_verify import parse, verify
from datasets import load_dataset
from transformers import AutoTokenizer

from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field

os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

@dataclass
class CustomScriptArguments(ScriptArguments):
    run_config: str = field(default=None)
    wandb_entity: str = field(default=None)
    wandb_project: str = field(default="grpo-training")

@dataclass
class CPPOConfig(GRPOConfig):
    cppo_budget: float = field(default=2.5)
    cliprange: float = field(default=0.1) # Corresponds to base c_t
    weight_mode: str = field(
        default="heuristic",
        metadata={"help": "Mode: 'heuristic', 'meta_learning', 'option_critic', 'epistemic_prm'"}
    )

class CPPOGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cppo_metrics = {
            "dynamic_epsilon": [],
            "budget_exceeded": [],
            "token_kl": [],
            "surrogate_loss": [],
            "response_length_k": [], 
            "rel_log_prob_error": [],
            "mean_w_t": [] # NEW: Log the position weights
        }
        
        # Initialize the weight learner module
        # Note: In a full pipeline, you would extract the hidden_size dynamically from model.config
        self.weight_learner = PositionWeightLearner(hidden_size=1024).to(self.accelerator.device)
        self.prm_oracle = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt_ids = inputs.get("prompt_input_ids", inputs.get("prompt_ids"))
        completion_ids = inputs.get("completion_input_ids", inputs.get("completion_ids"))
        
        prompt_mask = inputs.get("prompt_attention_mask", torch.ones_like(prompt_ids))
        comp_mask = inputs.get("completion_attention_mask", torch.ones_like(completion_ids))
        
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, comp_mask], dim=1)
        
        advantages = inputs["advantages"]
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1) 

        # We need output_hidden_states=True for the Option-Critic mode
        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        logits = outputs.logits[:, :-1, :]
        
        with torch.inference_mode():
            if self.ref_model is None:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_outputs = model(input_ids, attention_mask=attention_mask)
            else:
                ref_outputs = self.ref_model(input_ids, attention_mask=attention_mask)
            
            ref_logits = ref_outputs.logits[:, :-1, :]

        labels = input_ids[:, 1:]
        
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        ref_per_token_logps = torch.gather(ref_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        prompt_len = prompt_ids.shape[1]
        per_token_logps = per_token_logps[:, prompt_len - 1:]
        ref_per_token_logps = ref_per_token_logps[:, prompt_len - 1:]
        
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        ratio = torch.exp(per_token_logps - per_token_logps.detach())

        seq_len = ratio.size(1)
        positions = torch.arange(seq_len, device=ratio.device, dtype=ratio.dtype)
        
        # =====================================================================
        # EXPERIMENT: Calculate Dynamic Position Weights (w_t)
        # =====================================================================
        if self.args.weight_mode == "heuristic":
            w_t = self.weight_learner.compute_heuristic_weights(positions, seq_len)
            w_t = w_t.unsqueeze(0).expand_as(ratio)
            
        elif self.args.weight_mode == "meta_learning":
            w_t = self.weight_learner.compute_meta_learning_weights(positions)
            w_t = w_t.unsqueeze(0).expand_as(ratio)
            
        elif self.args.weight_mode == "option_critic":
            # Extract hidden states corresponding to the generated tokens
            hidden_states = outputs.hidden_states[-1][:, prompt_len - 1 : -1, :]
            w_t = self.weight_learner.compute_option_critic_weights(hidden_states)
            
        elif self.args.weight_mode == "epistemic_prm":
            if self.prm_oracle is None:
                self.prm_oracle = OptimisticPRMOracle().to(input_ids.device)
            c_t = self.prm_oracle.simulate_or_fetch_prm(input_ids, attention_mask)[:, prompt_len - 1:]
            w_t = self.weight_learner.compute_epistemic_prm_weights(c_t)

        # CPPO Bound Mathematics: l_t = c_t / w_t
        # This converts the position weight into the effective divergence threshold (epsilon_t)
        base_c_t = torch.full_like(ratio, self.args.cliprange)
        epsilon_t = base_c_t / w_t
        # =====================================================================

        cum_kl = torch.cumsum(per_token_kl, dim=1)
        budget_mask = (cum_kl <= self.args.cppo_budget).float()
        
        dynamic_epsilon = epsilon_t * budget_mask
        clip_frac = torch.clamp(ratio, 1.0 - dynamic_epsilon, 1.0 + dynamic_epsilon)

        surrogate_loss = -torch.min(ratio * advantages, clip_frac * advantages)
        loss = surrogate_loss + self.args.beta * per_token_kl
        
        loss = (loss * comp_mask).sum() / comp_mask.sum()

        with torch.no_grad():
            budget_exceeded_pct = 1.0 - budget_mask.mean().item()
            mean_length_k = comp_mask.sum(dim=1).float().mean().item() / 1000.0
            
            self._cppo_metrics["dynamic_epsilon"].append(dynamic_epsilon.mean().item())
            self._cppo_metrics["budget_exceeded"].append(budget_exceeded_pct)
            self._cppo_metrics["token_kl"].append(per_token_kl.mean().item())
            self._cppo_metrics["surrogate_loss"].append(surrogate_loss.mean().item())
            self._cppo_metrics["response_length_k"].append(mean_length_k)
            self._cppo_metrics["rel_log_prob_error"].append(per_token_kl.abs().mean().item())
            self._cppo_metrics["mean_w_t"].append(w_t.mean().item())

        if return_outputs:
            return loss, outputs
        return loss

    def log(self, logs: dict, *args, **kwargs):
        if len(self._cppo_metrics["dynamic_epsilon"]) > 0:
            for key in self._cppo_metrics:
                logs[f"cppo/{key}"] = sum(self._cppo_metrics[key]) / len(self._cppo_metrics[key])
                self._cppo_metrics[key] = [] 
        super().log(logs, *args, **kwargs)

# --- Standard Evaluation Hooks Below ---

def extract_boxed_answer(text):
    think_end = text.rfind("</think>")
    search_text = text[think_end + len("</think>") :] if think_end != -1 else text
    idx = search_text.find(r"\boxed{")
    if idx == -1:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(search_text) and depth > 0:
        if search_text[i] == "{":
            depth += 1
        elif search_text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return search_text[start : i - 1].strip()
    return None

def _preprocess_for_parse(answer):
    if answer is None:
        return None
    ratio_match = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*:\s*(-?\d+(?:\.\d+)?)\s*", answer)
    if ratio_match:
        return rf"\frac{{{ratio_match.group(1)}}}{{{ratio_match.group(2)}}}"
    return answer

def reward_correctness(completions, Answer, **kwargs):
    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, Answer)):
        pred_answer = extract_boxed_answer(completion)
        reward = 0.0
        gold_parsed = parse(ground_truth)
        pred_parsed = parse(_preprocess_for_parse(pred_answer))
        if gold_parsed is not None and pred_parsed is not None:
            try:
                reward = 1.0 if verify(gold_parsed, pred_parsed) else 0.0
            except Exception:
                pass
        if reward == 0.0:
            pred_norm = re.sub(r"\s+", "", pred_answer or "").lower()
            gt_norm = re.sub(r"\s+", "", ground_truth or "").lower()
            if pred_norm and pred_norm == gt_norm:
                reward = 1.0
        rewards.append(reward)
    return rewards

def make_format_prompt(tokenizer):
    def format_prompt(example):
        messages = [
            {
                "role": "user",
                "content": f"Problem: {example['Question']}\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {"prompt": prompt, "Answer": example["Answer"]}
    return format_prompt

if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, CPPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]
        full_wandb_run_name = (
            f"CPPO_{training_args.weight_mode}_{model_name}_"
            f"lr{lr_str}_bs{effective_batch_size}"
        )

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: POSITION WEIGHT LEARNING ($w_t$)")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Weight Mode: {training_args.weight_mode}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "effective_batch_size": effective_batch_size,
                "weight_mode": training_args.weight_mode,
            },
        )

    model_dtype = torch.bfloat16 
    
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        attn_implementation=model_args.attn_implementation or "sdpa", 
        torch_dtype=model_dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=getattr(model_args, "trust_remote_code", True),
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("/network/scratch/g/guangyuan.wang/credit_assignment/OPSD/data/Openthoughts_math_30k")
    train_dataset = dataset["train"]

    format_prompt = make_format_prompt(tokenizer)
    train_dataset = train_dataset.map(format_prompt, remove_columns=train_dataset.column_names)
    split_dataset = train_dataset.train_test_split(test_size=0.007, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    train_dataset = train_dataset.select(range(500))

    trainer = CPPOGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        checkpoints = sorted(
            [d for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]),
        )
        if checkpoints:
            resume_from_checkpoint = os.path.join(training_args.output_dir, checkpoints[-1])

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)