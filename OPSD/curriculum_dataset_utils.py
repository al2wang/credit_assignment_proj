import re

def parse_curriculum_phase(example, phase: int):
    """
    Transforms the standard dataset into a factorized curriculum target.
    In production, this relies on a pre-processed dataset with explicit lemma slicing.
    """
    original_question = example['Question']
    original_answer = example['Answer']
    
    if phase == 1:
        # Phase 1: Isolated Lemma
        # The prompt is constrained strictly to solving the first logical dependency.
        prompt = f"Problem: {original_question}\nSub-task: Identify the sample space or initial algebraic setup only. Do not solve the full problem. Place your conclusion in \\boxed{{}}."
        target_answer = original_answer # Placeholder: Requires actual sliced dataset ground truth
        
    elif phase == 2:
        # Phase 2: Compositional (Lemma A + Lemma B)
        # The prompt requires chaining the initial setup with the secondary calculation.
        prompt = f"Problem: {original_question}\nSub-task: Identify the initial setup, then calculate the intermediate combinatorial or polynomial step. Place the intermediate value in \\boxed{{}}."
        target_answer = original_answer # Placeholder
        
    else:
        # Phase 3: Full Theorem
        prompt = f"Problem: {original_question}\nPlease reason step by step, compose all lemmas, and put your final answer within \\boxed{{}}."
        target_answer = original_answer
        
    return {"prompt_text": prompt, "target_answer": target_answer}

def curriculum_reward_router(completions, targets, phase, base_reward_func):
    """
    Routes the reward calculation. 
    Phase 1 & 2 rely heavily on dense structural matching for the sub-tasks.
    Phase 3 integrates the full outcome reward.
    """
    if phase == 1:
        # Strict local advantage for the isolated chunk
        return base_reward_func(completions, targets, strict_lemma_matching=True)
    elif phase == 2:
        # Evaluates the transition and intermediate composition
        return base_reward_func(completions, targets, require_composition=True)
    else:
        # Full sequence evaluation
        return base_reward_func(completions, targets)