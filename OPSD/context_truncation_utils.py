import torch

def create_truncated_causal_mask(input_ids: torch.Tensor, prompt_lens: torch.Tensor, mode: str, k: int) -> torch.Tensor:
    """
    Constructs a 4D attention mask to enforce Markovian, Sliding Window, or Attention Sink assumptions.
    
    Args:
        input_ids: The concatenated prompt + completion tensor. Shape: (batch, seq_len)
        prompt_lens: 1D tensor containing the exact prompt length for each item in the batch.
        mode: "full", "markovian", "sliding", or "sinks".
        k: The context window horizon limit.
        
    Returns:
        mask_4d: Boolean attention mask. Shape: (batch, 1, seq_len, seq_len)
                 True indicates the position is masked (attention = -inf).
                 False indicates the position is attended to.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    if mode == "full":
        return None # Return None to let the model use its default causal mask
    
    # 1. Base causal mask (lower triangular) - True means masked
    base_causal_mask = ~torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    
    # Expand to 4D for the SDPA backend
    mask_4d = base_causal_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len).clone()
    
    for b in range(batch_size):
        p_len = prompt_lens[b].item()
        
        for i in range(p_len, seq_len):
            if mode == "sliding":
                # Mask out everything before (i - k + 1)
                start_idx = max(0, i - k + 1)
                mask_4d[b, 0, i, :start_idx] = True
                
            elif mode == "sinks":
                # Attend to prompt (0 to p_len) AND local window (i - k + 1 to i).
                # This means masking the gap between the prompt and the local window.
                local_start = max(p_len, i - k + 1)
                if local_start > p_len:
                    mask_4d[b, 0, i, p_len:local_start] = True
                    
            elif mode == "markovian":
                # Mask out all intermediate reasoning steps. 
                # Attend ONLY to the prompt and the immediate current token (i).
                mask_4d[b, 0, i, p_len:i] = True
                
    return mask_4d