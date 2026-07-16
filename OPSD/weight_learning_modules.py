import torch
import torch.nn as nn
import numpy as np

class PositionWeightLearner(nn.Module):
    """
    Central hub for calculating the position weight w_t used in the CPPO bound constraint.
    """
    def __init__(self, hidden_size=1024, max_seq_len=4096, base_weight=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.base_weight = base_weight
        
        # Meta-Learning Mode: Learnable array of weights tied directly to absolute token positions
        self.meta_weight_params = nn.Parameter(torch.ones(max_seq_len))
        
        # Option-Critic Mode: A lightweight head to predict option termination probability beta(s)
        self.option_termination_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def compute_heuristic_weights(self, positions: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        w_t = base + alpha * (t / T)
        The standard monotonically increasing penalty.
        """
        # Assume alpha = 0.8 for the heuristic scaling from 0.2 to 1.0
        return self.base_weight + 0.8 * (positions / seq_len)

    def compute_meta_learning_weights(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Outer-loop optimization of w_t per position.
        Uses a sigmoid to strictly bound the learned weights between [base_weight, 1.0].
        """
        raw_weights = self.meta_weight_params[positions.long()]
        scaled_weights = self.base_weight + (1.0 - self.base_weight) * torch.sigmoid(raw_weights)
        return scaled_weights

    def compute_option_critic_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Option-Critic framework.
        Calculates termination probability beta(s).
        w_t is held high during intra-option execution and drops sharply at option boundaries.
        w_t = 1.0 - (0.8 * beta(s))
        """
        # hidden_states shape: (batch, seq_len, hidden_size)
        termination_probs = self.option_termination_head(hidden_states).squeeze(-1)
        
        # If beta(s) is near 1 (termination), w_t drops to 0.2 (loose bounds for exploration)
        # If beta(s) is near 0 (intra-option), w_t stays near 1.0 (strict bounds)
        w_t = 1.0 - (0.8 * termination_probs)
        return torch.clamp(w_t, min=self.base_weight, max=1.0)

    def compute_epistemic_prm_weights(self, prm_confidences: torch.Tensor) -> torch.Tensor:
        """
        w_t is directly proportional to PRM confidence.
        If PRM is confident (C_t ~ 1), w_t is high (tight bound).
        If PRM is uncertain (C_t ~ 0), w_t drops (loose bound for exploration).
        """
        # Assume prm_confidences are normalized between [0, 1]
        w_t = self.base_weight + (1.0 - self.base_weight) * prm_confidences
        return torch.clamp(w_t, min=self.base_weight, max=1.0)