import torch
import torch.nn as nn
import numpy as np

class OptimisticPRMOracle(nn.Module):
    """
    Evaluates reasoning chunks to extract step-level confidence using
    Upper Confidence Bounds (UCB) for optimistic exploration in the MCTS tree.
    """
    def __init__(self, exploration_constant=np.sqrt(2)):
        super().__init__()
        self.c_puct = exploration_constant

    def compute_ucb_confidence(self, q_values: torch.Tensor, parent_visits: torch.Tensor, node_visits: torch.Tensor) -> torch.Tensor:
        """
        Calculates optimistic exploration values.
        """
        safe_node_visits = torch.clamp(node_visits, min=1e-8)
        
        # UCB Formula for optimistic exploration
        exploration_term = self.c_puct * torch.sqrt(torch.log(parent_visits) / safe_node_visits)
        ucb_score = q_values + exploration_term
        
        confidence = torch.sigmoid(ucb_score)
        return confidence

    def simulate_or_fetch_prm(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        For the scope of Experiment A's diagnostics, this simulates the offline MCTS stats.
        In the full pipeline, this queries the buffer index.
        """
        batch_size, seq_len = input_ids.shape
        
        # Simulated MCTS traversal artifacts
        simulated_q = torch.randn(batch_size, seq_len, device=input_ids.device) * 0.5
        simulated_parent_v = torch.randint(10, 100, (batch_size, seq_len), device=input_ids.device).float()
        simulated_node_v = torch.randint(1, 50, (batch_size, seq_len), device=input_ids.device).float()
        
        return self.compute_ucb_confidence(simulated_q, simulated_parent_v, simulated_node_v)