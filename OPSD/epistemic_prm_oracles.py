import torch
import torch.nn as nn
import numpy as np

class OptimisticUCTOracle(nn.Module):
    """
    Derives uncertainty from MCTS tree structure utilizing optimistic
    exploration via upper confidence bounds.
    """
    def __init__(self, c_puct=1.414):
        super().__init__()
        self.c_puct = c_puct

    def simulate_or_fetch_uct(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Simulates the UCT exploration metric based on branch visitation.
        U_t ~ sqrt(ln(N(s)) / N(s, a))
        """
        batch_size, seq_len = input_ids.shape
        
        # Simulate MCTS visit counts 
        parent_visits = torch.randint(10, 500, (batch_size, seq_len), device=input_ids.device).float()
        node_visits = torch.randint(1, 100, (batch_size, seq_len), device=input_ids.device).float()
        
        # Enforce N(s) >= N(s, a)
        parent_visits = torch.maximum(parent_visits, node_visits)
        
        safe_node_visits = torch.clamp(node_visits, min=1.0)
        uct_uncertainty = self.c_puct * torch.sqrt(torch.log(parent_visits) / safe_node_visits)
        
        # Normalize to [0, 1] for stable CPPO bound scaling
        return torch.clamp(uct_uncertainty, 0.0, 1.0)


class EpinetOracle(nn.Module):
    """
    Simulates an Epistemic Neural Network (Epinet) index over a Process Reward Model.
    Explicitly outputs the variance of the PRM's own predictions.
    """
    def __init__(self, hidden_dim=1024):
        super().__init__()
        # In a full implementation, this maps PRM hidden states to an epistemic index
        self.epistemic_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Softplus() # Variance must be positive
        )

    def simulate_or_fetch_epinet(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Simulates the Epinet variance readout.
        """
        batch_size, seq_len = input_ids.shape
        
        # Simulate PRM feature extraction
        simulated_hidden_states = torch.randn(batch_size, seq_len, 1024, device=input_ids.device)
        
        # Calculate variance
        prm_variance = self.epistemic_head(simulated_hidden_states).squeeze(-1)
        
        # Normalize to [0, 1] for stable CPPO bound scaling
        # High variance = High epistemic uncertainty (model is guessing)
        normalized_variance = torch.clamp(prm_variance / (torch.max(prm_variance) + 1e-8), 0.0, 1.0)
        
        return normalized_variance