import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticReplayBuffer:
    """
    Simulates a vector-database replay buffer to store prompt embeddings 
    and historical outcomes for cross-sequence knowledge sharing.
    """
    def __init__(self, embedding_dim=1024, max_size=50000):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        
        # Buffer storage
        self.embeddings = []
        self.rewards = []
        self.size = 0
        
    def add_trajectory(self, prompt_embedding: torch.Tensor, reward: float):
        """Asynchronously called after generation to store the prompt semantics and outcome."""
        # For simplicity in this diagnostic script, we hold it in memory
        if self.size < self.max_size:
            self.embeddings.append(prompt_embedding.detach().cpu())
            self.rewards.append(reward)
            self.size += 1
        else:
            # Simple FIFO eviction
            idx = np.random.randint(0, self.max_size)
            self.embeddings[idx] = prompt_embedding.detach().cpu()
            self.rewards[idx] = reward

    def get_knn_baseline(self, query_embedding: torch.Tensor, k=5) -> torch.Tensor:
        """
        Retrieves the KNN historical baseline b(x_i) from the buffer.
        """
        if self.size < k:
            return torch.zeros(query_embedding.size(0), device=query_embedding.device)
            
        query = query_embedding.detach().cpu()
        keys = torch.stack(self.embeddings) # (Buffer_size, Dim)
        rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32)
        
        # Compute cosine similarity
        query_norm = F.normalize(query, p=2, dim=-1)
        keys_norm = F.normalize(keys, p=2, dim=-1)
        similarity = torch.matmul(query_norm, keys_norm.T) # (Batch, Buffer_size)
        
        # Get top-K matches
        top_k_sims, top_k_indices = torch.topk(similarity, k=k, dim=-1)
        
        # Calculate weighted average of historical rewards
        # Softmax over similarities for weighting
        weights = F.softmax(top_k_sims, dim=-1)
        matched_rewards = rewards_tensor[top_k_indices] # (Batch, K)
        
        knn_baseline = torch.sum(weights * matched_rewards, dim=-1)
        return knn_baseline.to(query_embedding.device)


class ProcessValueCritic(nn.Module):
    """
    Simulates the offline trained V(s) and Q(s,c) models.
    Provides chunk-level advantages and epistemic uncertainty mapping.
    """
    def __init__(self):
        super().__init__()
        
    def fetch_chunk_advantage_and_uncertainty(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            A_chunk: Q(s, c) - V(s)
            U: Epistemic uncertainty [0, 1]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Simulate trained Process-Value returns
        # In reality, this evaluates the generated tokens against the offline critic
        q_s_c = torch.randn(batch_size, seq_len, device=device) * 0.5 + 0.5
        v_s = torch.randn(batch_size, seq_len, device=device) * 0.4 + 0.5
        
        a_chunk = q_s_c - v_s
        
        # Simulate converging epistemic uncertainty based on visit counts/buffer density
        # For mock purposes, drops over sequence length to simulate confidence in latter stages
        uncertainty = 0.8 * torch.exp(-torch.arange(seq_len, device=device).float() / 200).unsqueeze(0).expand(batch_size, seq_len)
        uncertainty = torch.clamp(uncertainty + (torch.randn_like(uncertainty) * 0.05), 0.0, 1.0)
        
        return a_chunk, uncertainty