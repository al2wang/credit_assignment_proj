import torch
import torch.nn as nn

class LangevinAdvantageInjector(nn.Module):
    """
    Implements continuous exploration by injecting Langevin noise 
    into the advantage estimates to simulate uncertainty-driven critic learning.
    """
    def __init__(self, base_temperature=0.15, decay_rate=0.999):
        super().__init__()
        self.temperature = base_temperature
        self.decay_rate = decay_rate
        self.step_count = 0

    def forward(self, advantages: torch.Tensor) -> torch.Tensor:
        """
        Injects properly scaled Langevin noise into the advantage tensor.
        
        Math: A_noisy = A + sqrt(2 * temperature) * N(0, 1) * A_std
        The noise is scaled by the empirical standard deviation of the group
        advantages to maintain the proper order of magnitude for the policy gradient.
        """
        self.step_count += 1
        current_temp = self.temperature * (self.decay_rate ** self.step_count)
        
        # Calculate standard deviation of advantages to scale the noise appropriately
        adv_std = torch.std(advantages, dim=0, keepdim=True) + 1e-8
        
        # Langevin noise injection
        langevin_noise = torch.randn_like(advantages) * torch.sqrt(torch.tensor(2.0 * current_temp))
        noisy_advantages = advantages + (langevin_noise * adv_std)
        
        return noisy_advantages