import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style to match the paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "lines.linewidth": 2
})

def generate_mock_data(steps):
    """Generates synthetic data matching the trends of CPPO vs GRPO."""
    x = np.linspace(0, 650, steps)
    
    # 1. Reward (Smoothed)
    # CPPO rises higher and stabilizes; GRPO plateaus lower and slightly degrades
    cppo_reward = 0.28 - 0.26 * np.exp(-x / 100) + np.random.normal(0, 0.01, steps)
    grpo_reward = 0.12 - 0.08 * np.exp(-x / 50) + np.random.normal(0, 0.01, steps)
    # Add a slight degradation to GRPO later in training
    grpo_reward -= np.clip((x - 300) / 10000, 0, None)

    # 2. Response Length (k)
    # CPPO stays high (~4.0k); GRPO collapses (~1.0k)
    cppo_length = 3.8 - 3.0 * np.exp(-x / 100) + np.random.normal(0, 0.3, steps)
    # GRPO rises then crashes
    grpo_length = 1.0 + 1.5 * np.exp(-((x - 150) ** 2) / (2 * 50 ** 2)) + np.random.normal(0, 0.2, steps)
    grpo_length = np.clip(grpo_length, 0.5, None) # Prevent negative lengths

    # 3. Relative Log-Prob Error
    # CPPO stays flat ~0.06; GRPO collapses to ~0.02
    cppo_kl = 0.06 - 0.04 * np.exp(-x / 50) + np.random.normal(0, 0.002, steps)
    # GRPO spikes slightly early, then drops
    grpo_kl = 0.02 + 0.03 * np.exp(-x / 100) + np.random.normal(0, 0.002, steps)
    grpo_kl[:50] = np.linspace(0.015, 0.045, 50) + np.random.normal(0, 0.002, 50)

    return x, cppo_reward * 2, grpo_reward * 2.5, cppo_length, grpo_length, cppo_kl, grpo_kl

# Generate data
steps = 300
x, cppo_rew, grpo_rew, cppo_len, grpo_len, cppo_kl, grpo_kl = generate_mock_data(steps)

# Create 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Plot 1: Reward
axes[0].plot(x, cppo_rew, label="CPPO (Ours)", color="#0055d4")
axes[0].plot(x, grpo_rew, label="GRPO", color="#4daf4a")
axes[0].set_title("Reward")
axes[0].set_ylim(0.0, 0.80)

# Plot 2: Response Length
axes[1].plot(x, cppo_len, label="CPPO (Ours)", color="#0055d4")
axes[1].plot(x, grpo_len, label="GRPO", color="#4daf4a")
axes[1].set_title("Response Length (k)")
axes[1].set_ylim(0.0, 6.0)

# Plot 3: Rel. Log-Prob Error
axes[2].plot(x, cppo_kl, label="CPPO (Ours)", color="#0055d4")
axes[2].plot(x, grpo_kl, label="GRPO", color="#4daf4a")
axes[2].set_title("Rel. Log-Prob. Error")
axes[2].set_ylim(0.01, 0.09)

# Formatting
for ax in axes:
    ax.set_xlabel("Training Step")
    # Custom grid lines to match the paper
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Add a shared legend at the bottom
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.05), frameon=False)

plt.tight_layout()

# Save as a high-quality PDF
output_file = "cppo_vs_grpo_diagnostics.pdf"
plt.savefig(output_file, format="pdf", bbox_inches="tight")
print(f"Mock plots successfully generated and saved to: {output_file}")