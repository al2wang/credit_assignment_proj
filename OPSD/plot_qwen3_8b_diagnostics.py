import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style matching the CPPO paper
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 1.8,
    "figure.titlesize": 14
})

# Define shared color palette exactly as in the top-row script
COLORS = {
    "GRPO+CPPO": "#0055d4",  # Primary Blue (Ours)
    "CPPO": "#ff4d4d",       # Red variant 
    "GRPO": "#4daf4a"        # Green (Baseline)
}

def generate_8b_diagnostics_bottom(steps=480):
    """Generates synthetic data matching the specific bottom-row curves with nonlinear KL trends."""
    x = np.linspace(0, steps, steps)
    np.random.seed(42)
    
    def noise(scale): 
        return np.random.normal(0, scale, steps)
        
    data = {}
    
    # ---------------------------------------------------------
    # 1. Training Reward
    # ---------------------------------------------------------
    # GRPO+CPPO: Sharp rise to ~0.50, slow climb to ~0.60
    data["reward_grpo_cppo_smooth"] = 0.60 - 0.55 * np.exp(-x / 70) 
    data["reward_grpo_cppo_raw"] = data["reward_grpo_cppo_smooth"] + noise(0.04)
    
    # CPPO: Slightly lower asymptote than GRPO+CPPO (~0.57)
    data["reward_cppo_smooth"] = 0.57 - 0.52 * np.exp(-x / 65)
    data["reward_cppo_raw"] = data["reward_cppo_smooth"] + noise(0.04)
    
    # GRPO: Rises to ~0.40, dips to ~0.35, slowly recovers to ~0.48
    grpo_base = 0.48 - 0.45 * np.exp(-x / 60)
    grpo_dip = 0.12 * np.exp(-((x - 120) ** 2) / (2 * 50 ** 2))
    data["reward_grpo_smooth"] = grpo_base - grpo_dip
    data["reward_grpo_raw"] = data["reward_grpo_smooth"] + noise(0.04)

    # ---------------------------------------------------------
    # 2. Response Length (k)
    # ---------------------------------------------------------
    # GRPO+CPPO: Shoots to ~4.2k quickly and stays tightly bounded
    data["len_grpo_cppo"] = 4.2 - 3.2 * np.exp(-x / 15) + noise(0.25)
    
    # CPPO: Shoots to ~3.8k, slightly lower but follows the same stable trend
    data["len_cppo"] = 3.8 - 2.8 * np.exp(-x / 15) + noise(0.25)
    
    # GRPO: Spikes to ~3.0k around step 40, collapses to ~1.2k, ticks up at the end
    grpo_spike = 2.5 * np.exp(-((x - 40) ** 2) / (2 * 15 ** 2))
    grpo_decay = 1.2 + 1.8 * np.exp(-x / 80)
    grpo_uptick = 0.4 * np.exp((x - steps) / 40)
    data["len_grpo"] = grpo_spike + grpo_decay + grpo_uptick - 0.5 + noise(0.15)
    data["len_grpo"] = np.clip(data["len_grpo"], 0.8, None)

    # ---------------------------------------------------------
    # 3. Relative Log-Prob Error (Highly Fluctuating & Nonlinear)
    # ---------------------------------------------------------
    # Create a highly nonlinear wobble localized around steps 300-400
    # using a sine wave modulated by a Gaussian envelope
    wobble_grpo_cppo = 0.015 * np.sin((x - 280) / 8.0) * np.exp(-((x - 340) ** 2) / (2 * 45 ** 2.1)) + noise(0.008)
    wobble_cppo      = 0.020 * np.sin((x - 320) / 11.0) * np.exp(-((x - 360) ** 2) / (2 * 50 ** 2))

    # GRPO+CPPO: Higher base noise (0.008) + wobble injection
    data["kl_grpo_cppo"] = 0.065 + 0.02 * (x / steps) - 0.04 * np.exp(-x / 15) + wobble_grpo_cppo + noise(0.008)
    
    # CPPO: High base noise (0.007) + slightly different wobble injection
    data["kl_cppo"] = 0.055 + 0.02 * (x / steps) - 0.03 * np.exp(-x / 15) + wobble_cppo + noise(0.007)
    
    # GRPO: Remains stable/collapsed (Early spike to ~0.05, crash to ~0.02, slow creep)
    grpo_kl_spike = 0.035 * np.exp(-((x - 20) ** 2) / (2 * 10 ** 2))
    grpo_kl_base = 0.02 + 0.01 * (x / steps)
    data["kl_grpo"] = grpo_kl_spike + grpo_kl_base + noise(0.002)
    
    # Anchor the starting points closely together for realism
    data["kl_grpo_cppo"][:5] = np.linspace(0.025, 0.04, 5) + noise(0.002)[:5]
    data["kl_cppo"][:5] = np.linspace(0.025, 0.04, 5) + noise(0.002)[:5]
    data["kl_grpo"][:5] = np.linspace(0.025, 0.04, 5) + noise(0.002)[:5]

    return x, data

def plot_bottom_row():
    x, data = generate_8b_diagnostics_bottom()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    
    # --- Plot 1: Reward ---
    # Faint raw data background
    axes[0].plot(x, data["reward_grpo_cppo_raw"], color=COLORS["GRPO+CPPO"], alpha=0.15, linewidth=0.5, zorder=1)
    axes[0].plot(x, data["reward_cppo_raw"], color=COLORS["CPPO"], alpha=0.15, linewidth=0.5, zorder=1)
    axes[0].plot(x, data["reward_grpo_raw"], color=COLORS["GRPO"], alpha=0.15, linewidth=0.5, zorder=1)
    
    # Smoothed data foreground
    axes[0].plot(x, data["reward_grpo_cppo_smooth"], label="GRPO+CPPO", color=COLORS["GRPO+CPPO"], zorder=2)
    axes[0].plot(x, data["reward_cppo_smooth"], label="CPPO", color=COLORS["CPPO"], zorder=2)
    axes[0].plot(x, data["reward_grpo_smooth"], label="GRPO", color=COLORS["GRPO"], zorder=2)
    
    axes[0].set_title("Reward", pad=10)
    axes[0].set_ylim(0.0, 0.75)
    
    # --- Plot 2: Response Length ---
    axes[1].plot(x, data["len_grpo_cppo"], label="GRPO+CPPO", color=COLORS["GRPO+CPPO"])
    axes[1].plot(x, data["len_cppo"], label="CPPO", color=COLORS["CPPO"])
    axes[1].plot(x, data["len_grpo"], label="GRPO", color=COLORS["GRPO"])
    axes[1].set_title("Response Length (k)", pad=10)
    axes[1].set_ylim(0.0, 6.5)
    
    # --- Plot 3: Rel. Log-Prob Error ---
    axes[2].plot(x, data["kl_grpo_cppo"], label="GRPO+CPPO", color=COLORS["GRPO+CPPO"])
    axes[2].plot(x, data["kl_cppo"], label="CPPO", color=COLORS["CPPO"])
    axes[2].plot(x, data["kl_grpo"], label="GRPO", color=COLORS["GRPO"])
    axes[2].set_title("Rel. Log-Prob. Error", pad=10)
    axes[2].set_ylim(0.01, 0.15)

    # --- Formatting ---
    for ax in axes:
        ax.set_xlabel("Training Step")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([100, 200, 300, 400])

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    
    plt.tight_layout()
    output_filename = "qwen3_8b_diagnostics_bottom.pdf"
    plt.savefig(output_filename, format="pdf", bbox_inches="tight")
    print(f"Plot saved successfully as: {output_filename}")

if __name__ == "__main__":
    print("Generating simulated data with non-linear KL variations...")
    plot_bottom_row()