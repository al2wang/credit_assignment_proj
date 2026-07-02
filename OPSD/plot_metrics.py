import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Config
WANDB_PROJECT = "alexgywang/OPSD" 
SMOOTHING_WINDOW = 5  # For reward smoothing

def fetch_wandb_data():
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT)
    
    dataframes = []
    for run in runs:
        # Only pull runs that are part of the Qwen 4B sweep
        if "qwen4b_cppo" not in run.name:
            continue
            
        # Download the specific metrics we care about
        history = run.history(keys=["step", "env/reward_mean", "cppo/response_length_k", "cppo/rel_log_prob_error"])
        history['run_name'] = run.name
        
        # Smooth the reward for readability (like the paper)
        if 'env/reward_mean' in history.columns:
            history['smoothed_reward'] = history['env/reward_mean'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()
            
        dataframes.append(history)
        
    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

def plot_paper_diagnostics(df):
    if df.empty:
        print("No data found! Check your WandB project path.")
        return

    # Use Seaborn's deep palette for high contrast lines
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ('smoothed_reward', 'Reward', axes[0], True), 
        ('cppo/response_length_k', 'Response Length (k)', axes[1], False),
        ('cppo/rel_log_prob_error', 'Rel. Log-Prob. Error', axes[2], False)
    ]
    
    for (col, title, ax, show_raw) in metrics:
        if col not in df.columns:
            continue
            
        sns.lineplot(data=df, x="step", y=col, hue="run_name", ax=ax, linewidth=2)
        
        # Overlay raw unsmoothed data faintly for the Reward plot
        if show_raw and 'env/reward_mean' in df.columns:
            for run_name in df['run_name'].unique():
                run_data = df[df['run_name'] == run_name]
                ax.plot(run_data['step'], run_data['env/reward_mean'], alpha=0.15, zorder=0)

        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel("")
        
        # Remove individual legends to create a single shared legend at the bottom
        ax.get_legend().remove()

    # Create a shared, clean legend at the bottom of the figure
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("cppo_training_diagnostics.png", dpi=300, bbox_inches='tight')
    print("Saved plot to cppo_training_diagnostics.png")

if __name__ == "__main__":
    print("Fetching data from WandB...")
    df = fetch_wandb_data()
    print(f"Data fetched. Generating plots...")
    plot_paper_diagnostics(df)