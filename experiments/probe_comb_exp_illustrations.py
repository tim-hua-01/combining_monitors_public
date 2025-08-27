#%%
from pathlib import Path
project_dir = Path(__file__).parent.parent
import sys
from typing import Optional
sys.path.append(str(project_dir))
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm.auto import tqdm
from strategy_class import (
    SingleMonitorStrategy,
    AuditEndThreshStrategy,
    Strategy,
    NaiveSingleMonitorStrategy,
    NeymanPearsonSingle,
    NeymanPearsonSingleRandom,
    TwoMonitorAlwaysNP,
    FullNPOneRegion
)  # ,BinaryMonitorStrategy, NaiveSingleMonitorStrategy.
from dataclasses import dataclass
import time
import psutil
import matplotlib.pyplot as plt
from math import floor

from calculate_safety import BudgetConfig, calculate_safety_df, resample_with_iid_attack_split, calculate_safety_df_sequential


#%%
in_sample_alpha_attack = pd.read_csv(
    project_dir
    / "data"
    / "processed_monitor_scores"
    / "sonnet_attack.csv"
)["score_fuzzed"].to_numpy()
in_sample_alpha_benign = pd.read_csv(
    project_dir
    / "data"
    / "processed_monitor_scores"
    / "sonnet_honest.csv"
)["score_fuzzed"].to_numpy()
in_sample_beta_attack = pd.read_csv(
    project_dir
    / "data"
    / "processed_monitor_scores"
    / "mean_pairs_attack.csv"
)["score_fuzzed"].to_numpy()
in_sample_beta_benign = pd.read_csv(
    project_dir / "data" / "processed_monitor_scores" / "mean_pairs_honest.csv"
)["score_fuzzed"].to_numpy()

# %%
fullnp = FullNPOneRegion()
fullnp.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

#%%
def plot_decision_regions_custom(
    strategy: FullNPOneRegion, # Takes the strategy instance
    alpha_plot_range: Optional[Tuple[float, float]] = None,
    beta_plot_range: Optional[Tuple[float, float]] = None,
    num_points: int = 50,
    save_path: Optional[str] = None,
    show: bool = True,
    title: str = "Log likelihood ratios at each info set",
    show_audit_threshold: bool = True,
    ) -> None:
    """
    Plots the decision regions, LLR heatmap, and audit cutoff after the model is fitted.
    Alpha scores are on the x-axis, Beta scores on the y-axis.
    Customized for larger font sizes.
    
    Args:
        save_path: Optional path to save the plot (e.g., 'plot.png', 'plot.pdf')
        show: Whether to display the plot (default True)
        title: Title for the plot
        show_audit_threshold: Whether to show the green audit threshold line (default True)
    """
    if not strategy.fitted:
        print("Model has not been fitted yet. Please call fit_method first.")
        return
    
    if strategy.top_thresh is None or strategy.bot_thresh is None or strategy.audit_thresh is None or strategy.N_first is None:
        print("Fit method did not properly set all required thresholds. Cannot plot.")
        return

    # Define score ranges for plotting
    alpha_range = alpha_plot_range if alpha_plot_range is not None else (-4, 4.0)
    beta_range = beta_plot_range if beta_plot_range is not None else (-2.0, 3)

    x_coords = np.linspace(alpha_range[0], alpha_range[1], num_points)
    y_coords = np.linspace(beta_range[0], beta_range[1], num_points)

    Alpha_grid, Beta_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
    alpha_scores_flat = Alpha_grid.ravel()
    beta_scores_flat = Beta_grid.ravel()

    llr_values_flat = np.zeros_like(alpha_scores_flat, dtype=float)

    if strategy.N_first:  # Beta monitor first
        # Region where 2D LLR is used (both monitors)
        mask_2d = (beta_scores_flat >= strategy.bot_thresh) & (beta_scores_flat < strategy.top_thresh)
        # Region where 1D LLR is used (only Beta monitor's score)
        mask_1d = ~mask_2d

        if np.any(mask_2d):
            llr_values_flat[mask_2d] = strategy.get_lnl_ratio(
                alpha_scores=alpha_scores_flat[mask_2d],
                beta_scores=beta_scores_flat[mask_2d]
            )
        if np.any(mask_1d):
            llr_values_flat[mask_1d] = strategy.get_lnl_ratio(
                alpha_scores=None,
                beta_scores=beta_scores_flat[mask_1d]
            )
    else:  # Alpha monitor first
        # Region where 2D LLR is used (both monitors)
        mask_2d = (alpha_scores_flat >= strategy.bot_thresh) & (alpha_scores_flat < strategy.top_thresh)
        # Region where 1D LLR is used (only Alpha monitor's score)
        mask_1d = ~mask_2d

        if np.any(mask_2d):
            llr_values_flat[mask_2d] = strategy.get_lnl_ratio(
                alpha_scores=alpha_scores_flat[mask_2d],
                beta_scores=beta_scores_flat[mask_2d]
            )
        if np.any(mask_1d):
            llr_values_flat[mask_1d] = strategy.get_lnl_ratio(
                alpha_scores=alpha_scores_flat[mask_1d],
                beta_scores=None
            )
    
    LLR_grid = llr_values_flat.reshape(Alpha_grid.shape)

    plt.figure(figsize=(10, 7), dpi = 300)
    
    # Plot LLR heatmap
    plt.imshow(
        LLR_grid.T,  # Transpose for correct orientation
        extent=(alpha_range[0], alpha_range[1], beta_range[0], beta_range[1]),
        origin='lower',
        aspect='auto',
        cmap='plasma',
        interpolation='nearest'
    )
    cb = plt.colorbar()
    cb.set_label('Log Likelihood Ratio (LLR)', fontsize=20) # Increased fontsize
    cb.ax.tick_params(labelsize=18)  # Add this line to increase colorbar tick font size

    # Plot top and bottom thresholds
    if strategy.N_first: # Beta first, thresholds are on beta axis (horizontal lines)
        plt.axhline(y=strategy.top_thresh, color='black', linestyle='--', linewidth=4, label=f'Top Threshold ({strategy.top_thresh:.2f})')
        plt.axhline(y=strategy.bot_thresh, color='black', linestyle=':', linewidth=4, label=f'Bottom Threshold ({strategy.bot_thresh:.2f})')
    else: # Alpha first, thresholds are on alpha axis (vertical lines)
        plt.axvline(x=strategy.top_thresh, color='black', linestyle='--', linewidth=4, label=f'Top Threshold ({strategy.top_thresh:.2f})')
        plt.axvline(x=strategy.bot_thresh, color='black', linestyle=':', linewidth=4, label=f'Bottom Threshold ({strategy.bot_thresh:.2f})')

    # Highlight region where LLR >= audit_thresh
    if show_audit_threshold and LLR_grid.size > 0: #Ensure grid is not empty
        contour_set = plt.contour(
            Alpha_grid, Beta_grid, LLR_grid,
            levels=[strategy.audit_thresh],
            colors='lime', 
            linewidths=2.5,
            linestyles='solid'
        )
        lime_line = plt.Line2D([0], [0], color='lime', lw=2.5, linestyle='solid')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(lime_line)
        labels.append(f'LLR Audit Threshold ({strategy.audit_thresh:.2f})')
        
        plt.legend(handles, labels, loc='best', fontsize=18) # Increased fontsize
    else:
        # Still show legend for thresholds if audit threshold is hidden
        plt.legend(loc='best', fontsize=18)

    plt.xlabel('Black Box Monitor Score', fontsize=20) # Increased fontsize
    plt.ylabel('Probe Monitor Score', fontsize=20) # Increased fontsize
    plt.title(f'{title}', fontsize=22) # Increased fontsize
    
    plt.tick_params(axis='both', which='major', labelsize=16) # Increased tick label size

    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    # Handle saving and/or showing
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory if not showing
#%%
plot_decision_regions_custom(fullnp, title = "Combine monitor scores using likelihood ratios", save_path = "intro.png")

plot_decision_regions_custom(fullnp, title = "Calculating LLR using one or two monitors", save_path = "calculate_llr.png", show_audit_threshold = False)


#%%
# Playign with different kernel widths

fullnp_bd35 = FullNPOneRegion(bandwidth_2d = 0.35, bandwidth_1d = 0.35)
fullnp_bd35.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(fullnp_bd35, title = "Kernel width = 0.35")

#%%

fullnp_bd4 = FullNPOneRegion(bandwidth_2d = 0.4, bandwidth_1d = 0.4)
fullnp_bd4.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(fullnp_bd4, title = "Kernel width = 0.4")
#%%

fullnp_bd6 = FullNPOneRegion(bandwidth_2d = 0.6, bandwidth_1d = 0.6)
fullnp_bd6.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(fullnp_bd6, title = "Kernel width = 0.6")

#%%

fullnp_bd8 = FullNPOneRegion(bandwidth_2d = 0.8, bandwidth_1d = 0.8)
fullnp_bd8.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(fullnp_bd8, title = "Kernel width = 0.8")

#%%

fullnp_bd12 = FullNPOneRegion(bandwidth_2d = 1.2, bandwidth_1d = 1.2)
fullnp_bd12.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(fullnp_bd12, title = "Kernel width = 1.2")


# %%
print(fullnp.get_use_two_monitor_prob())

print(fullnp.get_audit_budget_share())
# %%

np_y12 = FullNPOneRegion()
np_y12.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 1.2, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(np_y12, title = "Increased budget to $1.2, safety = 94.6%", save_path = "npy12.png")
np_y12
# %%
np_y2 = FullNPOneRegion()
np_y2.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 2, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(np_y2, title = "Increased budget to $2, safety = 96.3%", save_path = "npy2.png")

np_y04 = FullNPOneRegion()
np_y04.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_attack, Y = 0.4, c_m = 1, c_n = 1e-6, c_a = 20, verbose=True
)

plot_decision_regions_custom(np_y04, title = "Decreased budget to $0.4, safety = 87,7%", save_path = "npy04.png")
# %%
print(np_y12)
print(np_y2)
# %%
strategies = [FullNPOneRegion(), NeymanPearsonSingleRandom(alpha = True), NeymanPearsonSingleRandom(alpha = False)]

strategynames = {
    strategies[0]: "Two monitors",
    strategies[1]: "BB monitor only",
    strategies[2]: "Probe monitor only",
}
df = calculate_safety_df(
        configs = [BudgetConfig(Y=y, c_m=1, c_n=1e-6, c_a=20)
            for y in [0.8, 1, 1.2,1.4,1.6]],
        strategies = strategies,
        seeds = [-1],
        base_a_attack = in_sample_alpha_attack,
        base_b_attack = in_sample_beta_attack,
        base_a_benign = in_sample_alpha_benign,
        base_b_benign = in_sample_beta_benign,
        base_oos_a_attack = in_sample_alpha_attack,
        base_oos_b_attack = in_sample_beta_attack,
        strategy_names = strategynames,
        checkpoint_interval=20,
        output_file=None, max_workers=8
    )
df.to_csv(project_dir / "experiments" / "probe_comb_exp.csv", index=False)
# %%