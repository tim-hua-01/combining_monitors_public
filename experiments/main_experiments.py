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

def iid_attack_split_no_resample(
    in_sample_alpha_benign: np.ndarray,
    in_sample_alpha_attack: np.ndarray,
    in_sample_beta_benign: np.ndarray,
    in_sample_beta_attack: np.ndarray,
    oos_alpha_attack: np.ndarray,
    oos_beta_attack: np.ndarray,
    seed: int,
    n_test_samples: int = 100,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Resamples scores for bootstrapping if the seed is not -1. Also returns an i.i.d. split of the in-sample attacks.

    This bootstrapping procedure helps us calculate standard errors of our final safety metrics.
    We might also do some split of the in-sample benign scores in the future to see how badly we're
    sort of overfitting our "cost" component.

    Args:
        in_sample_alpha_benign: Array of benign scores from the alpha monitor.
        in_sample_alpha_attack: Array of attack scores from the alpha monitor.
        in_sample_beta_benign: Array of benign scores from the beta monitor.
        in_sample_beta_attack: Array of attack scores from the beta monitor.
        oos_alpha_attack: Array of out of distribution attack scores from the alpha monitor.
        oos_beta_attack: Array of out of distribution attack scores from the beta monitor.
        seed: Seed for bootstrapping. If -1, no bootstrapping is done.
        n_test_samples: Number of test samples to split from the in-sample attacks.
    """
    if seed == -1:
        return (
            in_sample_alpha_benign,
            in_sample_alpha_attack,
            in_sample_beta_benign,
            in_sample_beta_attack,
            in_sample_alpha_attack,
            in_sample_beta_attack,
            oos_alpha_attack,
            oos_beta_attack,
        )
    else:
        np.random.seed(seed)
        assert n_test_samples * 2 <= len(
            in_sample_alpha_attack
        ), "Not enough attack samples to split"


        # Split attack data into train and test sets
        total_attack_samples = len(in_sample_alpha_attack)
        test_indices = np.random.choice(
            total_attack_samples, size=n_test_samples, replace=False
        )
        train_indices = np.array(
            [i for i in range(total_attack_samples) if i not in test_indices]
        )

        # Split using same indices for alpha and beta
        train_alpha_attack = in_sample_alpha_attack[train_indices]
        train_beta_attack = in_sample_beta_attack[train_indices]
        test_alpha_attack = in_sample_alpha_attack[test_indices]
        test_beta_attack = in_sample_beta_attack[test_indices]

        return (
            in_sample_alpha_benign,
            train_alpha_attack,
            in_sample_beta_benign,
            train_beta_attack,
            test_alpha_attack,
            test_beta_attack,
            oos_alpha_attack,
            oos_beta_attack,
        )


def _process_single_item_no_resample(
    seed: int,
    config: BudgetConfig,
    strategy: Strategy,
    strategy_name: str,
    base_a_benign: np.ndarray,
    base_a_attack: np.ndarray,
    base_b_benign: np.ndarray,
    base_b_attack: np.ndarray,
    base_oos_a_attack: np.ndarray,
    base_oos_b_attack: np.ndarray,
) -> dict:
    """Process a single combination of seed, config, and strategy. Returns a dictionary of key metrics.

    Args:
        seed: Seed for bootstrapping. If -1, no bootstrapping is done.
        config: Budget config.
        strategy: Strategy to calculate safety for.
        strategy_name: Name of the strategy.
        base_a_benign: Array of benign scores from the alpha monitor.
        base_a_attack: Array of attack scores from the alpha monitor.
        base_b_benign: Array of benign scores from the beta monitor.
        base_b_attack: Array of attack scores from the beta monitor.
        base_oos_a_attack: Array of out of distribution attack scores from the alpha monitor.
    """
    (
        in_sample_alpha_benign,
        train_alpha_attack,
        in_sample_beta_benign,
        train_beta_attack,
        test_alpha_attack,
        test_beta_attack,
        oos_alpha_attack,
        oos_beta_attack,
    ) = iid_attack_split_no_resample(
        base_a_benign,
        base_a_attack,
        base_b_benign,
        base_b_attack,
        base_oos_a_attack,
        base_oos_b_attack,
        seed,
    )
    strategy.reset()
    strategy.fit_method(
        in_sample_alpha_benign=in_sample_alpha_benign,
        in_sample_alpha_attack=train_alpha_attack,
        in_sample_beta_benign=in_sample_beta_benign,
        in_sample_beta_attack=train_beta_attack,
        Y=config.Y,
        c_m=config.c_m,
        c_n=config.c_n,
        c_a=config.c_a,
    )
    return {
        "seed": seed,
        "Y": config.Y,
        "c_m": config.c_m,
        "c_n": config.c_n,
        "c_a": config.c_a,
        "strategy": strategy_name,
        "insample_safety": strategy.get_insample_safety(),
        "iid_test_safety": strategy.get_oos_safety(test_alpha_attack, test_beta_attack),
        "oos_safety": strategy.get_oos_safety(
            oos_alpha_attack, oos_beta_attack
        ),
        "audit_fpr": strategy.get_audit_fpr(),
        "audit_budget_share": strategy.get_audit_budget_share(),
        "use_two_monitor_prob": strategy.get_use_two_monitor_prob(),
        "strategy_params": strategy.save_parameters(),
    }


#%%
if __name__ == "__main__":
    start_time = time.time()
    in_sample_alpha_attack = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "BCB2_llama3170B_5_attack.csv"
    )["score_fuzzed"].to_numpy()
    in_sample_alpha_benign = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "BCB_llama3170B_5_honest.csv"
    )["score_fuzzed"].to_numpy()
    oos_alpha_attack_hard = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "APPS3_llama3170B_attack.csv"
    )["score_fuzzed"].to_numpy()
    #Not actually used just there to fill out 
    in_sample_beta_attack = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "BCB2_llama318B_5_attack.csv"
    )["score_fuzzed"].to_numpy()
    in_sample_beta_benign = pd.read_csv(
        project_dir / "data" / "processed_monitor_scores" / "BCB_llama318B_5_honest.csv"
    )["score_fuzzed"].to_numpy()
    oos_beta_attack_hard = pd.read_csv(
        project_dir / "data" / "processed_monitor_scores" / "APPS3_llama318B_attack.csv"
    )["score_fuzzed"].to_numpy()

    print(f"Total CPU cores available: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU cores available: {psutil.cpu_count(logical=False)}")
    #%%
    print("Experiment 1")
    npstrat = NeymanPearsonSingle(alpha=True, bandwidth='silverman')
    npstrat.fit_method(
    in_sample_alpha_benign = in_sample_alpha_benign,
    in_sample_alpha_attack=in_sample_alpha_attack,
    in_sample_beta_benign = in_sample_beta_benign,
    in_sample_beta_attack=in_sample_beta_benign, Y = 1.2, c_m = 1, c_n = 0.21, c_a = 20
    )
    n_above = floor(len(npstrat.benign_data)*0.02)
    fpr2_thresh = np.sort(npstrat.benign_ln_likelihood_ratios)[::-1][n_above - 1]
    n_above = floor(len(npstrat.benign_data)*0.03)
    fpr3_thresh = np.sort(npstrat.benign_ln_likelihood_ratios)[::-1][n_above - 1]

    plt.rcParams.update({
        'font.size': 14,      # Base font size
        'axes.labelsize': 16, # Axis label size
        'axes.titlesize': 18, # Title size
        'legend.fontsize': 16 # Legend size
    })
    plt.figure(figsize=(10, 6), dpi=300)

    # Plot the line
    plt.plot(np.arange(2, 9, 0.05), 
            npstrat.attack_kde.score_samples(np.arange(2, 9, 0.05).reshape(-1, 1)) - 
            npstrat.benign_kde.score_samples(np.arange(2, 9, 0.05).reshape(-1, 1)))

    plt.axhline(y=npstrat.audit_thresh, color='r', linestyle='-', label='1% FPR')
    plt.axhline(y=fpr2_thresh, color='g', linestyle='-', label='2% FPR')
    plt.axhline(y=fpr3_thresh, color='purple', linestyle='-', label='3% FPR')

    # Set integer x-axis ticks from 0 to 10
    plt.xticks(range(2,10))  # This creates ticks at 0,1,2,...,10
    
    # Add title and labels
    plt.suptitle("Log likelihood ratio at different scores", fontsize=24)
    plt.title("The curved down region around 8 do not get audited at low FPR", fontsize=14)
    plt.xlabel('Monitor score')
    plt.ylabel('Log likelihood ratio')
    plt.legend(loc='best')

    # Add a grid
    plt.grid(True)
    #%%
    #plt.savefig(project_dir / "results" / "onemon_likelihood.png", dpi=300)
    plt.close()
    strategies = [NaiveSingleMonitorStrategy(alpha = True),
                  NeymanPearsonSingle(alpha=True, bandwidth='silverman'),
                  ]
    
    configs = [BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=20)
            for y in [1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]]

    strategynames = {
        strategies[0]: "Naive single monitor",
        strategies[1]: "Optimized with Neyman-Pearson",
    }

    
    df = calculate_safety_df(
        configs = configs,
        strategies = strategies,
        seeds = [-1] + list(range(1,501)),
        base_a_attack = in_sample_alpha_attack,
        base_b_attack = in_sample_beta_attack,
        base_a_benign = in_sample_alpha_benign,
        base_b_benign = in_sample_beta_benign,
        base_oos_a_attack = oos_alpha_attack_hard,
        base_oos_b_attack = oos_beta_attack_hard,
        strategy_names = strategynames,
        max_workers=125, #The more the merrier
        checkpoint_interval=20,
        output_file=None,
        single_processor_function=_process_single_item_no_resample
    )
    df.to_csv(project_dir / "results" / "np_vs_naive_no_resample_hard.csv", index=False)

    #%%
    strategies = [NaiveSingleMonitorStrategy(alpha = True),
                  NeymanPearsonSingle(alpha=True, bandwidth='silverman'),NeymanPearsonSingleRandom(alpha=True, bandwidth='silverman')]
    
    configs = [BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=20)
            for y in [1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2]]
    print("Experiment 2")
    strategynames = {
        strategies[0]: "Naive single monitor",
        strategies[1]: "Neyman-Pearson",
        strategies[2]: "Neyman-Pearson Stochastic",
    }

    
    df = calculate_safety_df(
        configs = configs,
        strategies = strategies,
        seeds = [-1] + list(range(1,501)),
        base_a_attack = in_sample_alpha_attack,
        base_b_attack = in_sample_beta_attack,
        base_a_benign = in_sample_alpha_benign,
        base_b_benign = in_sample_beta_benign,
        base_oos_a_attack = oos_alpha_attack_hard,
        base_oos_b_attack = oos_beta_attack_hard,
        strategy_names = strategynames,
        max_workers=125, #The more the merrier
        checkpoint_interval=20,
        output_file=None,
        single_processor_function=_process_single_item_no_resample
    )
    df.to_csv(project_dir / "results" / "np_naive_stochastic_hard.csv", index=False)

    #%%
    configs = [BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=a)
            for y, a in [(1.1, 10), (1.2, 20), (1.4,40), (1.8, 80), (2.6, 160), (4.2, 320), (7.4, 640), (13.8,1280), (26.6, 2560)] ]

    print("Experiment 3 (c_a sweep)")
    df = calculate_safety_df(
        configs = configs,
        strategies = strategies,
        seeds = [-1] + list(range(1,501)),
        base_a_attack = in_sample_alpha_attack,
        base_b_attack = in_sample_beta_attack,
        base_a_benign = in_sample_alpha_benign,
        base_b_benign = in_sample_beta_benign,
        base_oos_a_attack = oos_alpha_attack_hard,
        base_oos_b_attack = oos_beta_attack_hard,
        strategy_names = strategynames,
        max_workers=125, #The more the merrier
        checkpoint_interval=20,
        output_file=None,
        single_processor_function=_process_single_item_no_resample
    )
    df.to_csv(project_dir / "results" / "np_naive_stochastic_casweep_hard.csv", index=False)


    #NeymanPearsonSingleRandom(alpha=True, bandwidth='silverman')
    #%%
    """
    fullnp = FullNPOneRegion()
    fullnp.fit_method(
        in_sample_alpha_benign=in_sample_alpha_benign,
        in_sample_alpha_attack=in_sample_alpha_attack,
        in_sample_beta_benign=in_sample_beta_benign,
        in_sample_beta_attack=in_sample_beta_attack,
        Y=1.2,
        c_m=1,
        c_n=0.21,
        c_a=20
    )
    """
    #%%
    def plot_decision_regions_custom(
    strategy: FullNPOneRegion, # Takes the strategy instance
    alpha_plot_range: Optional[Tuple[float, float]] = None,
    beta_plot_range: Optional[Tuple[float, float]] = None,
    num_points: int = 50,
) -> None:
        """
        Plots the decision regions, LLR heatmap, and audit cutoff after the model is fitted.
        Alpha scores are on the x-axis, Beta scores on the y-axis.
        Customized for larger font sizes.
        """
        if not strategy.fitted:
            print("Model has not been fitted yet. Please call fit_method first.")
            return
        
        if strategy.top_thresh is None or strategy.bot_thresh is None or strategy.audit_thresh is None or strategy.N_first is None:
            print("Fit method did not properly set all required thresholds. Cannot plot.")
            return

        # Define score ranges for plotting
        alpha_range = alpha_plot_range if alpha_plot_range is not None else (2.0, 9)
        beta_range = beta_plot_range if beta_plot_range is not None else (4.0, 7.5)

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
        if LLR_grid.size > 0 : #Ensure grid is not empty
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

        plt.xlabel('Llama 3.1 70B Score', fontsize=20) # Increased fontsize
        plt.ylabel('Llama 3.1 8 Score', fontsize=20) # Increased fontsize
        plt.title(f'Log likelihood ratios at each info set', fontsize=22) # Increased fontsize
        
        plt.tick_params(axis='both', which='major', labelsize=16) # Increased tick label size

        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()
    #plot_decision_regions_custom(fullnp)





    #%%
    print("Experiment 4 two monitors")
    strategies = [NeymanPearsonSingleRandom(alpha = True),
                  NeymanPearsonSingleRandom(alpha = False),
                  TwoMonitorAlwaysNP(),
                  FullNPOneRegion()]

    strategynames = {
        strategies[0]: "Neyman-Pearson Alpha",
        strategies[1]: "Neyman-Pearson Beta",
        strategies[2]: "Two monitors always",
        strategies[3]: "One second monitor region",
    }

    configs = [BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=20)
            for y in [0.25,0.5, 0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]]
    
    """
    df = calculate_safety_df(
        configs = configs,
        strategies = strategies,
        seeds = [-1] + list(range(1,100)),
        base_a_attack = in_sample_alpha_attack,
        base_b_attack = in_sample_beta_attack,
        base_a_benign = in_sample_alpha_benign,
        base_b_benign = in_sample_beta_benign,
        base_oos_a_attack = oos_alpha_attack_hard,
        base_oos_b_attack = oos_beta_attack_hard,
        strategy_names = strategynames,
        max_workers=87, #The more the merrier
        checkpoint_interval=20,
        output_file=None,
        single_processor_function=_process_single_item_no_resample
    )
    df.to_csv(project_dir / "results" / "two_mon_v1.csv", index=False)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
    """
# %%
