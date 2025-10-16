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

if __name__ == "__main__":
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

    #%%
    strategies = [FullNPOneRegion(), NeymanPearsonSingleRandom(alpha = True), NeymanPearsonSingleRandom(alpha = False)]

    strategynames = {
        strategies[0]: "Two monitors",
        strategies[1]: "BB monitor only",
        strategies[2]: "Probe monitor only",
    }
    df = calculate_safety_df(
            configs = [BudgetConfig(Y=y, c_m=1, c_n=1e-6, c_a=20)
                for y in [0.05,0.1,0.2]],
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
    df.to_csv(project_dir / "experiments" / "probe_comb_exp3.csv", index=False)