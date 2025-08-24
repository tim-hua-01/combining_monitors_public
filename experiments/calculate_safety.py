# %%

from pathlib import Path

project_dir = Path(__file__).parent.parent
import sys

sys.path.append(str(project_dir))
import numpy as np
import pandas as pd
from typing import Tuple, Callable, Any
from tqdm.auto import tqdm
from strategy_class import (
    SingleMonitorStrategy,
    AuditEndThreshStrategy,
    Strategy,
)  # ,BinaryMonitorStrategy, NaiveSingleMonitorStrategy.
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools
import time
import psutil
# %%


@dataclass
class BudgetConfig:
    Y: float
    c_m: float
    c_n: float
    c_a: float


def resample_with_iid_attack_split(
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

        # Resample benign data with replacement using same indices for alpha and beta
        # Need to do this together to preserve the joint distribution between alpha and beta
        benign_indices = np.random.choice(
            len(in_sample_alpha_benign), size=len(in_sample_alpha_benign), replace=True
        )
        resampled_alpha_benign = in_sample_alpha_benign[benign_indices]
        resampled_beta_benign = in_sample_beta_benign[benign_indices]

        # Add small noise to break ties
        resampled_alpha_benign += np.random.normal(
            0, 1e-8, size=len(resampled_alpha_benign)
        )
        resampled_beta_benign += np.random.normal(
            0, 1e-8, size=len(resampled_beta_benign)
        )

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

        # Resample train and test sets using same indices for alpha and beta
        train_resample_indices = np.random.choice(
            len(train_alpha_attack), size=len(train_alpha_attack), replace=True
        )
        train_alpha_attack = train_alpha_attack[train_resample_indices]
        train_beta_attack = train_beta_attack[train_resample_indices]

        test_resample_indices = np.random.choice(
            len(test_alpha_attack), size=len(test_alpha_attack), replace=True
        )
        test_alpha_attack = test_alpha_attack[test_resample_indices]
        test_beta_attack = test_beta_attack[test_resample_indices]

        # Add noise
        train_alpha_attack += np.random.normal(0, 1e-8, size=len(train_alpha_attack))
        train_beta_attack += np.random.normal(0, 1e-8, size=len(train_beta_attack))
        test_alpha_attack += np.random.normal(0, 1e-8, size=len(test_alpha_attack))
        test_beta_attack += np.random.normal(0, 1e-8, size=len(test_beta_attack))

        # Resample OOS attack data using same indices for alpha and beta
        oos_indices = np.random.choice(
            len(oos_alpha_attack), size=len(oos_alpha_attack), replace=True
        )
        resampled_oos_alpha_attack = oos_alpha_attack[oos_indices]
        resampled_oos_beta_attack = oos_beta_attack[oos_indices]

        # Add noise
        resampled_oos_alpha_attack += np.random.normal(
            0, 1e-8, size=len(resampled_oos_alpha_attack)
        )
        resampled_oos_beta_attack += np.random.normal(
            0, 1e-8, size=len(resampled_oos_beta_attack)
        )

        return (
            resampled_alpha_benign,
            train_alpha_attack,
            resampled_beta_benign,
            train_beta_attack,
            test_alpha_attack,
            test_beta_attack,
            resampled_oos_alpha_attack,
            resampled_oos_beta_attack,
        )


def _process_single_item(
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
        resampled_alpha_benign,
        train_alpha_attack,
        resampled_beta_benign,
        train_beta_attack,
        test_alpha_attack,
        test_beta_attack,
        resampled_oos_alpha_attack,
        resampled_oos_beta_attack,
    ) = resample_with_iid_attack_split(
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
        in_sample_alpha_benign=resampled_alpha_benign,
        in_sample_alpha_attack=train_alpha_attack,
        in_sample_beta_benign=resampled_beta_benign,
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
            resampled_oos_alpha_attack, resampled_oos_beta_attack
        ),
        "audit_fpr": strategy.get_audit_fpr(),
        "audit_budget_share": strategy.get_audit_budget_share(),
        "use_two_monitor_prob": strategy.get_use_two_monitor_prob(),
        "strategy_params": strategy.save_parameters(),
    }


def calculate_safety_df(
    configs: list[BudgetConfig],
    strategies: list[Strategy],
    seeds: list[int],
    base_a_attack: np.ndarray,
    base_b_attack: np.ndarray,
    base_a_benign: np.ndarray,
    base_b_benign: np.ndarray,
    base_oos_a_attack: np.ndarray,
    base_oos_b_attack: np.ndarray,
    strategy_names: dict[Strategy, str] | None = None,
    max_workers: int | None = None,
    output_file: Path | str | None = None,
    checkpoint_interval: int = 20,
    single_processor_function: Callable[..., dict] = _process_single_item,
) -> pd.DataFrame:
    """
    Calculate the safety of the strategies for the given configs, seeds, and base scores.
    Parallelized to utilize multiple cores.

    I usually just rent a high cpu count machine on vast.ai to run these.

    Args:
        configs: List of budget configs to calculate safety for.
        strategies: List of strategies to calculate safety for.
        seeds: List of seeds to bootstrap our dataset with. seed = -1 uses the original dataset.
        base_a_attack: Array of attack scores from the alpha monitor.
        base_b_attack: Array of attack scores from the beta monitor.
        base_a_benign: Array of benign scores from the alpha monitor.
        base_b_benign: Array of benign scores from the beta monitor.
        base_oos_a_attack: Array of out of distribution attack scores from the alpha monitor.
        base_oos_b_attack: Array of out of distribution attack scores from the beta monitor.
        strategy_names: Dictionary of strategy names.
        max_workers: Maximum number of workers to use for parallel processing.
        output_file: Path to save incremental results to. If None, results are only returned at the end.
        checkpoint_interval: Save results to file after this many completed tasks.
    """
    tasks = []
    for seed in seeds:
        for config in configs:
            for strategy in strategies:
                strategy_name = (
                    strategy_names[strategy] if strategy_names else str(strategy)
                )
                tasks.append((seed, config, strategy, strategy_name))

    results = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Add this code to check actual number of workers
        actual_max_workers = executor._max_workers  # type: ignore
        print(f"ProcessPoolExecutor created with {actual_max_workers} workers")

        process_func = functools.partial(
            single_processor_function,
            base_a_benign=base_a_benign,
            base_a_attack=base_a_attack,
            base_b_benign=base_b_benign,
            base_b_attack=base_b_attack,
            base_oos_a_attack=base_oos_a_attack,
            base_oos_b_attack=base_oos_b_attack,
        )

        # Create a list of futures
        futures = [
            executor.submit(process_func, seed, config, strategy, strategy_name)
            for seed, config, strategy, strategy_name in tasks
        ]

        # Process results as they complete (with progress bar)
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing tasks"
        ):
            result = future.result()
            results.append(result)
            completed_count += 1

            # Save intermediate results at regular intervals
            if output_file is not None and (
                completed_count % checkpoint_interval == 0
                or completed_count == len(futures)
            ):
                pd.DataFrame(results).to_csv(output_file, index=False, mode="w")

    return pd.DataFrame(results)


def calculate_safety_df_sequential(
    configs: list[BudgetConfig],
    strategies: list[Strategy],
    seeds: list[int],
    base_a_attack: np.ndarray,
    base_b_attack: np.ndarray,
    base_a_benign: np.ndarray,
    base_b_benign: np.ndarray,
    base_oos_a_attack: np.ndarray,
    base_oos_b_attack: np.ndarray,
    strategy_names: dict[Strategy, str] | None = None,
    max_workers: int | None = None,  # Kept for API compatibility
    single_processor_function: Callable[..., dict] = _process_single_item,
) -> pd.DataFrame:
    """
    Sequential version of calculate_safety_df that helps with debugging.
    Calculates the safety of strategies for given configs, seeds, and base scores.
    """
    results = []
    total_tasks = len(seeds) * len(configs) * len(strategies)
    current_task = 0

    print(f"Starting sequential processing of {total_tasks} tasks")

    for seed in seeds:
        for config in configs:
            for strategy in strategies:
                current_task += 1
                strategy_name = (
                    strategy_names[strategy] if strategy_names else str(strategy)
                )

                print(
                    f"[{current_task}/{total_tasks}] Processing: seed={seed}, Y={config.Y}, c_m={config.c_m}, "
                    f"c_n={config.c_n}, c_a={config.c_a}, strategy={strategy_name}"
                )

                result = single_processor_function(
                    seed=seed,
                    config=config,
                    strategy=strategy,
                    strategy_name=strategy_name,
                    base_a_benign=base_a_benign,
                    base_a_attack=base_a_attack,
                    base_b_benign=base_b_benign,
                    base_b_attack=base_b_attack,
                    base_oos_a_attack=base_oos_a_attack,
                    base_oos_b_attack=base_oos_b_attack,
                )

                # Print some result information for debugging
                print(
                    f"  Result: insample_safety={result['insample_safety']:.4f}, "
                    f"oos_safety={result['oos_safety']:.4f}, "
                    f"audit_fpr={result['audit_fpr']:.4f}"
                )

                results.append(result)

    print(f"Completed all {total_tasks} tasks")
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Various budget configurations.
    # Some of these costs are selected so that we can always use one of the monitors,
    # and then get a 1% or 2% FPR, similar to the original control paper.
    configs = (
        [
            BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=20)
            for y in [0.41, 0.61, 1, 1.2, 1.4, 2, 3]
        ]
        + [
            BudgetConfig(Y=y, c_m=1, c_n=0.21, c_a=100)
            for y in [1.21, 1.5, 2, 2.21, 3, 4, 6]
        ]
        + [
            BudgetConfig(Y=10, c_m=1, c_n=0.21, c_a=a)
            for a in [50, 100, 200, 500, 1000]
        ]
    )
    strategies = [
        AuditEndThreshStrategy(verbose=False),
        SingleMonitorStrategy(alpha=True),
        SingleMonitorStrategy(alpha=False),
    ]  # ,# BinaryMonitorStrategy(verbose = False)
    strategy_names = {
        strategies[0]: "AuditEndThresh",
        strategies[1]: "AlphaMonitorOnly",
        strategies[2]: "BetaMonitorOnly",
    }
    seeds = [-1] + list(range(1, 51))
    # Load in the monitor scores
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
    oos_alpha_attack = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "APPS3_llama3170B_attack.csv"
    )["score_fuzzed"].to_numpy()

    in_sample_beta_attack = pd.read_csv(
        project_dir
        / "data"
        / "processed_monitor_scores"
        / "BCB2_llama318B_5_attack.csv"
    )["score_fuzzed"].to_numpy()
    in_sample_beta_benign = pd.read_csv(
        project_dir / "data" / "processed_monitor_scores" / "BCB_llama318B_5_honest.csv"
    )["score_fuzzed"].to_numpy()
    oos_beta_attack = pd.read_csv(
        project_dir / "data" / "processed_monitor_scores" / "APPS3_llama318B_attack.csv"
    )["score_fuzzed"].to_numpy()

    print(f"Total CPU cores available: {psutil.cpu_count(logical=True)}")
    print(f"Physical CPU cores available: {psutil.cpu_count(logical=False)}")

    start_time = time.time()
    df = calculate_safety_df(
        configs,
        strategies,
        seeds,
        in_sample_alpha_attack,
        in_sample_beta_attack,
        in_sample_alpha_benign,
        in_sample_beta_benign,
        oos_alpha_attack,
        oos_beta_attack,
        strategy_names,
        max_workers=None, #The more the merrier
        checkpoint_interval=10,
        output_file=project_dir / "results" / "ongoing_run" / "mats_sample_run.csv",
    )

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    df.to_csv(project_dir / "results" / "mats_sample_run.csv", index=False)
