# %%
from typing import Callable, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import sys
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
from naive_single_monitor import get_naive_single_monitor_policy

def solve_1mon_optimal_policy(
    c_a: float, 
    c_m: float, 
    Y: float,
    attack_scores: np.ndarray,
    honest_scores: np.ndarray,
    a_m_min: Optional[float] = None,
    a_m_max: Optional[float] = None,
    n_ptile_points: int = 100,
    additional_am_around_naive: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Solve the corner-based optimization problem:
    
       max_{a_m, p_m, a_s}  p_m [1 - F_Q1(a_m)] + a_s
    
    subject to:
       (1) a_s + p_m <= 1
       (2) [c_m + c_a(1 - F_Q0(a_m))] * p_m + c_a * a_s <= Y
       (3) 0 <= p_m, a_s <= 1
    
    Here p_m is the probability of calling the monitor, a_m is the threshold for the monitor, a_s is the probability of auditing right away before the monitor is called, and F_Q1 is the cumulative distribution function of the attack scores, and F_Q0 is the cumulative distribution function of the honest scores. 

    (In reality we do not use cumulative distribution functions because those are fickle. We shouldn't use quantiles either tbh. Should just use np.mean() and stuff. )
    
    
    Parameters
    ----------
    c_a : float
        The per-unit 'cost' of awarding action a_s.
    c_m : float
        The per-unit 'cost' of awarding action p_m (with threshold).
    Y : float
        The overall budget constraint.
    attack_scores : np.ndarray
        Raw monitor scores for attack cases (Q=1).
    honest_scores : np.ndarray
        Raw monitor scores for honest cases (Q=0).
    a_m_min : float, optional
        Minimum threshold value (default: min of honest_scores).
    a_m_max : float, optional
        Maximum threshold value (default: max of honest_scores).
    n_ptile_points : int
        Number of percentile points to search over.
    
    Returns
    -------
    best_objective : float
        The maximum probability of A=1 given Q=1.
    best_a_m : float
        The threshold a_m achieving that solution.
    best_p_m : float
        The probability p_m at that solution.
    best_a_s : float
        The probability a_s at that solution.
    """
    # Set default min/max if not provided
    if a_m_min is None:
        a_m_min = np.min(honest_scores)
    if a_m_max is None:
        a_m_max = np.max(honest_scores)
    

    # alpha(a_m) = c_m + c_a * (1 - F_Q0(a_m))
    def monitor_and_audit_cost(a_m: float) -> float:
        return c_m + c_a*np.mean(honest_scores >= a_m) #type: ignore

    # ------------------------------------------------
    # 3) Corner 1: p_m = 0
    #    => a_s <= 1, and c_a * a_s <= Y
    #    => a_s = min(1, Y / c_a)
    # ------------------------------------------------
    # This corner is independent of a_m
    a_s_corner1 = min(1.0, Y/c_a)
    corner1_obj = a_s_corner1  # p_m=0 => objective = a_s

    # ------------------------------------------------
    # Generate threshold values based on percentiles
    # ------------------------------------------------
    # Filter scores to be within the specified range
    filtered_scores = honest_scores[(honest_scores >= a_m_min) & (honest_scores <= a_m_max)]
    unique_scores = np.unique(filtered_scores)
    
    # If we have fewer unique values than requested percentile points, use all unique values
    if len(unique_scores) <= n_ptile_points:
        a_m_values = unique_scores
    else:
        # Generate percentile-based thresholds
        percentiles = np.linspace(0, 100, n_ptile_points)
        a_m_values = np.percentile(filtered_scores, percentiles)
    
    # We want to always check the performance of the naive strategy (i.e., 
    # use monitor all the time and set the audit threshold to exhaust our audit budget). 
    if Y - c_m > 0:
        naive_a_m = get_naive_single_monitor_policy(honest_scores, Y, c_a, c_m)
    else:
        naive_a_m = a_m_max

    additional_a_m_values = np.array([naive_a_m])
    if naive_a_m < a_m_max:
        sorted_honest_scores = np.sort(honest_scores)
        naive_am_idx = np.sum(sorted_honest_scores < naive_a_m)
        ids_to_add = [(i-20)*5 + naive_am_idx for i in range(40) if (i-20)*5 + naive_am_idx < len(sorted_honest_scores) and (i-20)*5 + naive_am_idx > 0]
        additional_a_m_values = sorted_honest_scores[ids_to_add]
    
    a_m_values = np.concatenate((a_m_values, additional_a_m_values))
    a_m_values = np.unique(a_m_values)
    best_objective = -1.0
    best_a_m = 0.0
    best_p_m = 0.0
    best_a_s = 0.0
    
    for a_m in a_m_values:
        # Corner 1's objective (same for all a_m)
        obj_c1 = corner1_obj
        # ------------------------------------------------
        # Corner 2: a_s=0
        #   => p_m <= 1 and alpha(a_m)*p_m <= Y
        #   => p_m = min(1, Y / alpha(a_m))
        #   => objective = p_m * [1 - F_Q1(a_m)]
        # ------------------------------------------------
        denom = monitor_and_audit_cost(a_m)
        if denom > 1e-14:  # to avoid divide-by-zero
            pm_c2 = min(1.0, Y/denom)
        else:
            assert False, f"monitor and audit cost is too small what is going on your c_m is {c_m} and c_a is {c_a}"
        obj_c2 = pm_c2 * np.mean(attack_scores >= a_m)

        # ------------------------------------------------
        # Corner 3: 
        #   p_m + a_s = 1,
        #   alpha(a_m)*p_m + c_a*a_s = Y
        #   Y - c_a(1- p_m) = alpha(a_m)*p_m
        #   Y - c_a = alpha(a_m)*p_m - c_a*p_m
        #    
        #   Solve => p_m = (Y - c_a) / [alpha(a_m) - c_a]
        #           a_s = 1 - p_m
        #
        #   Feasibility: 0 <= p_m <= 1, 0 <= a_s <= 1
        #   objective = p_m [1 - F_Q1(a_m)] + (1 - p_m) = 1 - p_m * F_Q1(a_m)
        # ------------------------------------------------
        pm_c3 = None
        as_c3 = None
        obj_c3 = -9999.0
        feasible_c3 = False

        numerator   = (Y - c_a)
        denominator = (monitor_and_audit_cost(a_m) - c_a)

        if abs(denominator) > 1e-14:
            pm_temp = numerator / denominator
            as_temp = 1.0 - pm_temp
            # Check feasibility
            if (0.0 <= pm_temp <= 1.0) and (0.0 <= as_temp <= 1.0):
                feasible_c3 = True
                pm_c3 = pm_temp
                as_c3 = as_temp
                obj_c3 = 1 - pm_c3 + pm_c3*np.mean(attack_scores >= a_m) 
                #equivalent to p_m * [1 - F_Q1(a_m)] + (1 - p_m)
        
        # ------------------------------------------------
        # Pick the best corner among feasible
        # ------------------------------------------------
        corner_objs = [obj_c1, obj_c2]
        corner_sols: list[Any] = [
            (a_m, 0.0, a_s_corner1),  # corner 1
            (a_m, pm_c2, 0.0)         # corner 2
        ]
        
        if feasible_c3:
            corner_objs.append(obj_c3)
            corner_sols.append((a_m, pm_c3, as_c3))
        
        # Best corner for this particular a_m
        idx_best = max(range(len(corner_objs)), key=lambda i: corner_objs[i])
        if corner_objs[idx_best] > best_objective:
            best_objective = corner_objs[idx_best]
            (best_a_m, best_p_m, best_a_s) = corner_sols[idx_best]

    return best_objective, best_a_m, best_p_m, best_a_s


def evaluate_1mon_ood(
        attack_scores_ood: np.ndarray,
        a_m: float,
        p_m: float,
        a_s: float,
) -> float:
    """Gets the catch rate on out of distribution dataset."""
    return a_s + p_m*np.mean(attack_scores_ood >= a_m) #type: ignore


# -------------------------------------------------------------------
# Example usage:
# %%
if __name__ == "__main__":
    import pandas as pd
    project_dir = Path(__file__).parent.parent
    honest_scores = pd.read_csv(project_dir / 'single_monitor' / 'scores_for_debugging_honest.csv')['honest_scores'].to_numpy()
    attack_scores = pd.read_csv(project_dir / 'single_monitor' / 'scores_for_debugging_attack.csv')['attack_scores'].to_numpy()
    c_a = 10000
    c_m = 1
    Y = 221
    best_obj, a_m_star, p_m_star, a_s_star = solve_1mon_optimal_policy(
        c_a=c_a, c_m=c_m, Y=Y, 
        attack_scores=attack_scores, 
        honest_scores=honest_scores
    )
    a_m_naive = get_naive_single_monitor_policy(honest_scores, Y, c_a, c_m)
    print(f"Optimized best_obj: {best_obj:.5f}, Naive best_obj: {evaluate_1mon_ood(attack_scores_ood = attack_scores, a_m = a_m_naive, p_m = 1, a_s = 0):.5f}")

"""
if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)
    honest_scores = np.random.beta(1, 2, size=1000)  # Skewed toward lower values
    attack_scores = np.random.beta(2, 1, size=1000)  # Skewed toward higher values
    
    best_obj, a_m_star, p_m_star, a_s_star = solve_1mon_optimal_policy(
        c_a=20, c_m=1, Y=2, 
        attack_scores=attack_scores, 
        honest_scores=honest_scores
    )
    
    print("Best Objective :", best_obj)
    print("Best a_m       :", a_m_star)
    print("Best p_m       :", p_m_star)
    print("Best a_s       :", a_s_star)


"""
# %%
