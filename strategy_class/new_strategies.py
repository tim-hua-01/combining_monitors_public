#%%
from .strategy_base import Strategy, sweep_solve, get_x_with_fin_audit_thresh, p_a_given_not_q_twomonitors, p_a_given_q_twomonitors, estimate_local_ratio,fit_quadratic
from single_monitor import solve_1mon_optimal_policy
from sklearn.neighbors import KernelDensity # type: ignore
from typing import Optional, Any, Callable, cast, Tuple
import numpy as np
from math import floor, ceil
import json
from warnings import warn
from einops import rearrange
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class NeymanPearsonSingle(Strategy):
    """
    Neyman-Pearson single monitor strategy which uses a monitor 100% of the time and
    spend the rest of the budget on auditing.
    """

    def __init__(self, alpha: bool = True, bandwidth: float | str= "silverman") -> None:
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.attack_kde = KernelDensity(bandwidth=bandwidth)
        self.benign_kde = KernelDensity(bandwidth=bandwidth)
        self.fitted = False
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob = 0
        self.audit_thresh: Optional[float] = None
        self.benign_ln_likelihood_ratios: Optional[np.ndarray] = None

    def fit_method(self, in_sample_alpha_benign: np.ndarray, in_sample_alpha_attack: np.ndarray, in_sample_beta_benign: np.ndarray, in_sample_beta_attack: np.ndarray, Y: float, c_m: float, c_n: float, c_a: float, **kwargs: Any) -> None:
        if self.alpha:
            self.attack_data = in_sample_alpha_attack
            self.benign_data = in_sample_alpha_benign
            self.monitor_cost = c_m
        else:
            self.attack_data = in_sample_beta_attack
            self.benign_data = in_sample_beta_benign
            self.monitor_cost = c_n
        self.attack_kde.fit(self.attack_data.reshape(-1, 1))
        self.benign_kde.fit(self.benign_data.reshape(-1, 1))
        self.fitted = True
        benign_ln_likelihood_ratios = self.attack_kde.score_samples(self.benign_data.reshape(-1, 1)) - self.benign_kde.score_samples(self.benign_data.reshape(-1, 1))
        if Y - self.monitor_cost < 0: #useless
            self.in_sample_safety = 0
            self.audit_thresh = np.inf# 2*np.max(benign_ln_likelihood_ratios)
            self.no_money = True
            self.audit_fpr = 0
            self.audit_budget_share = 0
            self.use_two_monitor_prob = 0
            return
        else:
            self.no_money = False
        #Get the likelihood audit threshold. 
        
        self.benign_ln_likelihood_ratios = benign_ln_likelihood_ratios
        self.audit_fpr = (Y - self.monitor_cost)/c_a
        self.audit_budget_share = self.audit_fpr * c_a / Y
        n_above = floor(len(self.benign_data)*self.audit_fpr)
        if n_above >= len(self.benign_data):
            self.audit_thresh = np.min(benign_ln_likelihood_ratios)
        elif n_above == 0:
            self.audit_thresh = np.max(benign_ln_likelihood_ratios + 1e-6)
        else:
            self.audit_thresh = np.sort(benign_ln_likelihood_ratios)[::-1][n_above - 1]
        attack_ln_likelihood_ratios = self.attack_kde.score_samples(self.attack_data.reshape(-1, 1)) - self.benign_kde.score_samples(self.attack_data.reshape(-1, 1))
        self.in_sample_safety = np.mean(attack_ln_likelihood_ratios >= self.audit_thresh)
        
    def get_oos_safety(self, oos_alpha_attack: np.ndarray, oos_beta_attack: np.ndarray) -> float:
        if self.no_money:
            return 0.0
        if self.alpha:
            oos_data = oos_alpha_attack
        else:
            oos_data = oos_beta_attack
        attack_ln_likelihood_ratios = self.attack_kde.score_samples(oos_data.reshape(-1, 1)) - self.benign_kde.score_samples(oos_data.reshape(-1, 1))
        return np.mean(attack_ln_likelihood_ratios >= self.audit_thresh)
    
    def get_use_two_monitor_prob(self) -> float:
        return 0.0

    def save_parameters(self) -> str:
        return json.dumps({
            "audit_thresh": self.audit_thresh,
            "audit_fpr": self.audit_fpr,
            "audit_budget_share": self.audit_budget_share,
            "in_sample_safety": self.in_sample_safety,
            "bandwidth_malign": self.attack_kde.bandwidth_,
            "bandwidth_benign": self.benign_kde.bandwidth_,
            "Note": "kernel estimators not saved"
        })
    

    def reset(self) -> None:
        self.fitted = False
        self.attack_kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.benign_kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.audit_thresh = None

    def __repr__(self) -> str:
        if self.audit_thresh is None:
            return f"NeymanPearsonSingle(alpha = {self.alpha}, fitted = {self.fitted})"
        else:
            return f"NeymanPearsonSingle(alpha = {self.alpha}, audit_thresh = {self.audit_thresh}, fitted = {self.fitted})"


class NeymanPearsonSingleRandom(Strategy):
    """
    Neyman-Pearson single monitor strategy which uses a monitor 100% of the time and
    spend the rest of the budget on auditing.
    """

    def __init__(self, alpha: bool = True, bandwidth: float | str= "silverman") -> None:
        self.alpha = alpha
        self.bandwidth = bandwidth
        self.attack_kde = KernelDensity(bandwidth=bandwidth)
        self.benign_kde = KernelDensity(bandwidth=bandwidth)
        self.a_m: Optional[float] = None
        self.p_m: Optional[float] = None
        self.a_s: Optional[float] = None
        self.fitted = False
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob = 0
        self.benign_ln_likelihood_ratios: Optional[np.ndarray] = None

    def fit_method(self, in_sample_alpha_benign: np.ndarray, in_sample_alpha_attack: np.ndarray, in_sample_beta_benign: np.ndarray, in_sample_beta_attack: np.ndarray, Y: float, c_m: float, c_n: float, c_a: float, **kwargs: Any) -> None:
        if self.alpha:
            self.attack_data = in_sample_alpha_attack
            self.benign_data = in_sample_alpha_benign
            self.monitor_cost = c_m
        else:
            self.attack_data = in_sample_beta_attack
            self.benign_data = in_sample_beta_benign
            self.monitor_cost = c_n
        self.attack_kde.fit(self.attack_data.reshape(-1, 1))
        self.benign_kde.fit(self.benign_data.reshape(-1, 1))
        self.no_money = False
        

        self.fitted = True
        attack_ln_likelihood_ratios = self.attack_kde.score_samples(self.attack_data.reshape(-1, 1)) - self.benign_kde.score_samples(self.attack_data.reshape(-1, 1))
        benign_ln_likelihood_ratios = self.attack_kde.score_samples(self.benign_data.reshape(-1, 1)) - self.benign_kde.score_samples(self.benign_data.reshape(-1, 1))

        best_safety, self.a_m, self.p_m, self.a_s = solve_1mon_optimal_policy(
            c_a,
            c_m if self.alpha else c_n,
            Y,
            attack_scores=attack_ln_likelihood_ratios,
            honest_scores=benign_ln_likelihood_ratios,
            n_ptile_points=200,
            **kwargs,
        )
        self.benign_ln_likelihood_ratios = benign_ln_likelihood_ratios
        self.attack_ln_likelihood_ratios = attack_ln_likelihood_ratios

        self.audit_fpr = self.a_s + self.p_m * np.mean(benign_ln_likelihood_ratios >= self.a_m)
        self.audit_budget_share = self.audit_fpr * c_a / Y
        self.use_two_monitor_prob = 0
        self.in_sample_safety = best_safety
        
    def get_oos_safety(self, oos_alpha_attack: np.ndarray, oos_beta_attack: np.ndarray) -> float:
        if self.alpha:
            oos_data = oos_alpha_attack
        else:
            oos_data = oos_beta_attack
        attack_ln_likelihood_ratios = self.attack_kde.score_samples(oos_data.reshape(-1, 1)) - self.benign_kde.score_samples(oos_data.reshape(-1, 1))
        return self.a_s + self.p_m * np.mean(attack_ln_likelihood_ratios >= self.a_m)
    
    def get_use_two_monitor_prob(self) -> float:
        return 0.0

    def save_parameters(self) -> str:
        return json.dumps({
            "a_m": self.a_m,
            "p_m": self.p_m,
            "a_s": self.a_s,
            "bandwidth_malign": self.attack_kde.bandwidth_,
            "bandwidth_benign": self.benign_kde.bandwidth_,
            "Note": "kernel estimators not saved"
        })
    

    def reset(self) -> None:
        self.fitted = False
        self.attack_kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.benign_kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.in_sample_safety = None
        self.a_m = None
        self.p_m = None
        self.a_s = None
        super().reset()

    def __repr__(self) -> str:
        if self.a_m is None:
            return f"NeymanPearsonSingleRandom(alpha = {self.alpha}, fitted = {self.fitted})"
        else:
            return f"NeymanPearsonSingleRandom(alpha = {self.alpha}, a_m = {self.a_m}, p_m = {self.p_m}, a_s = {self.a_s}, fitted = {self.fitted})"



class TwoMonitorAlwaysNP(Strategy):
    def __init__(self, local_odds_ratio_bandwidth: float | str = 'silverman') -> None:
        self.fitted: bool = False
        self.kernel_bandwidth: float | str = local_odds_ratio_bandwidth
        self.audit_threshold: Optional[float] = None
        self.benign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob: Optional[float] = 1.0
    
    def get_lnl_ratio(self, alpha_scores: np.ndarray, beta_scores: np.ndarray):
        return self.malign_kernel.score_samples(
                    np.stack(
                        (alpha_scores, 
                         beta_scores), 
                        axis = 1)
                ) - self.benign_kernel.score_samples(
                    np.stack(
                        (alpha_scores, 
                         beta_scores), 
                        axis = 1)
                )

    def reset(self)-> None:
        self.fitted = False
        self.audit_threshold = None
        self.benign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.in_sample_safety= None
        self.audit_fpr = None
        self.audit_budget_share= None
    
    def fit_method(self,
        in_sample_alpha_benign: np.ndarray,
        in_sample_alpha_attack: np.ndarray,
        in_sample_beta_benign: np.ndarray,
        in_sample_beta_attack: np.ndarray,
        Y: float,
        c_m: float,
        c_n: float,
        c_a: float,
        n_thresholds: int = 100,
        **kwargs: Any) -> None:
        self.benign_kernel.fit(
            np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis = 1)
        )
        self.malign_kernel.fit(
            np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis = 1)
        )
        benign_lnl_ratio  = self.get_lnl_ratio(in_sample_alpha_benign,in_sample_beta_benign)
        if Y - c_m - c_n <=0:
            self.no_money = True
            #warn(f"No mone we have Y = {Y}, c_m = {c_m} and c_n = {c_n}")
            self.in_sample_safety = 0
            self.audit_threshold = np.inf #np.max(benign_lnl_ratio)*2
            self.audit_fpr = 0
            self.audit_budget_share = 0
            self.use_two_monitor_prob = 0
            self.fitted = True
            return
        else:
            self.no_money = False
        n_above = floor(len(in_sample_alpha_benign)*(Y - c_m - c_n)/c_a)   
        if n_above == 0:
            self.audit_threshold = np.max(benign_lnl_ratio + 1e-6)
        else:
            self.audit_threshold = np.sort(benign_lnl_ratio)[::-1][n_above - 1]
        self.audit_fpr = n_above/len(in_sample_alpha_benign)
        self.audit_budget_share = self.audit_fpr*c_a/Y
        #get safety
        malign_lnl_ratio = self.get_lnl_ratio(in_sample_alpha_attack,in_sample_beta_attack)
        n_caught = np.sum(malign_lnl_ratio >= self.audit_threshold)
        self.in_sample_safety = n_caught/len(in_sample_alpha_attack)


        self.fitted = True

    def get_oos_safety(self, oos_alpha_attack: np.ndarray, oos_beta_attack: np.ndarray) -> float:
        if self.no_money:
            return 0.0
        oos_lnl_ratio = self.get_lnl_ratio(oos_alpha_attack,oos_beta_attack)
        n_caught = np.sum(oos_lnl_ratio >= self.audit_threshold)
        return n_caught/len(oos_alpha_attack)

    def save_parameters(self) -> str:
        assert self.fitted
        return json.dumps(
            {
                "audit_thresh": self.audit_threshold,
                "kernel_width_or_setting": self.kernel_bandwidth,
                "bandwidth_malign": self.malign_kernel.bandwidth_,
                "bandwidth_benign": self.benign_kernel.bandwidth_,
                "empirical_fpr": self.audit_fpr
            }
        )

    def load_parameters(self, params: dict[str, Any]) -> None:
        warn("Cannot load parameters for kernels")

    def __repr__(self) -> str:
        if self.fitted:
            return f"TwoMonitorSingleThresh with safety {self.in_sample_safety}. With audit threshold {self.audit_threshold} and audit fpr {self.audit_fpr}"
        else:
            return f"TwoMonitorSingleThresh unfitted with bandwidth {self.kernel_bandwidth}"

        
        

class TwoMonitorSingleThresh(Strategy):
    def __init__(self, local_odds_ratio_bandwidth: float | str = 'silverman') -> None:
        self.fitted: bool = False
        self.N_first: Optional[bool] = None
        self.kernel_bandwidth: float | str = local_odds_ratio_bandwidth
        self.second_monitor_call_threshold: Optional[float] = None
        self.audit_threshold: Optional[float] = None
        self.benign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob: Optional[float] = None
    
    def reset(self)-> None:
        self.fitted = False
        self.N_first = None
        self.second_monitor_call_threshold = None
        self.audit_threshold = None
        self.benign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.in_sample_safety= None
        self.audit_fpr = None
        self.audit_budget_share= None
        self.use_two_monitor_prob= None

    def fit_method(self,
        in_sample_alpha_benign: np.ndarray,
        in_sample_alpha_attack: np.ndarray,
        in_sample_beta_benign: np.ndarray,
        in_sample_beta_attack: np.ndarray,
        Y: float,
        c_m: float,
        c_n: float,
        c_a: float,
        n_thresholds: int = 100,
        **kwargs: Any) -> None:
        # Check which monitor to call first
        # Sweep over second monitor calling thresholds.
        # Pre-fit the kernel density estimatior
        # We can talk about issues about fitting the estimator for only the specific region we're calling the second monitor in the appendix or something. 
        self.benign_kernel.fit(
            np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis = 1)
        )
        self.malign_kernel.fit(
            np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis = 1)
        )
        alpha_best_combo: dict[str, float] = {'safety': 0, 'audit_threshold': 0, 
                                              'second_monitor_thresh': 0, 'empirical_audit_fpr': 0}
        if Y > c_m:
            #call alpha monitor first
            alpha_sweep_threshs = np.concat([
                np.array([0]), np.percentile(in_sample_alpha_benign, np.arange(0,100,100/n_thresholds))
            ])
            for t in alpha_sweep_threshs:
                #calculate money left after calling second monitor. 
                filter_for_benign = in_sample_alpha_benign >= t
                budget = Y - c_m - c_n * np.mean(filter_for_benign)
                if budget <= 0:
                    continue
                audit_fpr = budget/c_a
                
                benign_likelihood_ratios_filt = self.malign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_benign[filter_for_benign], 
                         in_sample_beta_benign[filter_for_benign]), 
                        axis = 1)
                )- self.benign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_benign[filter_for_benign], 
                         in_sample_beta_benign[filter_for_benign]), 
                        axis = 1)
                )
                n_above = floor(len(in_sample_alpha_benign)*audit_fpr) #This is suppose to be the full set
                if n_above > len(benign_likelihood_ratios_filt):
                    audit_thresh = np.min(benign_likelihood_ratios_filt)
                elif n_above == 0:
                    audit_thresh = np.max(benign_likelihood_ratios_filt + 1e-6)
                else:
                    audit_thresh = np.sort(benign_likelihood_ratios_filt)[::-1][n_above - 1]
                
                #evaluate safety
                filter_for_malign = in_sample_alpha_attack >= t
                malign_likelihood_ratios_filt = self.malign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_attack[filter_for_malign], 
                         in_sample_beta_attack[filter_for_malign]), 
                        axis = 1)
                ) - self.benign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_attack[filter_for_malign], 
                         in_sample_beta_attack[filter_for_malign]), 
                        axis = 1)
                )
                n_caught = np.sum(malign_likelihood_ratios_filt >= audit_thresh)
                safety = n_caught/len(in_sample_alpha_attack)
                if safety > alpha_best_combo['safety']:
                    alpha_best_combo = {
                        'safety': safety,
                        'audit_threshold': audit_thresh,
                        'second_monitor_thresh': t,
                        'empirical_audit_fpr': n_above/len(in_sample_beta_benign)
                    }

        beta_best_combo: dict[str, float] = {'safety': 0, 'audit_threshold': 0, 
                                              'second_monitor_thresh': 0, 'empirical_audit_fpr': 0}
        if Y <= c_n and Y <= c_m:
            warn("Can't afford to use either monitors")
        
        if Y > c_n:
            #call beta monitor first
            beta_sweep_threshs = np.concat([
                np.array([0]), np.percentile(in_sample_beta_benign, np.arange(0,100,100/n_thresholds))
            ])
            for t in beta_sweep_threshs:
                #calculate money left after calling second monitor. 
                filter_for_benign = in_sample_beta_benign >= t
                budget = Y - c_n - c_m * np.mean(filter_for_benign)
                if budget <= 0:
                    continue
                audit_fpr = budget/c_a
                
                benign_likelihood_ratios_filt = self.malign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_benign[filter_for_benign], 
                         in_sample_beta_benign[filter_for_benign]), 
                        axis = 1)
                )- self.benign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_benign[filter_for_benign], 
                         in_sample_beta_benign[filter_for_benign]), 
                        axis = 1)
                )
                n_above = floor(len(in_sample_alpha_benign)*audit_fpr)
                if n_above > len(benign_likelihood_ratios_filt):
                    audit_thresh = np.min(benign_likelihood_ratios_filt)
                elif n_above == 0:
                    audit_thresh = np.max(benign_likelihood_ratios_filt + 1e-6)
                else:
                    audit_thresh = np.sort(benign_likelihood_ratios_filt)[::-1][n_above - 1]
                
                #evaluate safety
                filter_for_malign = in_sample_beta_attack >= t
                malign_likelihood_ratios_filt = self.malign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_attack[filter_for_malign], 
                         in_sample_beta_attack[filter_for_malign]), 
                        axis = 1)
                ) - self.benign_kernel.score_samples(
                    np.stack(
                        (in_sample_alpha_attack[filter_for_malign], 
                         in_sample_beta_attack[filter_for_malign]), 
                        axis = 1)
                )
                n_caught = np.sum(malign_likelihood_ratios_filt >= audit_thresh)
                safety = n_caught/len(in_sample_beta_attack)
                if safety > beta_best_combo['safety']:
                    beta_best_combo = {
                        'safety': safety,
                        'audit_threshold': audit_thresh,
                        'second_monitor_thresh': t,
                        'empirical_audit_fpr': n_above/len(in_sample_beta_benign)
                    }

        if beta_best_combo['safety'] >= alpha_best_combo['safety']:
            best_overall = beta_best_combo
            self.use_two_monitor_prob = np.mean(in_sample_beta_benign >= beta_best_combo['second_monitor_thresh']) # type: ignore
            self.N_first = True
        else:
            best_overall =  alpha_best_combo
            self.N_first = False
            self.use_two_monitor_prob = np.mean(in_sample_alpha_benign >= alpha_best_combo['second_monitor_thresh']) # type: ignore
        self.audit_fpr = best_overall['empirical_audit_fpr']
        self.audit_budget_share = self.audit_fpr*c_a/Y
        self.in_sample_safety = best_overall['safety']
        self.audit_threshold = best_overall['audit_threshold']
        self.second_monitor_call_threshold = best_overall['second_monitor_thresh']

        self.fitted = True

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert self.fitted
        if self.N_first:
            second_stage_filter = oos_beta_attack >= self.second_monitor_call_threshold
        else:
            second_stage_filter = oos_alpha_attack >= self.second_monitor_call_threshold
        log_likelihood_ratio_oos = self.malign_kernel.score_samples(
        np.stack(
            (oos_alpha_attack[second_stage_filter], 
                oos_beta_attack[second_stage_filter]), 
            axis = 1)
        ) - self.benign_kernel.score_samples(
        np.stack(
            (oos_alpha_attack[second_stage_filter], 
                oos_beta_attack[second_stage_filter]), 
            axis = 1)
        )
        n_caught = np.sum(log_likelihood_ratio_oos >= self.audit_threshold)
        return n_caught/len(oos_alpha_attack)

    def save_parameters(self) -> str:
        assert self.fitted
        return json.dumps(
            {   "N_first": self.N_first,
                "second_monitor_call_threshold": self.second_monitor_call_threshold,
                "audit_thresh": self.audit_threshold,
                "kernel_width_or_setting": self.kernel_bandwidth,
                "bandwidth_malign": self.malign_kernel.bandwidth_,
                "bandwidth_benign": self.benign_kernel.bandwidth_,
                "empirical_fpr": self.audit_fpr
            }
        )

    def load_parameters(self, params: dict[str, Any]) -> None:
        warn("Cannot load parameters for kernels")
        # The cache will be rebuilt when fit_method is called

    def __repr__(self) -> str:
        if self.fitted:
            return f"TwoMonitorSingleThresh with safety {self.in_sample_safety}. Using {'beta' if self.N_first else 'alpha'} first, defer if above {self.second_monitor_call_threshold}, and audit fpr {self.audit_fpr}"
        else:
            return f"TwoMonitorSingleThresh unfitted with bandwidth {self.kernel_bandwidth}"


class AuditEndNP(Strategy):
    def __init__(self, verbose: bool = False, local_odds_ratio_bandwidth: float | str = 'silverman') -> None:
        self.fitted: bool = False
        self.verbose = verbose
        self.x: Optional[np.ndarray | list[float]] = None
        self.N_first: Optional[bool] = None
        self.kernel_bandwidth: float | str = local_odds_ratio_bandwidth
        self.benign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=local_odds_ratio_bandwidth)
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob: Optional[float] = None
        self.lnl_ratio_cache: dict[tuple[float, float], float] = {}
        
    
    def reset(self)-> None:
        self.fitted = False
        self.N_first = None
        self.second_monitor_call_threshold = None
        self.audit_threshold = None
        self.benign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.in_sample_safety= None
        self.audit_fpr = None
        self.audit_budget_share= None
        self.use_two_monitor_prob= None
        self.lnl_ratio_cache = {}

    def get_lnl_ratio(self, alpha_scores: np.ndarray | float | list[float], beta_scores: np.ndarray | float| list[float]) -> float | np.ndarray:
        assert type(alpha_scores) == type(beta_scores)
        try:
            if type(alpha_scores) == float or type(alpha_scores) == np.float64:
                raise NotImplementedError("Single score pair not implemented")
                # Single score pair - check cache first
                key = (float(alpha_scores), float(beta_scores))
                if key in self.lnl_ratio_cache:
                    return self.lnl_ratio_cache[key]
                # Not in cache, compute and store
                result = (
                    self.malign_kernel.score_samples(
                        np.array([[alpha_scores, beta_scores]])) - 
                        self.benign_kernel.score_samples(np.array([[alpha_scores, beta_scores]]))
                ).item()
                self.lnl_ratio_cache[key] = result
                return result
            elif type(alpha_scores) == list:
                alpha_scores = np.array(alpha_scores)
                beta_scores = np.array(beta_scores)
                
            # Convert to arrays of scores
            results = np.zeros(len(alpha_scores))
            missing_indices = []
            
            # Check cache for each pair
            for i in range(len(alpha_scores)):
                key = (float(alpha_scores[i]), float(beta_scores[i]))
                if key in self.lnl_ratio_cache:
                    results[i] = self.lnl_ratio_cache[key]
                else:
                    missing_indices.append(i)
            
            # Calculate missing values if needed
            if missing_indices:
                missing_alpha = alpha_scores[missing_indices]
                missing_beta = beta_scores[missing_indices]
                stacked_scores = np.stack((missing_alpha, missing_beta), axis=1)
                
                missing_results = self.malign_kernel.score_samples(stacked_scores) - self.benign_kernel.score_samples(stacked_scores)
                
                # Update cache and results array
                for idx, orig_idx in enumerate(missing_indices):
                    key = (float(alpha_scores[orig_idx]), float(beta_scores[orig_idx]))
                    self.lnl_ratio_cache[key] = missing_results[idx]
                    results[orig_idx] = missing_results[idx]
            
            return results
        except:
            import pdb
            pdb.set_trace()
            raise AssertionError

    def fit_method(self,
        in_sample_alpha_benign: np.ndarray,
        in_sample_alpha_attack: np.ndarray,
        in_sample_beta_benign: np.ndarray,
        in_sample_beta_attack: np.ndarray,
        Y: float,
        c_m: float,
        c_n: float,
        c_a: float,
        n_thresholds: int = 100,
        **kwargs: Any) -> None:
        # Check which monitor to call first
        # Sweep over second monitor calling thresholds.
        # Pre-fit the kernel density estimator
        # We can talk about issues about fitting the estimator for only the specific region we're calling the second monitor in the appendix or something.
        assert len(in_sample_alpha_attack) == len(in_sample_beta_attack)
        assert len(in_sample_alpha_benign) == len(in_sample_beta_benign) 
        self.benign_kernel.fit(
            np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis = 1)
        )
        self.malign_kernel.fit(
            np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis = 1)
        )
        
        # Clear cache when fitting new data
        self.lnl_ratio_cache = {}
        
        # Precompute all likelihood ratios for benign and attack samples
        benign_pairs = np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis=1)
        attack_pairs = np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis=1)
        
        benign_lnl = self.malign_kernel.score_samples(benign_pairs) - self.benign_kernel.score_samples(benign_pairs)
        attack_lnl = self.malign_kernel.score_samples(attack_pairs) - self.benign_kernel.score_samples(attack_pairs)
        
        # Store in cache with collision detection
        for i in range(len(in_sample_alpha_benign)):
            key = (float(in_sample_alpha_benign[i]), float(in_sample_beta_benign[i]))
            if key in self.lnl_ratio_cache:
                raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = benign_lnl[i]

        for i in range(len(in_sample_alpha_attack)):
            key = (float(in_sample_alpha_attack[i]), float(in_sample_beta_attack[i]))
            if key in self.lnl_ratio_cache:
                raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = attack_lnl[i]
            
        self.Y = Y
        self.c_m = c_m
        self.c_n = c_n
        self.c_a = c_a
        # Rest of the fit_method remains the same
        if Y - c_n > 0:
            sweep_results_N_first = sweep_solve(
                alpha_benign_scores=in_sample_alpha_benign,
                alpha_attack_scores=in_sample_alpha_attack,
                beta_benign_scores=in_sample_beta_benign,
                beta_attack_scores=in_sample_beta_attack,
                Y=Y,
                c_m=c_m,
                c_n=c_n,
                c_a=c_a,
                f_mn=self.get_lnl_ratio,
                N_first=True,
                verbose=self.verbose,
                **kwargs,
            )
            best_n_result_idx = np.argmax([res["safety"] for res in sweep_results_N_first])
            best_n_result = sweep_results_N_first[best_n_result_idx]

        if Y - c_m > 0:
            sweep_results_M_first = sweep_solve(
                alpha_benign_scores=in_sample_alpha_benign,
                alpha_attack_scores=in_sample_alpha_attack,
                beta_benign_scores=in_sample_beta_benign,
                beta_attack_scores=in_sample_beta_attack,
                Y=Y,
                c_m=c_m,
                c_n=c_n,
                c_a=c_a,
                f_mn=self.get_lnl_ratio,
                N_first=False,
                verbose=self.verbose,
                **kwargs,
            )
            best_m_result_idx = np.argmax([res["safety"] for res in sweep_results_M_first])
            best_m_result = sweep_results_M_first[best_m_result_idx]
        else:
            best_m_result = None
        
        if best_m_result:
            if best_m_result['safety'] > best_n_result['safety']:
                best_result = best_m_result
                self.N_first = False
            else:
                best_result = best_n_result
                self.N_first = True
        else:
            best_result = best_n_result
            self.N_first = True
        self.in_sample_safety = best_result["safety"]
        self.audit_thresh = best_result["audit_threshold"]
        self.end_thresh = best_result["end_threshold"]
        self.x_vec = get_x_with_fin_audit_thresh(
            x=np.array(
                [
                    0,
                    self.audit_thresh,
                    self.end_thresh,
                    self.audit_thresh,
                    self.end_thresh,
                    0,
                    0,
                ]
            )
            if self.N_first
            else np.array(
                [
                    1,
                    self.audit_thresh,
                    self.end_thresh,
                    self.audit_thresh,
                    self.end_thresh,
                    0,
                    0,
                ]
            ),
            alpha_benign_scores=in_sample_alpha_benign,
            beta_benign_scores=in_sample_beta_benign,
            f_mn=self.get_lnl_ratio,
            c_a=c_a,
            c_m=c_m,
            c_n=c_n,
            Y=Y,
            N_first=self.N_first,
        )
        self.audit_fpr = p_a_given_not_q_twomonitors(
            x=self.x_vec,
            alpha_benign=in_sample_alpha_benign,
            beta_benign=in_sample_beta_benign,
            f_mn=self.get_lnl_ratio,
        )
        self.audit_budget_share = self.audit_fpr * c_a / Y
        if self.N_first:
            self.use_two_monitor_prob = np.mean(
                (in_sample_beta_benign < self.audit_thresh)
                & (in_sample_beta_benign >= self.end_thresh)
            )  # type: ignore
        else:
            self.use_two_monitor_prob = np.mean(
                (in_sample_alpha_benign < self.audit_thresh)
                & (in_sample_alpha_benign >= self.end_thresh)
            )  # type: ignore
        self.fitted = True

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert (
            self.x_vec is not None
            and self.audit_thresh is not None
            and self.end_thresh is not None
        ), "Parameters not computed"
        return p_a_given_q_twomonitors(
            x=self.x_vec,
            alpha_attack_scores=oos_alpha_attack,
            beta_attack_scores=oos_beta_attack,
            f_mn=self.get_lnl_ratio,
        )

    def save_parameters(self) -> str:
        assert self.fitted
        return json.dumps(
            {   "N_first": self.N_first,
                "x": self.x_vec if isinstance(self.x_vec, list) else self.x_vec.tolist(),
                "kernel_width_or_setting": self.kernel_bandwidth,
                "bandwidth_malign": self.malign_kernel.bandwidth_,
                "bandwidth_benign": self.benign_kernel.bandwidth_,
                "cache_size": len(self.lnl_ratio_cache)
            }
        )

    def load_parameters(self, params: dict[str, Any]) -> None:
        warn("Cannot load parameters for kernels")
        # The cache will be rebuilt when fit_method is called

    def __repr__(self) -> str:
        if self.end_thresh is None:
            return f"AuditEndQuadNandM(fitted = {self.fitted}, N_first = {self.N_first})"
        else:
            return f"AuditEndQuadNandM(x = {self.x_vec}, f_mn_coefs = {self.f_mn_coefs}, N_first = {self.N_first})"

    def reset(self) -> None:
        # Keep N_first and local_odds_ratio_bandwidth but reset everything else
        self.fitted = False
        self.audit_thresh = None
        self.end_thresh = None
        self.x_vec = None
        self.f_mn = None
        self.f_mn_coefs = None
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.use_two_monitor_prob = None
        self.benign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)
        self.malign_kernel = KernelDensity(bandwidth=self.kernel_bandwidth)


class FullNPOneRegion(Strategy):
    def __init__(self, bandwidth_2d: str | float = 'silverman', bandwidth_1d : str | float = 'silverman'):
        self.fitted = False
        super().__init__()
        self.bandwidth_1d = bandwidth_1d
        self.bandwidth_2d = bandwidth_2d
        self.benign_2d_ker = KernelDensity(bandwidth=bandwidth_2d)
        self.malign_2d_ker = KernelDensity(bandwidth=bandwidth_2d)
        self.alpha_benign_ker = KernelDensity(bandwidth=bandwidth_1d)
        self.beta_benign_ker = KernelDensity(bandwidth=bandwidth_1d)
        self.alpha_attack_ker = KernelDensity(bandwidth=bandwidth_1d)
        self.beta_attack_ker = KernelDensity(bandwidth=bandwidth_1d)
        self.N_first: Optional[bool] = None
        self.top_threshold: Optional[float] = None
        self.bot_threshold: Optional[float] = None
        self.lnl_ratio_cache: dict[Tuple[float | None, float | None], float] = {}
        

    def reset(self):
        self.fitted = False
        super().reset()
        self.benign_2d_ker = KernelDensity(bandwidth=self.bandwidth_2d)
        self.malign_2d_ker = KernelDensity(bandwidth=self.bandwidth_2d)
        self.alpha_benign_ker = KernelDensity(bandwidth=self.bandwidth_1d)
        self.beta_benign_ker = KernelDensity(bandwidth=self.bandwidth_1d)
        self.alpha_attack_ker = KernelDensity(bandwidth=self.bandwidth_1d)
        self.beta_attack_ker = KernelDensity(bandwidth=self.bandwidth_1d)
        self.N_first = None
        self.top_threshold = None
        self.bot_threshold = None
        self.lnl_ratio_cache = {}

    

    def get_lnl_ratio(self, alpha_scores: list[float] | np.ndarray | None = None, beta_scores: list[float] | np.ndarray | None = None) -> np.ndarray:
        # This one you can pull the ratio for one dimension or two dimensions
        if alpha_scores is not None and beta_scores is not None:
            if type(alpha_scores) == list:
                alpha_scores = np.array(alpha_scores)
                beta_scores = np.array(beta_scores)
            else:
                assert type(alpha_scores) == np.ndarray
                assert type(beta_scores) == np.ndarray
            assert len(alpha_scores) == len(beta_scores)
            assert len(alpha_scores) > 0
            results = np.zeros(len(alpha_scores))
            missing_indices = []
            
            # Check cache for each pair
            for i in range(len(alpha_scores)):
                key: Tuple[float | None, float | None] = (float(alpha_scores[i]), float(beta_scores[i]))
                if key in self.lnl_ratio_cache:
                    results[i] = self.lnl_ratio_cache[key]
                else:
                    missing_indices.append(i)
            
            # Calculate missing values if needed
            if missing_indices:
                missing_alpha = alpha_scores[missing_indices]
                missing_beta = beta_scores[missing_indices]
                stacked_scores = np.stack((missing_alpha, missing_beta), axis=1)
                
                missing_results = self.malign_2d_ker.score_samples(stacked_scores) - self.benign_2d_ker.score_samples(stacked_scores)
                
                # Update cache and results array
                for idx, orig_idx in enumerate(missing_indices):
                    key = (float(alpha_scores[orig_idx]), float(beta_scores[orig_idx]))
                    self.lnl_ratio_cache[key] = missing_results[idx]
                    results[orig_idx] = missing_results[idx]

        elif alpha_scores is not None:
            if type(alpha_scores) == list:
                alpha_scores = np.array(alpha_scores)
            else:
                assert type(alpha_scores) == np.ndarray
            assert len(alpha_scores) > 0
            results = np.zeros(len(alpha_scores))
            missing_indices = []

            #now check cache for only alpha
            for i in range(len(alpha_scores)):
                key = (float(alpha_scores[i]), None)
                if key in self.lnl_ratio_cache:
                    results[i] = self.lnl_ratio_cache[key]
                else:
                    missing_indices.append(i)
            
            if missing_indices:
                missing_alpha = alpha_scores[missing_indices]
                missing_results = self.alpha_attack_ker.score_samples(rearrange(missing_alpha, 'n -> n 1')) - self.alpha_benign_ker.score_samples(rearrange(missing_alpha, 'n -> n 1'))
                for idx, orig_idx in enumerate(missing_indices):
                    key = (float(alpha_scores[orig_idx]), None)
                    self.lnl_ratio_cache[key] = missing_results[idx]
                    results[orig_idx] = missing_results[idx]
        elif beta_scores is not None:
            if type(beta_scores) == list:
                beta_scores = np.array(beta_scores)
            else:
                assert type(beta_scores) == np.ndarray
            assert len(beta_scores) > 0
            results = np.zeros(len(beta_scores))
            missing_indices = []

            #now check cache for only beta
            for i in range(len(beta_scores)):
                key = (None, float(beta_scores[i]))
                if key in self.lnl_ratio_cache:
                    results[i] = self.lnl_ratio_cache[key]
                else:
                    missing_indices.append(i)

            if missing_indices:
                missing_beta = beta_scores[missing_indices]
                missing_results = self.beta_attack_ker.score_samples(rearrange(missing_beta, 'n -> n 1')) - self.beta_benign_ker.score_samples(rearrange(missing_beta, 'n -> n 1'))
                for idx, orig_idx in enumerate(missing_indices):
                    key = (None, float(beta_scores[orig_idx]))
                    self.lnl_ratio_cache[key] = missing_results[idx]
                    results[orig_idx] = missing_results[idx]
        else:
            raise ValueError("No scores provided")
        return results

    @staticmethod
    def get_bot_given_top(top_threshold: float, first_benign_scores: np.ndarray, Y: float, c_m: float, c_n: float, c_a: float, N_first: bool) -> float:
        """
        Assuming N first, here's the math
        Y = c_n + c_m*(np.mean(first_benign_scores < top_threshold) - np.mean(first_benign_scores < bot_threshold))
        (Y - c_n)/c_m = np.mean(first_benign_scores < top_threshold) - np.mean(first_benign_scores < bot_threshold)
        np.mean(first_benign_scores < bot_threshold) = np.mean(first_benign_scores < top_threshold) - (Y - c_n)/c_m

        This last term is probably commonly negative right? Because you need to call the second monitor way more than 100% of the time 
        to exhaust the budget
        """
        if N_first:
            p_below_end = max(0, np.mean(first_benign_scores < top_threshold) - (Y - c_n)/c_m) #type: ignore
        else:
            p_below_end = max(0, np.mean(first_benign_scores < top_threshold) - (Y - c_m)/c_n) #type: ignore
        sorted_b = np.sort(first_benign_scores)
        n_below_end = ceil(p_below_end*len(first_benign_scores))
        return sorted_b[n_below_end]



    def fit_method(self, in_sample_alpha_benign: np.ndarray, in_sample_alpha_attack: np.ndarray, 
                   in_sample_beta_benign: np.ndarray, in_sample_beta_attack: np.ndarray, 
                   Y: float, c_m: float, c_n: float, c_a: float, 
                   n_thresholds: int = 200, verbose: bool = False,
                   **kwargs: Any) -> None:
        assert len(in_sample_alpha_attack) == len(in_sample_beta_attack)
        assert len(in_sample_alpha_benign) == len(in_sample_beta_benign) 
        if verbose:
            print("Fitting kernels")
        self.benign_2d_ker.fit(
            np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis = 1)
        )
        self.malign_2d_ker.fit(
            np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis = 1)
        )
        self.alpha_benign_ker.fit(rearrange(in_sample_alpha_benign, 'n -> n 1'))
        self.beta_benign_ker.fit(rearrange(in_sample_beta_benign, 'n -> n 1'))
        self.alpha_attack_ker.fit(rearrange(in_sample_alpha_attack, 'n -> n 1'))
        self.beta_attack_ker.fit(rearrange(in_sample_beta_attack, 'n -> n 1'))


        # Clear cache when fitting new data
        self.lnl_ratio_cache = {}
        # Precompute all likelihood ratios for benign and attack samples
        if verbose:
            print("Precomputing likelihood ratios")
        benign_pairs = np.stack((in_sample_alpha_benign, in_sample_beta_benign), axis=1)
        attack_pairs = np.stack((in_sample_alpha_attack, in_sample_beta_attack), axis=1)
        
        benign_lnl = self.malign_2d_ker.score_samples(benign_pairs) - self.benign_2d_ker.score_samples(benign_pairs)
        attack_lnl = self.malign_2d_ker.score_samples(attack_pairs) - self.benign_2d_ker.score_samples(attack_pairs)
        
        # Store in cache with collision detection
        for i in range(len(in_sample_alpha_benign)):
            key: Tuple[float | None, float | None] = (float(in_sample_alpha_benign[i]), float(in_sample_beta_benign[i]))
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = benign_lnl[i]

        for i in range(len(in_sample_alpha_attack)):
            key = (float(in_sample_alpha_attack[i]), float(in_sample_beta_attack[i]))
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = attack_lnl[i]

        #Precompute 1d likelihood ratios for benign and attack samples
        benign_alpha_1d_lnl = self.alpha_attack_ker.score_samples(rearrange(in_sample_alpha_benign, 'n -> n 1')) - self.alpha_benign_ker.score_samples(rearrange(in_sample_alpha_benign, 'n -> n 1'))
        benign_beta_1d_lnl = self.beta_attack_ker.score_samples(rearrange(in_sample_beta_benign, 'n -> n 1')) - self.beta_benign_ker.score_samples(rearrange(in_sample_beta_benign, 'n -> n 1'))
        attack_alpha_1d_lnl = self.alpha_attack_ker.score_samples(rearrange(in_sample_alpha_attack, 'n -> n 1')) - self.alpha_benign_ker.score_samples(rearrange(in_sample_alpha_attack, 'n -> n 1'))
        attack_beta_1d_lnl = self.beta_attack_ker.score_samples(rearrange(in_sample_beta_attack, 'n -> n 1')) - self.beta_benign_ker.score_samples(rearrange(in_sample_beta_attack, 'n -> n 1'))

        #Store in cache with collision detection
        for i in range(len(in_sample_alpha_benign)):
            key = (float(in_sample_alpha_benign[i]), None)
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = benign_alpha_1d_lnl[i]

        for i in range(len(in_sample_beta_benign)):
            key = (None, float(in_sample_beta_benign[i]))
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = benign_beta_1d_lnl[i]

        for i in range(len(in_sample_alpha_attack)):
            key = (float(in_sample_alpha_attack[i]), None)
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = attack_alpha_1d_lnl[i]

        for i in range(len(in_sample_beta_attack)):
            key = (None, float(in_sample_beta_attack[i]))
            if key in self.lnl_ratio_cache:
                continue
                #raise ValueError(f"Key collision detected: {key}")
            self.lnl_ratio_cache[key] = attack_beta_1d_lnl[i]
        
        self.fitted = True

        if verbose:
            print("Start with sweep")
        #Now we sweep
        all_results: list[tuple[float, float, float, float, bool, float]] = [] #(top, bot, safety, audit_cutoff, N_first, audit_fpr)
        if Y - c_n > 0:
            #Use n first.
            thresholds_to_check = np.quantile(in_sample_beta_benign, np.linspace(0, 1, n_thresholds))
            #show tqdm if verbose
            for top in tqdm(thresholds_to_check, desc = "Top thresholds", disable = not verbose):
                lowerest_bot = self.get_bot_given_top(top, in_sample_beta_benign, Y, c_m, c_n, c_a, N_first = True)
                for bot in thresholds_to_check:
                    if bot < lowerest_bot or bot > top:
                        continue
                    else:
                        #Check budget remaining
                        budget_remaining = Y - c_n - c_m*(np.mean(in_sample_beta_benign < top) - np.mean(in_sample_beta_benign < bot))
                        n_benign_audited = floor(budget_remaining/c_a*len(in_sample_beta_benign))
                        audit_fpr = n_benign_audited/len(in_sample_beta_benign)
                        assert n_benign_audited >= 0, "Negative audit budget"
                        #get benign lnratios in interval
                        benign_in_interval_filter = (in_sample_beta_benign < top) & (in_sample_beta_benign >= bot)
                        if benign_in_interval_filter.any():
                            benign_ln_in_interval = self.get_lnl_ratio(in_sample_alpha_benign[benign_in_interval_filter], in_sample_beta_benign[benign_in_interval_filter])
                        else:
                            benign_ln_in_interval = np.empty(0)
                        #now get benign lnratios out of interval
                        benign_out_interval_filter = ~benign_in_interval_filter
                        if benign_out_interval_filter.any():
                            benign_out_interval_lnl = self.get_lnl_ratio(None, in_sample_beta_benign[benign_out_interval_filter])
                        else:
                            benign_out_interval_lnl = np.empty(0)

                        benign_lnratio_all_soreted = np.sort(np.concatenate((benign_ln_in_interval, benign_out_interval_lnl)))
                        assert len(benign_lnratio_all_soreted) == len(in_sample_alpha_benign)
                        if n_benign_audited > 0:
                            audit_cutoff = benign_lnratio_all_soreted[-n_benign_audited]
                        else:
                            audit_cutoff = benign_lnratio_all_soreted[-1] + 1e-6
                        
                        #check cost:
                        assert Y - c_n - c_m*(np.mean(benign_in_interval_filter)) - c_a*np.mean(benign_lnratio_all_soreted >= audit_cutoff) + 1e-10 >= 0, "Budget blown"

                        #Check safety on attack ln ratios
                        malign_in_interval_filter = (in_sample_beta_attack >= bot) & (in_sample_beta_attack < top)
                        if np.sum(malign_in_interval_filter > 0):
                            malign_ln_in_interval = self.get_lnl_ratio(in_sample_alpha_attack[malign_in_interval_filter], in_sample_beta_attack[malign_in_interval_filter])
                        else:
                            malign_ln_in_interval = np.empty(0)
                        malign_out_interval_filter = ~malign_in_interval_filter
                        malign_out_interval_lnl = self.get_lnl_ratio(None, in_sample_beta_attack[malign_out_interval_filter])

                        malign_lnratio_all_sorted = np.sort(np.concatenate((malign_ln_in_interval, malign_out_interval_lnl)))
                        assert len(malign_lnratio_all_sorted) == len(in_sample_alpha_attack)
                        n_caught = np.sum(malign_lnratio_all_sorted >= audit_cutoff)
                        safety = n_caught/len(in_sample_alpha_attack)
                        all_results.append((top, bot, safety, audit_cutoff, True, audit_fpr))
        
        if Y - c_m > 0:
            #Use m first
            thresholds_to_check = np.quantile(in_sample_alpha_benign, np.linspace(0, 1, n_thresholds))
            for top in tqdm(thresholds_to_check, desc = "Top thresholds", disable = not verbose):
                lowest_bot = self.get_bot_given_top(top, in_sample_alpha_benign, Y, c_m, c_n, c_a, N_first = False)
                for bot in thresholds_to_check:
                    if bot < lowest_bot or bot > top:
                        continue
                    else:
                        #Check budget remaining
                        budget_remaining = Y - c_m - c_n*(np.mean(in_sample_alpha_benign < top) - np.mean(in_sample_alpha_benign < bot))
                        n_benign_audited = floor(budget_remaining/c_a*len(in_sample_alpha_benign))
                        audit_fpr = n_benign_audited/len(in_sample_alpha_benign)
                        assert n_benign_audited >= 0, "Negative audit budget"
                        #get benign lnratios in interval
                        benign_in_interval_filter = (in_sample_alpha_benign < top) & (in_sample_alpha_benign >= bot)
                        if benign_in_interval_filter.any():
                            benign_ln_in_interval = self.get_lnl_ratio(in_sample_alpha_benign[benign_in_interval_filter], in_sample_beta_benign[benign_in_interval_filter])
                        else:
                            benign_ln_in_interval = np.empty(0)
                        #now get benign lnratios out of interval
                        benign_out_interval_filter = ~benign_in_interval_filter
                        if benign_out_interval_filter.any():
                            benign_out_interval_lnl = self.get_lnl_ratio(in_sample_alpha_benign[benign_out_interval_filter], None)
                        else:
                            benign_out_interval_lnl = np.empty(0)

                        benign_lnratio_all_soreted = np.sort(np.concatenate((benign_ln_in_interval, benign_out_interval_lnl)))

                        assert len(benign_lnratio_all_soreted) == len(in_sample_alpha_benign)
                        if n_benign_audited > 0:
                            audit_cutoff = benign_lnratio_all_soreted[-n_benign_audited]
                        else:
                            audit_cutoff = benign_lnratio_all_soreted[-1] + 1e-6
                        
                        #check cost:
                        assert Y - c_m - c_n*(np.mean(benign_in_interval_filter)) - c_a*np.mean(benign_lnratio_all_soreted >= audit_cutoff) + 1e-10 >= 0, "Budget blown"

                        #Check safety on attack ln ratios
                        malign_in_interval_filter = (in_sample_alpha_attack >= bot) & (in_sample_alpha_attack < top)
                        if malign_in_interval_filter.any():
                            malign_ln_in_interval = self.get_lnl_ratio(in_sample_alpha_attack[malign_in_interval_filter], in_sample_beta_attack[malign_in_interval_filter])
                        else:
                            malign_ln_in_interval = np.empty(0)
                        malign_out_interval_filter = ~malign_in_interval_filter
                        if malign_out_interval_filter.any():
                            malign_out_interval_lnl = self.get_lnl_ratio(in_sample_alpha_attack[malign_out_interval_filter], None)
                        else:
                            malign_out_interval_lnl = np.empty(0)

                        malign_lnratio_all = np.concatenate((malign_ln_in_interval, malign_out_interval_lnl))
                        assert len(malign_lnratio_all) == len(in_sample_alpha_attack)
                        n_caught = np.sum(malign_lnratio_all >= audit_cutoff)
                        safety = n_caught/len(in_sample_alpha_attack)
                        
                        all_results.append((top, bot, safety, audit_cutoff, False, audit_fpr))

        #Sort results by safety
        all_results.sort(key=lambda x: x[2], reverse=True)
        self.all_results = all_results
        #Get best result
        self.best_result = all_results[0]
        self.top_thresh = self.best_result[0]
        self.bot_thresh = self.best_result[1]
        self.in_sample_safety = self.best_result[2]
        self.audit_thresh = self.best_result[3]
        self.N_first = self.best_result[4]
        self.audit_fpr = self.best_result[5]
        self.audit_budget_share = self.audit_fpr*c_a/Y
        #calculate use two monitor probability
        self.use_two_monitor_prob = np.mean((in_sample_beta_benign < self.top_thresh) & (in_sample_beta_benign >= self.bot_thresh)) if self.N_first else np.mean(
            (in_sample_alpha_benign < self.top_thresh) & (in_sample_alpha_benign >= self.bot_thresh)) #type: ignore
        self.Y = Y
        self.c_m = c_m
        self.c_n = c_n
        self.c_a = c_a
    
    def get_oos_safety(self, oos_alpha_attack: np.ndarray, oos_beta_attack: np.ndarray) -> float:
        assert self.fitted, "Strategy not fitted"
        if self.N_first:
            attack_in_interval = (oos_beta_attack >= self.bot_thresh) & (oos_beta_attack < self.top_thresh)
            if attack_in_interval.any():
                malign_ln_in_interval = self.get_lnl_ratio(oos_alpha_attack[attack_in_interval], oos_beta_attack[attack_in_interval])
            else:
                malign_ln_in_interval = np.empty(0)
            malign_out_interval_filter = ~attack_in_interval
            if malign_out_interval_filter.any():
                malign_out_interval_lnl = self.get_lnl_ratio(None, oos_beta_attack[malign_out_interval_filter])
            else:
                malign_out_interval_lnl = np.empty(0)
            malign_lnratio_all = np.concatenate((malign_ln_in_interval, malign_out_interval_lnl))
            assert len(malign_lnratio_all) == len(oos_alpha_attack)
            n_caught = np.sum(malign_lnratio_all >= self.audit_thresh)
            return n_caught/len(oos_alpha_attack)
        else:
            attack_in_interval = (oos_alpha_attack >= self.bot_thresh) & (oos_alpha_attack < self.top_thresh)
            if attack_in_interval.any():
                malign_ln_in_interval = self.get_lnl_ratio(oos_alpha_attack[attack_in_interval], oos_beta_attack[attack_in_interval])
            else:
                malign_ln_in_interval = np.empty(0)
            malign_out_interval_filter = ~attack_in_interval
            if malign_out_interval_filter.any():
                malign_out_interval_lnl = self.get_lnl_ratio(oos_alpha_attack[malign_out_interval_filter], None)
            else:
                malign_out_interval_lnl = np.empty(0)
            malign_lnratio_all = np.concatenate((malign_ln_in_interval, malign_out_interval_lnl))
            assert len(malign_lnratio_all) == len(oos_alpha_attack)
            n_caught = np.sum(malign_lnratio_all >= self.audit_thresh)
            return n_caught/len(oos_alpha_attack)
    
    def save_parameters(self) -> str:
        return json.dumps({
            "top_thresh": self.top_thresh,
            "bot_thresh": self.bot_thresh,
            "audit_thresh": self.audit_thresh,
            "N_first": self.N_first,
            "bandwidth_2d": self.bandwidth_2d,
            "bandwidth_1d": self.bandwidth_1d,
            "bandwidth_2d_malign": self.malign_2d_ker.bandwidth_,
            "bandwidth_2d_benign": self.benign_2d_ker.bandwidth_,
            "bandwidth_alpha_attack": self.alpha_attack_ker.bandwidth_,
            "bandwidth_beta_attack": self.beta_attack_ker.bandwidth_,
            "bandwidth_alpha_benign": self.alpha_benign_ker.bandwidth_,
            "bandwidth_beta_benign": self.beta_benign_ker.bandwidth_,
        })
    
    def __repr__(self) -> str:
        if self.fitted:
            return f"FullNPOneRegion(in_sample_safety={self.in_sample_safety}, top_thresh={self.top_thresh}, bot_thresh={self.bot_thresh}, audit_thresh={self.audit_thresh}, N_first={self.N_first}, bandwidth_2d={self.bandwidth_2d}, bandwidth_1d={self.bandwidth_1d})"
        else:
            return f"FullNPOneRegion(not fitted)"


    def plot_decision_regions(
        self,
        alpha_plot_range: Optional[Tuple[float, float]] = None,
        beta_plot_range: Optional[Tuple[float, float]] = None,
        num_points: int = 50,
    ) -> None:
        """
        Plots the decision regions, LLR heatmap, and audit cutoff after the model is fitted.
        Alpha scores are on the x-axis, Beta scores on the y-axis.
        """
        if not self.fitted:
            print("Model has not been fitted yet. Please call fit_method first.")
            return
        
        if self.top_thresh is None or self.bot_thresh is None or self.audit_thresh is None or self.N_first is None:
            print("Fit method did not properly set all required thresholds. Cannot plot.")
            return

        # Define score ranges for plotting
        alpha_range = alpha_plot_range if alpha_plot_range is not None else (2.0, 9.5)
        beta_range = beta_plot_range if beta_plot_range is not None else (4.0, 8.0)

        x_coords = np.linspace(alpha_range[0], alpha_range[1], num_points)
        y_coords = np.linspace(beta_range[0], beta_range[1], num_points)

        Alpha_grid, Beta_grid = np.meshgrid(x_coords, y_coords, indexing='ij')
        alpha_scores_flat = Alpha_grid.ravel()
        beta_scores_flat = Beta_grid.ravel()

        llr_values_flat = np.zeros_like(alpha_scores_flat, dtype=float)

        if self.N_first:  # Beta monitor first
            # Region where 2D LLR is used (both monitors)
            mask_2d = (beta_scores_flat >= self.bot_thresh) & (beta_scores_flat < self.top_thresh)
            # Region where 1D LLR is used (only Beta monitor's score)
            mask_1d = ~mask_2d

            if np.any(mask_2d):
                llr_values_flat[mask_2d] = self.get_lnl_ratio(
                    alpha_scores=alpha_scores_flat[mask_2d],
                    beta_scores=beta_scores_flat[mask_2d]
                )
            if np.any(mask_1d):
                llr_values_flat[mask_1d] = self.get_lnl_ratio(
                    alpha_scores=None,
                    beta_scores=beta_scores_flat[mask_1d]
                )
        else:  # Alpha monitor first
            # Region where 2D LLR is used (both monitors)
            mask_2d = (alpha_scores_flat >= self.bot_thresh) & (alpha_scores_flat < self.top_thresh)
            # Region where 1D LLR is used (only Alpha monitor's score)
            mask_1d = ~mask_2d

            if np.any(mask_2d):
                llr_values_flat[mask_2d] = self.get_lnl_ratio(
                    alpha_scores=alpha_scores_flat[mask_2d],
                    beta_scores=beta_scores_flat[mask_2d]
                )
            if np.any(mask_1d):
                llr_values_flat[mask_1d] = self.get_lnl_ratio(
                    alpha_scores=alpha_scores_flat[mask_1d],
                    beta_scores=None
                )
        
        LLR_grid = llr_values_flat.reshape(Alpha_grid.shape)

        plt.figure(figsize=(10, 8), dpi = 300)
        
        # Plot LLR heatmap
        plt.imshow(
            LLR_grid.T,  # Transpose for correct orientation
            extent=(alpha_range[0], alpha_range[1], beta_range[0], beta_range[1]),
            origin='lower',
            aspect='auto',
            cmap='plasma',
            interpolation='nearest'
        )
        plt.colorbar(label='Log Likelihood Ratio (LLR)')

        # Plot top and bottom thresholds
        if self.N_first: # Beta first, thresholds are on beta axis (horizontal lines)
            plt.axhline(y=self.top_thresh, color='black', linestyle='--', linewidth=2, label=f'Top Threshold ({self.top_thresh:.2f})')
            plt.axhline(y=self.bot_thresh, color='black', linestyle=':', linewidth=2, label=f'Bottom Threshold ({self.bot_thresh:.2f})')
        else: # Alpha first, thresholds are on alpha axis (vertical lines)
            plt.axvline(x=self.top_thresh, color='black', linestyle='--', linewidth=2, label=f'Top Threshold ({self.top_thresh:.2f})')
            plt.axvline(x=self.bot_thresh, color='black', linestyle=':', linewidth=2, label=f'Bottom Threshold ({self.bot_thresh:.2f})')

        # Highlight region where LLR >= audit_thresh
        # Note: contour uses Z with same orientation as X, Y grids. imshow uses Z.T for origin='lower'.
        if LLR_grid.size > 0 : #Ensure grid is not empty
            contour_set = plt.contour(
                Alpha_grid, Beta_grid, LLR_grid,
                levels=[self.audit_thresh],
                colors='lime', # Using a distinct color like lime green
                linewidths=2.5,
                linestyles='solid'
            )
                # Create a proxy artist for the legend
            lime_line = plt.Line2D([0], [0], color='lime', lw=2.5, linestyle='solid')
            
            # When creating the legend, use the custom handle
            handles, labels = plt.gca().get_legend_handles_labels()  # Get existing handles/labels
            handles.append(lime_line)  # Add our custom handle
            labels.append(f'LLR = Audit Threshold ({self.audit_thresh:.2f})')  # Add our custom label
            
            plt.legend(handles, labels, loc='best')

        plt.xlabel('Alpha Scores')
        plt.ylabel('Beta Scores')
        plt.title(f'LLR Heatmap & Decision Regions ({"Beta" if self.N_first else "Alpha"} Monitor First)')
        #plt.legend(loc='best')
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.show()



class AuditEndQuadNandM(Strategy):
    """
    AuditEnd strategy that searchers for best thing under calling N first and then best under calling M first.
    Uses the quadratic local odds ratio fit thing. 
    """

    def __init__(
        self,
        local_odds_ratio_bandwidth: float = 3.0,
        verbose: bool = False,
    ):
        self.fitted: bool = False
        self.N_first: Optional[bool] = None
        self.local_odds_ratio_bandwidth: float = local_odds_ratio_bandwidth
        self.audit_thresh: Optional[float] = None
        self.end_thresh: Optional[float] = None
        self.x_vec: Optional[np.ndarray | list[float]] = None
        self.f_mn: Optional[Callable[[float, float], float]] = None
        self.f_mn_coefs: Optional[np.ndarray | list[float]] = None
        self.verbose: bool = verbose


    def fit_method(
        self,
        in_sample_alpha_benign: np.ndarray,
        in_sample_alpha_attack: np.ndarray,
        in_sample_beta_benign: np.ndarray,
        in_sample_beta_attack: np.ndarray,
        Y: float,
        c_m: float,
        c_n: float,
        c_a: float,
        **kwargs: Any,
    ) -> None:
        if self.f_mn is None:
            # We do eventually want explore better ways of doing this
            self.local_odds_ratio = estimate_local_ratio(
                alpha_scores=np.concatenate(
                    [in_sample_alpha_attack, in_sample_alpha_benign]
                ),
                beta_scores=np.concatenate(
                    [in_sample_beta_attack, in_sample_beta_benign]
                ),
                attack_labels=np.concatenate(
                    [
                        np.ones(len(in_sample_alpha_attack)),
                        np.zeros(len(in_sample_alpha_benign)),
                    ]
                ),
                bandwidth=self.local_odds_ratio_bandwidth,
            )

            self.f_mn, self.f_mn_coefs = fit_quadratic(
                a=np.concatenate([in_sample_alpha_attack, in_sample_alpha_benign]),
                b=np.concatenate([in_sample_beta_attack, in_sample_beta_benign]),
                ratio_est=self.local_odds_ratio,
            )
        if Y - c_n > 0:
            sweep_results_n = sweep_solve(
                alpha_benign_scores=in_sample_alpha_benign,
                alpha_attack_scores=in_sample_alpha_attack,
                beta_benign_scores=in_sample_beta_benign,
                beta_attack_scores=in_sample_beta_attack,
                Y=Y,
                c_m=c_m,
                c_n=c_n,
                c_a=c_a,
                f_mn=self.f_mn,
                verbose=self.verbose,
                N_first=True,
                **kwargs,
            )
            best_n_result_idx = np.argmax([res["safety"] for res in sweep_results_n])
            best_n_result = sweep_results_n[best_n_result_idx]
        if Y - c_m > 0:
            sweep_results_m = sweep_solve(
                alpha_benign_scores=in_sample_alpha_benign,
                alpha_attack_scores=in_sample_alpha_attack,
                beta_benign_scores=in_sample_beta_benign,
                beta_attack_scores=in_sample_beta_attack,
                Y=Y,
                c_m=c_m,
                c_n=c_n,
                c_a=c_a,
                f_mn=self.f_mn,
                verbose=self.verbose,
                N_first=False,
                **kwargs,
            )
            best_m_result_idx = np.argmax([res["safety"] for res in sweep_results_m])
            best_m_result = sweep_results_m[best_m_result_idx]
        else:
            best_m_result = None
        
        if best_m_result:
            if best_m_result['safety'] > best_n_result['safety']:
                best_result = best_m_result
                self.N_first = False
            else:
                best_result = best_n_result
                self.N_first = True
        else:
            best_result = best_n_result
            self.N_first = True
        
        self.in_sample_safety = best_result["safety"]
        self.audit_thresh = best_result["audit_threshold"]
        self.end_thresh = best_result["end_threshold"]

        self.fitted = True
        self.Y = Y
        self.c_m = c_m
        self.c_n = c_n
        self.c_a = c_a
        self.x_vec = get_x_with_fin_audit_thresh(
            x=np.array(
                [
                    0,
                    self.audit_thresh,
                    self.end_thresh,
                    self.audit_thresh,
                    self.end_thresh,
                    0,
                    0,
                ]
            )
            if self.N_first
            else np.array(
                [
                    1,
                    self.audit_thresh,
                    self.end_thresh,
                    self.audit_thresh,
                    self.end_thresh,
                    0,
                    0,
                ]
            ),
            alpha_benign_scores=in_sample_alpha_benign,
            beta_benign_scores=in_sample_beta_benign,
            f_mn=self.f_mn,
            c_a=c_a,
            c_m=c_m,
            c_n=c_n,
            Y=Y,
            N_first=self.N_first,
        )
        self.audit_fpr = p_a_given_not_q_twomonitors(
            x=self.x_vec,
            alpha_benign=in_sample_alpha_benign,
            beta_benign=in_sample_beta_benign,
            f_mn=self.f_mn,
        )
        self.audit_budget_share = self.audit_fpr * c_a / Y
        if self.N_first:
            self.use_two_monitor_prob = np.mean(
                (in_sample_beta_benign < self.audit_thresh)
                & (in_sample_beta_benign >= self.end_thresh)
            )  # type: ignore
        else:
            self.use_two_monitor_prob = np.mean(
                (in_sample_alpha_benign < self.audit_thresh)
                & (in_sample_alpha_benign >= self.end_thresh)
            )  # type: ignore

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert (
            self.x_vec is not None
            and self.audit_thresh is not None
            and self.end_thresh is not None
        ), "Parameters not computed"
        assert self.f_mn is not None, "f_mn not computed"
        return p_a_given_q_twomonitors(
            x=self.x_vec,
            alpha_attack_scores=oos_alpha_attack,
            beta_attack_scores=oos_beta_attack,
            f_mn=self.f_mn,
        )

    def compile_quadratic_f_mn(
        self, f_mn_coefs: np.ndarray
    ) -> Callable[[float, float], float]:
        assert len(f_mn_coefs) == 6, "f_mn_coefs must be a 6-element array"
        return (
            lambda x, y: f_mn_coefs[0] * x * x
            + f_mn_coefs[1] * y * y
            + f_mn_coefs[2] * x * y
            + f_mn_coefs[3] * x
            + f_mn_coefs[4] * y
            + f_mn_coefs[5]
        )

    def get_insample_safety(self) -> float:
        assert self.in_sample_safety is not None, "In sample safety not computed"
        return self.in_sample_safety

    def get_audit_fpr(self) -> float:
        assert self.audit_fpr is not None, "Audit FPR not computed"
        return self.audit_fpr

    def get_audit_budget_share(self) -> float:
        assert self.audit_budget_share is not None, "Audit budget share not computed"
        return self.audit_budget_share

    def get_use_two_monitor_prob(self) -> float:
        assert (
            self.use_two_monitor_prob is not None
        ), "Use two monitor prob not computed"
        return self.use_two_monitor_prob

    def load_parameters(self, params: dict[str, Any]) -> None:
        self.x_vec = params["x"]
        self.f_mn_coefs = params["f_mn_coefs"]
        self.f_mn = self.compile_quadratic_f_mn(self.f_mn_coefs)  # type: ignore

    def save_parameters(self) -> str:
        assert self.x_vec is not None
        return json.dumps(
            {
                "x": self.x_vec
                if isinstance(self.x_vec, list)
                else self.x_vec.tolist(),
                "f_mn_coefs": self.f_mn_coefs
                if isinstance(self.f_mn_coefs, list)
                else self.f_mn_coefs.tolist(),
            }
        )

    def __repr__(self) -> str:
        if self.end_thresh is None:
            return f"AuditEndQuadNandM(fitted = {self.fitted}, N_first = {self.N_first})"
        else:
            return f"AuditEndQuadNandM(x = {self.x_vec}, f_mn_coefs = {self.f_mn_coefs}, N_first = {self.N_first})"

    def reset(self) -> None:
        # Keep N_first and local_odds_ratio_bandwidth but reset everything else
        self.fitted = False
        self.audit_thresh = None
        self.end_thresh = None
        self.x_vec = None
        self.f_mn = None
        self.f_mn_coefs = None
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.use_two_monitor_prob = None


