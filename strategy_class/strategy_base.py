# %%
from pathlib import Path

project_dir = Path(__file__).parent.parent
import sys

sys.path.append(str(project_dir))
import numpy as np
from typing import Any, Callable, Optional, cast
from abc import ABC, abstractmethod
import json
from binary_monitors import adaptive_mesh_optimization, oos_safety_binary
from binary_monitors import p_a_given_q0 as p_a_given_q0_binary
from two_monitors import (
    sweep_solve,
    fit_quadratic,
    estimate_local_ratio,
    get_x_with_fin_audit_thresh,
)
from math import floor
from two_monitors import p_a_given_q as p_a_given_q_twomonitors
from two_monitors import p_a_given_not_q as p_a_given_not_q_twomonitors
from single_monitor import solve_1mon_optimal_policy, evaluate_1mon_ood
from single_monitor import get_naive_single_monitor_policy, apply_naive_strategy


class Strategy(ABC):
    """
    Base class for all strategies.
    """
    def __init__(self) -> None:
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob: Optional[float] = None
    

    def reset(self) -> None:
        """
        Reset the strategy to the initial state.
        """
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.use_two_monitor_prob = None

    @abstractmethod
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
        """
        Takes in-sample training data and the budget parameters, and fits the strategy.
        """
        pass

    @abstractmethod
    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        """
        If the strategy is fitted, return the out-of-sample safety of the strategy.
        """
        pass

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
    
    def get_insample_safety(self) -> float:
        assert self.in_sample_safety is not None, "In sample safety not computed"
        return self.in_sample_safety

    @abstractmethod
    def save_parameters(self) -> str:
        """
        Save the parameters of the strategy to a string.
        """
        pass



    @abstractmethod
    def __repr__(self) -> str:
        pass


class SingleMonitorStrategy(Strategy):
    """
    Optimized version of the single monitor strategy which could
    call the monitor with some probability that's not one, and possibly
    audit immediately without calling the monitor at all (or end immediately).

    Change which monitor to use by changing the `alpha` parameter.

    See poster for details (https://tinyurl.com/hierarchical-monitoring)
    """

    def __init__(self, alpha: bool = True) -> None:
        self.alpha = alpha
        self.a_m: Optional[float] = None
        self.a_s: Optional[float] = None
        self.p_m: Optional[float] = None
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.Y: Optional[float] = None
        self.c_m: Optional[float] = None
        self.c_a: Optional[float] = None
        self.fitted: bool = False

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
        self.in_sample_safety, self.a_m, self.p_m, self.a_s = solve_1mon_optimal_policy(
            c_a,
            c_m if self.alpha else c_n,
            Y,
            attack_scores=in_sample_alpha_attack
            if self.alpha
            else in_sample_beta_attack,
            honest_scores=in_sample_alpha_benign
            if self.alpha
            else in_sample_beta_benign,
            **kwargs,
        )
        self.fitted = True
        self.Y = Y
        self.c_m = c_m if self.alpha else c_n
        self.c_a = c_a
        fpr = (
            self.a_s + self.p_m * np.mean(in_sample_alpha_benign >= self.a_m)
            if self.alpha
            else self.a_s + self.p_m * np.mean(in_sample_beta_benign >= self.a_m)
        )
        self.audit_fpr = cast(float, fpr)
        self.audit_budget_share = self.audit_fpr * self.c_a / Y

    def get_insample_safety(self) -> float:
        assert self.in_sample_safety is not None, "In sample safety not computed"
        return self.in_sample_safety

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert (
            self.a_m is not None and self.p_m is not None and self.a_s is not None
        ), "Parameters not computed"
        return evaluate_1mon_ood(
            oos_alpha_attack if self.alpha else oos_beta_attack,
            self.a_m,
            self.p_m,
            self.a_s,
        )

    def get_audit_fpr(self) -> float:
        assert self.audit_fpr is not None, "Audit FPR not computed"
        return self.audit_fpr

    def get_audit_budget_share(self) -> float:
        assert self.audit_budget_share is not None, "Audit budget share not computed"
        return self.audit_budget_share

    def load_parameters(self, params: dict[str, Any]) -> None:
        """
        Load parameters from a dictionary.
        """
        self.alpha = params["alpha"]
        self.a_m = params["a_m"]
        self.p_m = params["p_m"]
        self.a_s = params["a_s"]

    def save_parameters(self) -> str:
        return json.dumps(
            {
                "alpha": self.alpha,
                "a_m": self.a_m,
                "p_m": self.p_m,
                "a_s": self.a_s,
            }
        )

    def get_use_two_monitor_prob(self) -> float:
        return 0.0

    def __repr__(self) -> str:
        if self.a_m is None:
            return f"SingleMonitorStrategy(alpha = {self.alpha})"
        else:
            return f"SingleMonitorStrategy(alpha = {self.alpha}, a_m = {self.a_m}, p_m = {self.p_m}, a_s = {self.a_s}, fitted = {self.fitted})"

    def reset(self) -> None:
        # Keep alpha but reset everything else to initial state
        self.a_m = None
        self.a_s = None
        self.p_m = None
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.Y = None
        self.c_m = None
        self.c_a = None
        self.fitted = False


class BinaryMonitorStrategy(Strategy):
    """
    An older strategy which turns the two monitors into two binary (i.e., outputs scores of 
    0 or 1) monitors first. Mesh searchers over a bunch of possible thresholds for binerizing the monitors.
    """

    def __init__(self, verbose: bool = False) -> None:
        self.fitted: bool = False
        self.alpha_cutoff: Optional[float] = None
        self.beta_cutoff: Optional[float] = None
        self.x_vec: Optional[np.ndarray | list[float]] = None
        self.Y: Optional[float] = None
        self.c_m: Optional[float] = None
        self.c_n: Optional[float] = None
        self.c_a: Optional[float] = None
        self.in_sample_safety: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.use_two_monitor_prob: Optional[float] = None
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
        best_strategy, _ = adaptive_mesh_optimization(
            Y,
            c_m,
            c_n,
            c_a,
            honest_beta_scores=in_sample_beta_benign,
            honest_alpha_scores=in_sample_alpha_benign,
            attack_beta_scores=in_sample_beta_attack,
            attack_alpha_scores=in_sample_alpha_attack,
            verbose=self.verbose,
            **kwargs,
        )

        self.fitted = True
        self.prob = json.loads(best_strategy["prob_json"])  # type: ignore
        self.x_vec = cast(np.ndarray, best_strategy["x"])  # type: ignore
        self.alpha_cutoff = best_strategy["actual_alpha"]  # type: ignore
        self.beta_cutoff = best_strategy["actual_beta"]  # type: ignore
        self.in_sample_safety = best_strategy["best_value"]  # type: ignore
        self.audit_fpr = p_a_given_q0_binary(self.x_vec, self.prob)
        self.audit_budget_share = self.audit_fpr * c_a / Y
        self.Y = Y
        self.c_m = c_m
        self.c_n = c_n
        self.c_a = c_a
        pM = self.x_vec[0]
        pQ_M1 = self.x_vec[3]
        pQ_M0 = self.x_vec[4]
        pQ_N1 = self.x_vec[7]
        pQ_N0 = self.x_vec[8]

        pm1_q0 = self.prob["p_m_given_not_q"]
        pm0_q0 = 1.0 - pm1_q0
        pn1_q0 = self.prob["p_n_given_not_q"]
        pn0_q0 = 1.0 - pn1_q0
        p_q_after_m = pm1_q0 * pQ_M1 + pm0_q0 * pQ_M0
        p_q_after_n = pn1_q0 * pQ_N1 + pn0_q0 * pQ_N0
        self.use_two_monitor_prob = p_q_after_m * pM + p_q_after_n * (1 - pM)

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert (
            self.x_vec is not None
            and self.alpha_cutoff is not None
            and self.beta_cutoff is not None
        ), "Parameters not computed"
        return oos_safety_binary(
            alpha_cutoff=self.alpha_cutoff,
            beta_cutoff=self.beta_cutoff,
            x=self.x_vec,
            oos_alpha_attack=oos_alpha_attack,
            oos_beta_attack=oos_beta_attack,
        )

    def get_insample_safety(self) -> float:
        assert self.in_sample_safety is not None, "In sample safety not computed"
        return self.in_sample_safety  # type: ignore

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
        self.alpha_cutoff = params["alpha_cutoff"]
        self.beta_cutoff = params["beta_cutoff"]

    def save_parameters(self) -> str:
        assert self.x_vec is not None, "x_vec not computed"
        return json.dumps(
            {
                "alpha_cutoff": self.alpha_cutoff,
                "beta_cutoff": self.beta_cutoff,
                "x": self.x_vec
                if isinstance(self.x_vec, list)
                else self.x_vec.tolist(),
                "prob": self.prob,
            }
        )

    def __repr__(self) -> str:
        if self.alpha_cutoff is None:
            return f"BinaryMonitorStrategy(fitted = {self.fitted})"
        else:
            return f"BinaryMonitorStrategy(alpha_cutoff = {self.alpha_cutoff}, beta_cutoff = {self.beta_cutoff}, x = {self.x_vec}, fitted = {self.fitted})"

    def reset(self) -> None:
        self.fitted = False
        self.alpha_cutoff = None
        self.beta_cutoff = None
        self.x_vec = None
        self.Y = None
        self.c_m = None
        self.c_n = None
        self.c_a = None
        self.in_sample_safety = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.use_two_monitor_prob = None


class AuditEndThreshStrategy(Strategy):
    """
    See poster for details (https://tinyurl.com/hierarchical-monitoring). We
    ensemble the two monitors by first estimating the local
    bayes factor P(Q=1|m,n)/P(Q=0|m,n) and then fitting a quadratic function to it.
    We then use this as our estimate for like possible level curves of the likelihood ratio and 
    decide where the cutoff point for auditing should be after calling both monitors.

    See e.g., https://proceedings.neurips.cc/paper_files/paper/2008/file/c6e19e830859f2cb9f7c8f8cacb8d2a6-Paper.pdf

    N_first = True means using the cheaper monitor (100% of the time) first. N_first = False has not been implemented yet.
    """

    def __init__(
        self,
        N_first: bool = True,
        local_odds_ratio_bandwidth: float = 3.0,
        verbose: bool = False,
    ):
        self.fitted: bool = False
        self.N_first: bool = N_first
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

        sweep_results = sweep_solve(
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
            N_first=self.N_first,
            **kwargs,
        )
        self.fitted = True
        self.Y = Y
        self.c_m = c_m
        self.c_n = c_n
        self.c_a = c_a
        best_result_idx = np.argmax([res["safety"] for res in sweep_results])
        best_result = sweep_results[best_result_idx]
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
            return f"AuditEndThreshStrategy(fitted = {self.fitted}, N_first = {self.N_first})"
        else:
            return f"AuditEndThreshStrategy(x = {self.x_vec}, f_mn_coefs = {self.f_mn_coefs}, fitted = {self.fitted}, N_first = {self.N_first})"

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


class NaiveSingleMonitorStrategy(Strategy):
    """
    Naive single monitor strategy which just uses a monitor 100% of the time and
    spend the rest of the budget on auditing.

    Change which monitor to use by changing the `alpha` parameter.
    """

    def __init__(self, alpha: bool = True) -> None:
        self.alpha = alpha
        self.a_m: Optional[float] = None
        self.audit_fpr: Optional[float] = None
        self.audit_budget_share: Optional[float] = None
        self.in_sample_safety: Optional[float] = None
        self.use_two_monitor_prob: float = 0
        self.fitted: bool = False

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
        self.a_m = (
            get_naive_single_monitor_policy(in_sample_alpha_benign, Y, c_a, c_m)
            if self.alpha
            else get_naive_single_monitor_policy(in_sample_beta_benign, Y, c_a, c_n)
        )
        self.audit_fpr = apply_naive_strategy(
            in_sample_alpha_benign if self.alpha else in_sample_beta_benign, self.a_m
        )
        self.audit_budget_share = self.audit_fpr * c_a / Y
        self.in_sample_safety = apply_naive_strategy(
            in_sample_alpha_attack if self.alpha else in_sample_beta_attack, self.a_m
        )
        self.fitted = True

    def get_oos_safety(
        self,
        oos_alpha_attack: np.ndarray,
        oos_beta_attack: np.ndarray,
    ) -> float:
        assert self.a_m is not None, "a_m not computed"
        return apply_naive_strategy(
            oos_alpha_attack if self.alpha else oos_beta_attack, self.a_m
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
        return 0.0

    def load_parameters(self, params: dict[str, Any]) -> None:
        self.a_m = params["a_m"]

    def save_parameters(self) -> str:
        return json.dumps(
            {
                "a_m": self.a_m,
            }
        )

    def __repr__(self) -> str:
        if self.a_m is None:
            return f"NaiveSingleMonitorStrategy(alpha = {self.alpha})"
        else:
            return f"NaiveSingleMonitorStrategy(alpha = {self.alpha}, a_m = {self.a_m}, fitted = {self.fitted})"

    def reset(self) -> None:
        self.fitted = False
        self.a_m = None
        self.audit_fpr = None
        self.audit_budget_share = None
        self.in_sample_safety = None
        self.use_two_monitor_prob = 0




# %%
