# %%
import numpy as np
from numpy import ndarray
import pandas as pd
import scipy.interpolate as interp  # type: ignore
from typing import Callable, Tuple, Any, Union, List
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc  # type: ignore
from pathlib import Path
import warnings


def gaussian_kernel(d2: np.ndarray, h: float) -> np.ndarray:
    """
    Returns a Gaussian kernel weight given squared distances d2 and bandwidth h.
    d2: 1D array of squared distances
    h : bandwidth parameter
    """
    return np.exp(-d2 / (2.0 * h * h))


def estimate_local_ratio(
    alpha_scores: np.ndarray,
    beta_scores: np.ndarray,
    attack_labels: np.ndarray,
    bandwidth: float = 1.0,
) -> np.ndarray:
    """
    For each index i, compute a ratio:
        ratio_i = sum_j [K(dist_ij) * I(attack_j=1)]  /  sum_j [K(dist_ij) * I(attack_j=0)]
    Then clamp ratio_i to [0, 100].

    alpha_scores, beta_scores, and attack_labels must be 1D numpy arrays of the same length.

    Returns: 1D np.array of ratio values, same length as alpha_scores.
    """
    coords = np.column_stack((alpha_scores, beta_scores))  # shape (N,2)
    attacks_int = attack_labels.astype(float)  # 1 if True, 0 if False

    N = len(attack_labels)
    ratio_values = np.zeros(N, dtype=float)

    for i in range(N):
        # Squared Euclidean distance from point i to all other points
        dist2 = np.sum((coords - coords[i]) ** 2, axis=1)
        # Kernel weights
        w = gaussian_kernel(dist2, h=bandwidth)

        num = np.sum(w * attacks_int)  # Weighted sum of t=1
        den = np.sum(w * (1 - attacks_int))  # Weighted sum of t=0

        # ratio = num/den, handle den=0 by small offset
        ratio_raw = num / max(den, 1e-6)
        # clamp ratio to [0, 100]
        ratio_values[i] = np.clip(ratio_raw, 0, 100.0)

    return ratio_values


def fit_quadratic(
    a: np.ndarray, b: np.ndarray, ratio_est: np.ndarray, verbose: bool = False
) -> tuple[Callable[[float, float], float], np.ndarray]:
    """
    Fits a quadratic function f(a,b) = A a^2 + B b^2 + C a b + D a + E b + F
    using least squares to the ratio_est values at each (a, b).

    a, b: 1D arrays of the same length
    ratio_est: 1D array of ratio estimates (same length)
    verbose: if True, print regression summary statistics

    Returns: A function f_fitted(x, y) that computes the fitted polynomial value.
    """
    # Construct design matrix X
    X = np.column_stack([a * a, b * b, a * b, a, b, np.ones_like(a)])

    # Solve for coefficients [A,B,C,D,E,F]
    beta, residuals, rank, s = np.linalg.lstsq(X, ratio_est, rcond=None)

    if verbose:
        print("\nQuadratic Regression Summary:")
        print("-" * 30)
        terms = ["a²", "b²", "ab", "a", "b", "intercept"]
        for term, coef in zip(terms, beta):
            print(f"{term:10} coefficient: {coef:10.6f}")
        if len(residuals) > 0:
            mse = residuals[0] / len(ratio_est)
            print(f"\nMean squared error: {mse:.6f}")
            print(
                f"R-squared: {1 - residuals[0]/np.sum((ratio_est - np.mean(ratio_est))**2):.6f}"
            )
            print(f"Matrix rank: {rank}")

    def f_fitted(x: float | np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the fitted polynomial at (x,y).
        """
        return (
            beta[0] * x * x
            + beta[1] * y * y
            + beta[2] * x * y
            + beta[3] * x
            + beta[4] * y
            + beta[5]
        )

    return f_fitted, beta  # type: ignore


def create_quadratic_function(coefficients: Union[List[float], np.ndarray]) -> Callable[[float, float], float]:
    """
    Creates a quadratic function f(m,n) = A m^2 + B n^2 + C m n + D m + E n + F
    from a list/array of coefficients [A, B, C, D, E, F].
    
    Parameters:
    -----------
    coefficients : Union[List[float], np.ndarray]
        A list or numpy array of 6 coefficients [A, B, C, D, E, F] for the quadratic function
    
    Returns:
    --------
    Callable[[float, float], float]
        A function that takes two inputs (m, n) and returns the quadratic value
        
    Examples:
    ---------
    >>> coefs = [1, 2, 3, 4, 5, 6]  # A=1, B=2, C=3, D=4, E=5, F=6
    >>> f = create_quadratic_function(coefs)
    >>> f(2, 3)  # 1*2^2 + 2*3^2 + 3*2*3 + 4*2 + 5*3 + 6 = 4 + 18 + 18 + 8 + 15 + 6 = 69
    69.0
    """
    if len(coefficients) != 6:
        raise ValueError("Exactly 6 coefficients [A, B, C, D, E, F] are required for a quadratic function")
    
    coefficients = np.array(coefficients, dtype=float)
    
    def f_mn(m: Union[float, np.ndarray], n: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluate the quadratic function f(m,n) = A m^2 + B n^2 + C m n + D m + E n + F
        
        Parameters:
        -----------
        m : float or np.ndarray
            First input variable
        n : float or np.ndarray
            Second input variable
            
        Returns:
        --------
        float or np.ndarray
            The value of the quadratic function at (m,n)
        """
        A, B, C, D, E, F = coefficients
        return A * m * m + B * n * n + C * m * n + D * m + E * n + F
    
    return f_mn

def compute_threshold_curve(
    alpha_scores: np.ndarray,
    beta_scores: np.ndarray,
    attack_labels: np.ndarray,
    f_model: Callable[[float, float], float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    1. Evaluate f_model at each row's (alpha_scores, beta_scores).
    2. Let t_min = min(f_model(...) ), t_max = max(f_model(...) ).
    3. Scan thresholds T from t_max down to t_min (100 steps).
    4. For each threshold, compute false positive rate and true positive rate in
       the region {f_model(a,b) >= threshold}.

    Returns:
        thresholds: 1D array of thresholds
        frac_fp:   1D array of false positive rates at those thresholds
        frac_tp:   1D array of true positive rates at those thresholds
    """
    # Convert 'attack' to 0 or 1
    t_vals = attack_labels.astype(int)  # 1 if attack=True else 0

    # Evaluate f_model on each row
    coords = np.column_stack((alpha_scores, beta_scores))
    vals = np.array([f_model(x, y) for (x, y) in coords])

    # Overall count of positives (1) and negatives (0)
    n_pos = np.sum(t_vals == 1)
    n_neg = np.sum(t_vals == 0)

    t_min, t_max = np.min(vals), np.max(vals)
    thresholds = np.linspace(t_max, t_min, 101)

    frac_fp: list[float] = []
    frac_tp: list[float] = []

    for th in thresholds:
        in_region = vals >= th
        # false positive rate (fraction of negatives in region)
        fp_in_reg = np.sum((t_vals == 0) & in_region)
        frac_fp_val = fp_in_reg / n_neg

        # true positive rate (fraction of positives in region)
        tp_in_reg = np.sum((t_vals == 1) & in_region)
        frac_tp_val = tp_in_reg / n_pos

        if len(frac_fp) == 0:
            frac_fp.append(frac_fp_val)
            frac_tp.append(frac_tp_val)
            continue
        if frac_fp_val >= frac_fp[-1]:
            frac_fp.append(frac_fp_val)
        else:
            print(f"False positive rate {frac_fp_val} < {frac_fp[-1]}")
            frac_fp.append(frac_fp[-1])
        if frac_tp_val >= frac_tp[-1]:
            frac_tp.append(frac_tp_val)
        else:
            print(f"True positive rate {frac_tp_val} < {frac_tp[-1]}")
            frac_tp.append(frac_tp[-1])

    return thresholds, np.array(frac_fp), np.array(frac_tp)


def build_interpolators(
    thresholds: np.ndarray,  # will already be sorted
    frac_fp: np.ndarray,
    frac_tp: np.ndarray,
    tp_only: bool = False,
) -> Union[
    Tuple[Callable[[float], float], Callable[[float], float]], Callable[[float], float]
]:
    """
    Given arrays of thresholds, false positive rates, and true positive rates
    (aligned by index), build two 1D interpolators:

    1) fp_rate -> true_positive_rate
    2) fp_rate -> threshold

    We sort by false positive rate so the interpolators run in ascending order.

    Returns: (interp_tp_given_fp, interp_thresh_given_fp)
    """
    x_sorted = frac_fp
    y_sorted = frac_tp
    t_sorted = thresholds

    # Interpolation from false positive rate to true positive rate
    interp_tp_given_fp = interp.interp1d(
        x_sorted, y_sorted, bounds_error=False, fill_value=(y_sorted[0], y_sorted[-1])
    )
    if tp_only:
        return interp_tp_given_fp
    # Interpolation from false positive rate to threshold
    interp_thresh_given_fp = interp.interp1d(
        x_sorted, t_sorted, bounds_error=False, fill_value=(t_sorted[0], t_sorted[-1])
    )

    return interp_tp_given_fp, interp_thresh_given_fp


def training_function(
    fp_rate: float, interp_tp_given_fp: Callable[[float], float]
) -> float:
    """
    For a desired false positive rate = fp_rate, returns the true positive rate
    from the interpolation.
    """
    return float(interp_tp_given_fp(fp_rate))


def inference_function(
    a: float,
    b: float,
    fp_rate: float,
    interp_thresh_given_fp: Callable[[float], float],
    f_model: Callable[[float, float], float],
) -> bool:
    """
    For a new point (a,b), we get the threshold that corresponds to false positive rate = fp_rate,
    then return True if f_model(a,b) >= that threshold, else False.
    """
    t_fp = float(interp_thresh_given_fp(fp_rate))
    return f_model(a, b) >= t_fp


# %%
# Plotting functions


def plot_attack_scatter(
    score_df: pd.DataFrame,
    jitter_scale: float = 0.1,
    alpha: float = 0.5,
    titletext: str = "Scatter Plot of Scores by Attack Status",
    y_axis_label: str = "Beta Scores",
    x_axis_label: str = "Alpha Scores",
) -> None:
    """
    Create a scatter plot of alpha and beta scores,
    color-coded by attack status with optional jitter and transparency.

    Parameters:
    -----------
    score_df : pd.DataFrame
        DataFrame containing 'alpha_scores', 'beta_scores', and 'attack' columns
    jitter_scale : float, optional
        Amount of random noise to add to points (default 0.1)
    alpha : float, optional
        Transparency of points (default 0.5)
    """
    plt.figure(figsize=(10, 6), dpi=500)

    # Add small random jitter
    jittered_alpha = score_df["alpha_scores"] + np.random.normal(
        0, jitter_scale, len(score_df)
    )
    jittered_beta = score_df["beta_scores"] + np.random.normal(
        0, jitter_scale, len(score_df)
    )

    # Scatter plot with different colors for attacks
    plt.scatter(
        jittered_alpha[~score_df["attack"]],
        jittered_beta[~score_df["attack"]],
        c="green",
        label="Honest outputs",
        alpha=alpha,
    )
    plt.scatter(
        jittered_alpha[score_df["attack"]],
        jittered_beta[score_df["attack"]],
        c="red",
        label="Malign outputs",
        alpha=alpha,
    )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(titletext)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_local_odds_ratio(
    score_df: pd.DataFrame,
    bandwidth: float = 1.0,
    cmap: str = "viridis",
    titletext: str = "Kernel estimated odds ratio at a point",
    y_axis_label: str = "Beta Scores",
    x_axis_label: str = "Alpha Scores",
) -> None:
    """
    Create a scatter plot of alpha and beta scores,
    with color representing the local odds ratio estimate.

    Parameters:
    -----------
    score_df : pd.DataFrame
        DataFrame containing 'alpha_scores', 'beta_scores', and 'attack' columns
    bandwidth : float, optional
        Bandwidth for kernel estimation (default 1.0)
    cmap : str, optional
        Colormap for visualization (default 'viridis')
    """
    # Estimate local ratio
    ratio_values = estimate_local_ratio(
        score_df["alpha_scores"].to_numpy(),
        score_df["beta_scores"].to_numpy(),
        score_df["attack"].to_numpy(),
        bandwidth=bandwidth,
    )

    plt.figure(figsize=(12, 8), dpi=500)

    # Scatter plot with color representing local odds ratio
    scatter = plt.scatter(
        score_df["alpha_scores"],
        score_df["beta_scores"],
        c=ratio_values,
        cmap=cmap,
        alpha=0.7,
    )

    plt.colorbar(scatter, label="Local Odds Ratio")
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(
        f"{titletext} (Bandwidth: {bandwidth})"
    )  # Added note for bandwidth in the title
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_local_odds_ratio_binned(
    score_df: pd.DataFrame,
    bin_size: float = 2.0,
    cmap: str = "viridis",
    titletext: str = "Binned log odds ratio",
    y_axis_label: str = "Beta Scores",
    x_axis_label: str = "Alpha Scores",
) -> None:
    """
    Create a scatter plot of alpha and beta scores,
    with color representing the binned log odds ratio estimate.

    Parameters:
    -----------
    score_df : pd.DataFrame
        DataFrame containing 'alpha_scores', 'beta_scores', and 'attack' columns
    bin_size : float, optional
        Size of bins for calculating local odds ratio (default 2.0)
    cmap : str, optional
        Colormap for visualization (default 'viridis')
    titletext : str, optional
        Title for the plot
    y_axis_label : str, optional
        Label for the y-axis
    x_axis_label : str, optional
        Label for the x-axis
    """
    # Calculate bin edges
    if not (score_df["alpha_scores"].min() >= 0 and score_df["beta_scores"].min() >= 0):
        warnings.warn("Alpha and beta scores must be positive")
    alpha_min, alpha_max = 0, score_df["alpha_scores"].max()
    beta_min, beta_max = 0, score_df["beta_scores"].max()

    # Create bins
    alpha_bins = np.arange(alpha_min, alpha_max + bin_size, bin_size)
    beta_bins = np.arange(beta_min, beta_max + bin_size, bin_size)

    # Initialize array to store odds ratios
    log_odds_ratios = np.zeros((len(beta_bins) - 1, len(alpha_bins) - 1))

    # Calculate odds ratio for each bin
    for i in range(len(beta_bins) - 1):
        for j in range(len(alpha_bins) - 1):
            # Select points in this bin
            bin_mask = (
                (score_df["alpha_scores"] >= alpha_bins[j])
                & (score_df["alpha_scores"] < alpha_bins[j + 1])
                & (score_df["beta_scores"] >= beta_bins[i])
                & (score_df["beta_scores"] < beta_bins[i + 1])
            )

            bin_df = score_df[bin_mask]

            # Calculate odds ratio for this bin
            if len(bin_df) > 0:
                # Count attacks and non-attacks
                attacks = bin_df["attack"].sum()
                non_attacks = len(bin_df) - attacks

                # Avoid division by zero
                odds_ratio = attacks / (non_attacks + 1e-3) + 1e-3
                log_odds_ratios[i, j] = np.clip(np.log(odds_ratio), -10, 10)

    plt.figure(figsize=(12, 8), dpi=500)

    # Create a mesh grid for plotting
    alpha_centers = (alpha_bins[:-1] + alpha_bins[1:]) / 2
    beta_centers = (beta_bins[:-1] + beta_bins[1:]) / 2
    X, Y = np.meshgrid(alpha_centers, beta_centers)

    # Plot heatmap of odds ratios
    im = plt.pcolormesh(X, Y, log_odds_ratios, cmap=cmap, shading="auto")
    plt.colorbar(im, label="Binned Log Odds Ratio")

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(f"{titletext} (Bin Size: {bin_size})")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_roc_curve(
    score_df: pd.DataFrame, 
    f_quad: Callable[[float, float], float] = None,
    zoom_range: Tuple[float, float] = None
) -> None:
    """
    Plot ROC curves for:
    1. Combined monitor (quadratic interpolation) if f_quad is provided
    2. Alpha scores
    3. Beta scores

    Parameters:
    -----------
    score_df : pd.DataFrame
        DataFrame containing 'alpha_scores', 'beta_scores', and 'attack' columns
    f_quad : Callable, optional
        Quadratic interpolation function for combined monitor
    zoom_range : Tuple[float, float], optional
        Range of false positive rates to focus on (min_fpr, max_fpr)
        If provided, the plot will be zoomed to this range
    """
    plt.figure(figsize=(10, 8), dpi=500)

    # True labels
    y_true = score_df["attack"].to_numpy()

    # Plot ROC for alpha scores
    fpr_alpha, tpr_alpha, _ = roc_curve(y_true, score_df["alpha_scores"])
    roc_auc_alpha = auc(fpr_alpha, tpr_alpha)
    plt.plot(fpr_alpha, tpr_alpha, label=f"Alpha Scores (AUC = {roc_auc_alpha:.2f})")

    # Plot ROC for beta scores
    fpr_beta, tpr_beta, _ = roc_curve(y_true, score_df["beta_scores"])
    roc_auc_beta = auc(fpr_beta, tpr_beta)
    plt.plot(fpr_beta, tpr_beta, label=f"Beta Scores (AUC = {roc_auc_beta:.2f})")

    # Plot ROC for combined monitor if f_quad is provided
    if f_quad is not None:
        # Compute f_quad values for each point
        quad_scores = np.array(
            [
                f_quad(a, b)
                for a, b in zip(score_df["alpha_scores"], score_df["beta_scores"])
            ]
        )

        fpr_quad, tpr_quad, _ = roc_curve(y_true, quad_scores)
        roc_auc_quad = auc(fpr_quad, tpr_quad)
        plt.plot(
            fpr_quad, tpr_quad, label=f"Combined Monitor (AUC = {roc_auc_quad:.2f})"
        )

    # Plot random classifier reference line
    if zoom_range is None:
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
    else:
        # Set the zoom range
        min_fpr, max_fpr = zoom_range
        
        # Plot diagonal reference line only within zoom range
        plt.plot([min_fpr, max_fpr], [min_fpr, max_fpr], "k--", label="Random Classifier")
        
        # Set the axis limits according to zoom range
        plt.xlim([min_fpr, max_fpr])
        
        # For y-axis, find the max TPR values within the zoom range for each curve
        max_tpr_in_range = 0
        
        # Check alpha scores
        mask_alpha = (fpr_alpha >= min_fpr) & (fpr_alpha <= max_fpr)
        if np.any(mask_alpha):
            max_tpr_in_range = max(max_tpr_in_range, np.max(tpr_alpha[mask_alpha]))
        
        # Check beta scores
        mask_beta = (fpr_beta >= min_fpr) & (fpr_beta <= max_fpr)
        if np.any(mask_beta):
            max_tpr_in_range = max(max_tpr_in_range, np.max(tpr_beta[mask_beta]))
        
        # Check quad scores if available
        if f_quad is not None:
            mask_quad = (fpr_quad >= min_fpr) & (fpr_quad <= max_fpr)
            if np.any(mask_quad):
                max_tpr_in_range = max(max_tpr_in_range, np.max(tpr_quad[mask_quad]))
        
        # Set y-axis range with a small buffer, ensuring it doesn't exceed 1.05
        y_max = min(max_tpr_in_range + 0.05, 1.05)
        y_min = max(0, min_fpr - 0.05)  # Start slightly below diagonal line
        plt.ylim([y_min, y_max])
        
        # Add title indicating zoom
        plt.title(f"ROC Curve (Zoom: FPR {min_fpr:.4f}-{max_fpr:.4f})")
    
    if zoom_range is None:
        plt.title("Receiver Operating Characteristic (ROC) Curve")
    
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


def plot_monitor_decision_boundary_overlay(
    score_df: pd.DataFrame,
    f_model: Callable[[float, float], float],
    interp_thresh: Callable[[float], float],
    fp_rate: float = 0.1,
    jitter_scale: float = 0.1,
    titletext: str = "Monitor Scatter with Decision Boundary",
    x_axis_label: str = "Alpha Scores",
    y_axis_label: str = "Beta Scores",
) -> None:
    """
    Plot the scatter of alpha and beta scores (with optional jitter) along with
    the decision boundary defined by f_model(a, b) = interp_thresh(fp_rate).

    Points to the right of this boundary (i.e. points for which f_model(a,b)
    exceeds the threshold) are considered, for example, malign outputs.

    Parameters:
    -----------
    score_df : pd.DataFrame
        DataFrame containing 'alpha_scores', 'beta_scores', and 'attack' columns.
    f_model : Callable[[float, float], float]
        The fitted quadratic function, e.g. f_quad.
    interp_thresh : Callable[[float], float]
        Interpolator that maps a false positive rate to a threshold.
    fp_rate : float, default 0.1
        Desired false positive rate, used to extract the threshold.
    jitter_scale : float, default 0.1
        Amount of noise to add for better scatter plot visualization.
    titletext : str, default 'Monitor Scatter with Decision Boundary'
        Title of the plot.
    x_axis_label, y_axis_label : str
        Labels for the x and y axes.
    """

    # Get the threshold corresponding to the given false positive rate.
    t_thresh = float(interp_thresh(fp_rate))

    plt.figure(figsize=(10, 6), dpi=500)

    # Add jitter to the scatter plot points for better visualization.
    jittered_alpha = score_df["alpha_scores"] + np.random.normal(
        0, jitter_scale, len(score_df)
    )
    jittered_beta = score_df["beta_scores"] + np.random.normal(
        0, jitter_scale, len(score_df)
    )

    # Plot honest outputs in green and malign outputs in red.
    scatter_honest = plt.scatter(
        jittered_alpha[~score_df["attack"]],
        jittered_beta[~score_df["attack"]],
        c="green",
        label="Aligned Outputs",
        alpha=0.5,
    )
    scatter_malign = plt.scatter(
        jittered_alpha[score_df["attack"]],
        jittered_beta[score_df["attack"]],
        c="red",
        label="Misaligned Outputs",
        alpha=0.5,
    )

    # Define a grid spanning the range of your scores (with a small margin).
    a_min = score_df["alpha_scores"].min() - 1
    a_max = score_df["alpha_scores"].max() + 1
    b_min = score_df["beta_scores"].min() - 1
    b_max = score_df["beta_scores"].max() + 1

    a_vals = np.linspace(a_min, a_max, 300)
    b_vals = np.linspace(b_min, b_max, 300)
    A_grid, B_grid = np.meshgrid(a_vals, b_vals)

    # Evaluate f_model on the grid.
    Z = f_model(A_grid, B_grid)

    # Plot the contour line at the threshold value.
    cs = plt.contour(
        A_grid,
        B_grid,
        Z,
        levels=[t_thresh],
        colors="black",
        linewidths=2,
        linestyles="--",
    )

    # Instead of setting the contour label, create a custom legend entry.
    decision_boundary_line = Line2D(
        [],
        [],
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Decision Boundary: f(m,n) = {t_thresh:.2f}",
    )

    # Fetch existing handles/labels and append the decision boundary entry.
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(decision_boundary_line)
    labels.append(decision_boundary_line.get_label())

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(titletext)
    plt.legend(handles=handles, labels=labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


# %%
def run_two_monitor_interpolation(
    alpha_attack_path: str | Path = None,
    alpha_honest_path: str | Path = None,
    beta_attack_path: str | Path = None,
    beta_honest_path: str | Path = None,
    score_df: pd.DataFrame = None,
    bandwidth: float = 3.0,
    bin_size: float = 1.0,
    fp_rate: float = 0.05,
    jitter_scale: float = 0.1,
    x_axis_label: str = "Llama 31-70B Scores",
    y_axis_label: str = "Llama 31-8B Scores"
) -> tuple[Callable[[float, float], float], Callable[[float], float], Callable[[float], float]]:
    """
    Run the full two-monitor interpolation workflow and generate visualizations.
    
    Can either provide file paths to load data or directly pass a DataFrame with scores.
    
    Parameters:
    -----------
    alpha_attack_path : str | Path, optional
        Path to CSV file containing alpha scores for attack samples
    alpha_honest_path : str | Path, optional
        Path to CSV file containing alpha scores for honest samples
    beta_attack_path : str | Path, optional
        Path to CSV file containing beta scores for attack samples
    beta_honest_path : str | Path, optional
        Path to CSV file containing beta scores for honest samples
    score_df : pd.DataFrame, optional
        DataFrame with columns 'alpha_scores', 'beta_scores', and 'attack' (boolean)
        If provided, the file paths will be ignored
    bandwidth : float
        Bandwidth parameter for kernel density estimation
    bin_size : float
        Size of bins for odds ratio visualization
    fp_rate : float
        False positive rate for decision boundary
    jitter_scale : float
        Scale of jitter to add to scatter plots
    x_axis_label : str
        Label for x-axis in plots
    y_axis_label : str
        Label for y-axis in plots
    
    Returns:
    --------
    tuple[Callable, Callable, Callable]
        (f_quad, interp_tp, interp_thresh) - The fitted model and interpolators
    """
    # Check if score_df is provided
    if score_df is None:
        # Validate that all file paths are provided
        if any(path is None for path in [alpha_attack_path, alpha_honest_path, beta_attack_path, beta_honest_path]):
            raise ValueError("If score_df is not provided, all file paths must be specified")
        
        # Load data from files
        bcb_a_attack = pd.read_csv(alpha_attack_path)
        bcb_b_attack = pd.read_csv(beta_attack_path)
        bcb_a_honest = pd.read_csv(alpha_honest_path)
        bcb_b_honest = pd.read_csv(beta_honest_path)
        
        # Create a DataFrame with attack and honest samples
        new_df = pd.DataFrame({
            'alpha_scores': bcb_a_honest['score_fuzzed'].to_numpy(), 
            'beta_scores': bcb_b_honest['score_fuzzed'].to_numpy(), 
            'attack': [False for _ in range(len(bcb_b_honest))]
        })
        new_df2 = pd.DataFrame({
            'alpha_scores': bcb_a_attack['score_fuzzed'].to_numpy(), 
            'beta_scores': bcb_b_attack['score_fuzzed'].to_numpy(), 
            'attack': [True for _ in range(len(bcb_b_attack))]
        })
        score_df = pd.concat((new_df, new_df2)).reset_index(drop=True)
    else:
        # Validate that the score_df has the required columns
        required_columns = ['alpha_scores', 'beta_scores', 'attack']
        if not all(col in score_df.columns for col in required_columns):
            raise ValueError(f"score_df must contain the following columns: {required_columns}")

    # 1) Estimate local ratio
    ratio_est = estimate_local_ratio(
        score_df["alpha_scores"].to_numpy(),
        score_df["beta_scores"].to_numpy(),
        score_df["attack"].to_numpy(),
        bandwidth=bandwidth,
    )
    
    # 2) Fit a quadratic polynomial f(a,b)
    a_vals = score_df["alpha_scores"].to_numpy()
    b_vals = score_df["beta_scores"].to_numpy()
    f_quad, beta = fit_quadratic(a_vals, b_vals, ratio_est, True)

    # 3) Scan over thresholds & compute false/true positive rates
    thresholds, frac_fp, frac_tp = compute_threshold_curve(
        a_vals, b_vals, score_df["attack"].to_numpy(), f_quad
    )

    # 4) Build interpolators
    interp_tp, interp_thresh = build_interpolators(thresholds, frac_fp, frac_tp)

    # 5) Generate visualization plots
    plot_attack_scatter(
        score_df,
        titletext="Scatter Plot of Monitor Scores for Misaligned/Aligned Outputs",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )
    
    plot_local_odds_ratio_binned(
        score_df,
        bin_size=bin_size,
        titletext="Binned log odds ratio capped between -10 and 10",
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )
    
    plot_local_odds_ratio(
        score_df,
        bandwidth=bandwidth,
        titletext="Kernel estimated odds ratio at each data point",
        y_axis_label=y_axis_label,
        x_axis_label=x_axis_label,
    )
    
    plot_roc_curve(score_df, f_quad)
    plot_roc_curve(score_df, f_quad, zoom_range=(0, fp_rate))
    
    plot_monitor_decision_boundary_overlay(
        score_df,
        f_quad,
        interp_thresh,
        fp_rate=fp_rate,
        jitter_scale=jitter_scale,
        titletext=f"Decision Boundary for False Positive Rate of {fp_rate*100}%",
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
    )
    
    return f_quad, interp_tp, interp_thresh


# %%

if __name__ == "__main__":
    from pathlib import Path
    
    project_dir = Path(__file__).parent.parent
    alpha_attack_path = project_dir / "data" / "processed_monitor_scores" / "BCB2_llama3170B_5_attack.csv"
    alpha_honest_path = project_dir / "data" / "processed_monitor_scores" / "BCB_llama3170B_5_honest.csv"
    beta_attack_path = project_dir / "data" / "processed_monitor_scores" / "BCB2_llama318B_5_attack.csv"
    beta_honest_path = project_dir / "data" / "processed_monitor_scores" / "BCB_llama318B_5_honest.csv"
    
    f_quad, interp_tp, interp_thresh = run_two_monitor_interpolation(
        alpha_attack_path,
        alpha_honest_path,
        beta_attack_path,
        beta_honest_path,
        bandwidth=3,
        bin_size=1,
        fp_rate=0.05,
        jitter_scale=0.1,
        x_axis_label="Monitor M: Llama 3.1-70B Scores",
        y_axis_label="Monitor N: Llama 3.1-8B Scores"
    )

    #Now do it with gemma as weak monitor
    beta_attack_path = project_dir / "data" / "processed_monitor_scores" / "BCB2_gemma29B_5_attack.csv"
    beta_honest_path = project_dir / "data" / "processed_monitor_scores" / "BCB_gemma29B_5_honest.csv"
    
    f_quad, interp_tp, interp_thresh = run_two_monitor_interpolation(
        alpha_attack_path,
        alpha_honest_path,
        beta_attack_path,
        beta_honest_path,
        bandwidth=3,
        bin_size=1,
        fp_rate=0.05,
        jitter_scale=0.1,
        y_axis_label="Gemma 2-9B Scores",
        x_axis_label="Llama 31-70B Scores"
    )
    
    #now do it with gemma as alpha monitor and llama as beta monitor
    alpha_attack_path = project_dir / "data" / "processed_monitor_scores" / "BCB2_gemma29B_5_attack.csv"
    alpha_honest_path = project_dir / "data" / "processed_monitor_scores" / "BCB_gemma29B_5_honest.csv"
    beta_attack_path = project_dir / "data" / "processed_monitor_scores" / "BCB2_llama318B_5_attack.csv"
    beta_honest_path = project_dir / "data" / "processed_monitor_scores" / "BCB_llama318B_5_honest.csv"
    
    f_quad, interp_tp, interp_thresh = run_two_monitor_interpolation(
        alpha_attack_path,
        alpha_honest_path,
        beta_attack_path,
        beta_honest_path,
        bandwidth=3,
        bin_size=1,
        fp_rate=0.05,
        jitter_scale=0.1,
        y_axis_label="Llama 31-8B Scores",
        x_axis_label="Gemma 2-9B Scores"
    )


    simulated_scores = pd.read_csv(project_dir / "data" / "processed_monitor_scores" / "beta_sim_scores.csv")

    beta_score_df = pd.DataFrame({
        'alpha_scores': np.concatenate((simulated_scores['honest_scores'].to_numpy(), simulated_scores['attack_scores'].to_numpy())),
        'beta_scores': np.concatenate((simulated_scores['weaker_monitor_honest'].to_numpy(), simulated_scores['weaker_monitor_attack'].to_numpy())),
        'attack': np.concatenate([[False for _ in range(len(simulated_scores['honest_scores']))], [True for _ in range(len(simulated_scores['attack_scores']))]])
    })
    
    f_quad, interp_tp, interp_thresh = run_two_monitor_interpolation(
        score_df=beta_score_df,
        bandwidth=3,
        bin_size=1,
        fp_rate=0.05,
        jitter_scale=0,
        y_axis_label="Weaker Monitor Scores",
        x_axis_label="Stronger Monitor Scores"
    )



# %%
