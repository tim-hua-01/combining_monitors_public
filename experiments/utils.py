import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Any

def plot_safety_pareto_curves(
    x_values: list[float],
    labels: list[str],
    in_sample_safeties: Optional[list[list[float]]] = None,
    oos_safeties: Optional[list[list[float]]] = None,
    x_label: str = 'Budget (Y)',
    y_label: str = 'Attack Detection Rate %',
    in_sample_title: str = 'In-Sample Safety vs. Budget',
    oos_title: str = 'Out-of-Sample Safety vs. Budget',
    colors: Optional[list[str]] = None,
    invert_x: bool = True,
    save_dir: Optional[str] = None,
    filename: Optional[str] = None,
    plot_style_settings: Optional[dict[str, Any]] = None
) -> None:
    """
    Plot safety Pareto curves with optional in-sample and out-of-sample subplots.
    
    Args:
        x_values: X-axis values
        labels: Labels for each safety array (should match length of safety arrays)
        in_sample_safeties: List of numpy arrays for in-sample safety values (optional)
        oos_safeties: List of numpy arrays for out-of-sample safety values (optional)
        x_label: Label for x-axis
        y_label: Label for y-axis
        in_sample_title: Title for in-sample subplot
        oos_title: Title for out-of-sample subplot
        colors: Optional list of colors for different lines
        invert_x: Whether to invert the x-axis (default: True)
        save_dir: Directory to save the plot (if None, won't save)
        filename: Filename for saving (if None, won't save)
        plot_style_settings: Dictionary of matplotlib style settings
    """
    # Default style settings
    default_style = {
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    }
    
    # Update with custom settings if provided
    if plot_style_settings:
        default_style.update(plot_style_settings)
    
    # Apply style settings
    plt.rcParams.update(default_style)
    
    # Default colors if not provided
    default_colors = ['#38A5E0', '#2ECC71', '#8000FF', '#FF5733']
    if not colors:
        colors = default_colors
    
    # Default markers and line styles
    markers = ['o', 's', '^', 'D']
    
    # Determine number of subplots needed
    n_plots = 0
    if in_sample_safeties is not None:
        n_plots += 1
    if oos_safeties is not None:
        n_plots += 1
    
    if n_plots == 0:
        raise ValueError("At least one of in_sample_safeties or oos_safeties must be provided")
    
    # Create figure and axes
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
        axes = [ax]
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        axes = [ax1, ax2]
    
    # Plot in-sample safety if provided
    if in_sample_safeties is not None:
        for i, safety in enumerate(in_sample_safeties):
            color_idx = i % len(colors)
            marker_idx = i % len(markers)
            axes[0].plot(
                x_values, 
                [i * 100 for i in safety], 
                f'{markers[marker_idx]}-', 
                color=colors[color_idx],
                linewidth=2,
                markersize=8,
                label=labels[i]
            )
        
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel(y_label)
        axes[0].set_title(in_sample_title, fontweight='bold', pad=15)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(frameon=True, framealpha=0.9)
        if invert_x:
            axes[0].invert_xaxis()
    
    # Plot out-of-sample safety if provided
    if oos_safeties is not None:
        ax_idx = 0 if in_sample_safeties is None else 1
        for i, safety in enumerate(oos_safeties):
            color_idx = i % len(colors)
            marker_idx = i % len(markers)
            axes[ax_idx].plot(
                x_values, 
                [i * 100 for i in safety], 
                f'{markers[marker_idx]}-', 
                color=colors[color_idx],
                linewidth=2,
                markersize=8,
                label=labels[i]
            )
        
        axes[ax_idx].set_xlabel(x_label)
        axes[ax_idx].set_ylabel(y_label)
        axes[ax_idx].set_title(oos_title, fontweight='bold', pad=15)
        axes[ax_idx].grid(True, alpha=0.3)
        axes[ax_idx].legend(frameon=True, framealpha=0.9)
        if invert_x:
            axes[ax_idx].invert_xaxis()
    
    # Adjust layout
    plt.tight_layout(pad=3.0)
    
    # Save plot if directory is provided
    if save_dir and filename:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show plot if no filename is provided
    if filename is None:
        plt.show()