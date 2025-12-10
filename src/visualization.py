"""
Utility functions for creating tables and visualisations for the empirical application.

Functions for:
- Summary tables in pandas/LaTeX format
- Overlap diagnostic plots
- κ_DML diagnostic plots

Publication-ready figure settings for The Econometrics Journal.

Author: Gabriel Saco
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple


# =============================================================================
# Publication-Quality Figure Settings
# =============================================================================

# Color palette for academic figures (colorblind-friendly)
COLORS = {
    'experimental': '#2C7BB6',  # Blue
    'observational': '#D7191C',  # Red  
    'LIN': '#1B9E77',           # Teal
    'LASSO': '#D95F02',         # Orange
    'RF': '#7570B3',            # Purple
    'benchmark': '#66A61E',     # Green
    'threshold': '#E7298A',     # Magenta
    'neutral': '#666666',       # Gray
}


def set_publication_style():
    """Set matplotlib rcParams for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (6.5, 4.5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'mathtext.fontset': 'cm',
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
    })


# =============================================================================
# Table Formatting
# =============================================================================

def results_to_dataframe(
    results: List[Dict[str, Any]],
    design_name: str = ""
) -> pd.DataFrame:
    """
    Convert a list of DML result dictionaries to a formatted DataFrame.
    
    Parameters
    ----------
    results : list of dict
        List of summary dictionaries from PLRDoubleMLKappa.summary_dict().
    design_name : str, optional
        Name of the design (e.g., 'Experimental', 'Observational').
        
    Returns
    -------
    pd.DataFrame
        Formatted results table.
    """
    df = pd.DataFrame(results)
    
    if design_name:
        df.insert(0, 'Design', design_name)
    
    # Rename columns for presentation
    rename_map = {
        'theta_hat': 'θ̂',
        'se_dml': 'SE',
        'ci_95_lower': 'CI Lower',
        'ci_95_upper': 'CI Upper',
        'ci_length': 'CI Length',
        'kappa_dml': 'κ_DML',
        'learner_m': 'Learner',
        'r_squared_d': 'R²(D|X)'
    }
    
    df = df.rename(columns=rename_map)
    
    # Select and order columns
    cols_order = ['Design', 'Learner', 'θ̂', 'SE', 'CI Lower', 'CI Upper', 
                  'CI Length', 'κ_DML', 'n', 'R²(D|X)']
    cols_order = [c for c in cols_order if c in df.columns]
    df = df[cols_order]
    
    return df


def combine_results_tables(
    results_dict: Dict[str, List[Dict[str, Any]]]
) -> pd.DataFrame:
    """
    Combine results from multiple designs into a single table.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary mapping design names to lists of result dictionaries.
        
    Returns
    -------
    pd.DataFrame
        Combined results table.
    """
    dfs = []
    for design_name, results in results_dict.items():
        df = results_to_dataframe(results, design_name=design_name)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)


def format_results_table(
    df: pd.DataFrame,
    float_format: str = '{:.3f}'
) -> pd.DataFrame:
    """
    Format a results DataFrame for display.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results table.
    float_format : str
        Format string for floating point numbers.
        
    Returns
    -------
    pd.DataFrame
        Formatted table (as strings).
    """
    df_fmt = df.copy()
    
    for col in df_fmt.columns:
        if df_fmt[col].dtype in [np.float64, np.float32]:
            df_fmt[col] = df_fmt[col].apply(lambda x: float_format.format(x) if pd.notna(x) else '')
    
    return df_fmt


def results_to_latex(
    df: pd.DataFrame,
    caption: str = "DML Estimation Results with $\\kappa_{\\mathrm{DML}}$ Diagnostic",
    label: str = "tab:dml_results",
    float_format: str = "%.2f"
) -> str:
    """
    Convert results DataFrame to publication-ready LaTeX table.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results table.
    caption : str
        Table caption.
    label : str
        LaTeX label.
    float_format : str
        Format for floats.
        
    Returns
    -------
    str
        LaTeX table code (uses booktabs package).
    """
    # Create a copy with formatted column names
    df_latex = df.copy()
    
    # Rename columns to LaTeX-friendly names
    col_rename = {
        'theta': r'$\hat{\theta}$',
        'se': 'SE',
        'ci_lower': 'CI Lower',
        'ci_upper': 'CI Upper',
        'ci_length': 'CI Length',
        'kappa_dml': r'$\kappa_{\mathrm{DML}}$',
        'theta_hat': r'$\hat{\theta}$',
        'se_dml': 'SE',
        'ci_95_lower': 'CI Lower',
        'ci_95_upper': 'CI Upper',
        'r_squared_d': r'$R^2(D|X)$',
    }
    df_latex = df_latex.rename(columns=col_rename)
    
    latex = df_latex.to_latex(
        index=False,
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
        column_format='l' * len(df_latex.columns)
    )
    
    # Replace unicode symbols with LaTeX
    latex = latex.replace('θ̂', r'$\hat{\theta}$')
    latex = latex.replace('κ_DML', r'$\kappa_{\mathrm{DML}}$')
    latex = latex.replace('R²(D|X)', r'$R^2(D|X)$')
    
    return latex


# =============================================================================
# Overlap Diagnostic Plots
# =============================================================================

def plot_propensity_histogram(
    e_hat: np.ndarray,
    D: np.ndarray,
    title: str = "Propensity Score Distribution",
    ax: Optional[plt.Axes] = None,
    bins: int = 30
) -> plt.Axes:
    """
    Plot histogram of propensity scores by treatment group.
    
    Parameters
    ----------
    e_hat : np.ndarray
        Estimated propensity scores.
    D : np.ndarray
        Treatment indicator.
    title : str
        Plot title.
    ax : plt.Axes, optional
        Axes to plot on.
    bins : int
        Number of histogram bins.
        
    Returns
    -------
    plt.Axes
        The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    D = np.asarray(D).ravel()
    
    ax.hist(e_hat[D == 0], bins=bins, alpha=0.5, label='Control', 
            density=True, color='steelblue')
    ax.hist(e_hat[D == 1], bins=bins, alpha=0.5, label='Treated', 
            density=True, color='darkorange')
    
    ax.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='e(X) = 0.5')
    ax.axvline(x=0.1, color='red', linestyle=':', alpha=0.5)
    ax.axvline(x=0.9, color='red', linestyle=':', alpha=0.5, label='Trimming bounds')
    
    ax.set_xlabel('Propensity Score ê(X)')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    
    return ax


def plot_overlap_comparison(
    overlap_results: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot propensity score distributions for multiple designs side by side.
    
    Parameters
    ----------
    overlap_results : dict
        Dictionary mapping design names to overlap diagnostic results
        (must include 'e_hat' and the D vector).
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        The figure.
    """
    n_designs = len(overlap_results)
    fig, axes = plt.subplots(1, n_designs, figsize=figsize)
    
    if n_designs == 1:
        axes = [axes]
    
    for ax, (design_name, results) in zip(axes, overlap_results.items()):
        e_hat = results['e_hat']
        D = results['D']
        plot_propensity_histogram(e_hat, D, title=f"{design_name}\n(n={len(D)})", ax=ax)
    
    plt.tight_layout()
    return fig


# =============================================================================
# κ_DML Diagnostic Plots
# =============================================================================

def plot_kappa_by_design(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot κ_DML by design and learner.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Combined results table with 'Design', 'Learner', and 'κ_DML' columns.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    designs = results_df['Design'].unique()
    learners = results_df['Learner'].unique()
    
    x = np.arange(len(designs))
    width = 0.25
    
    colors = {'LIN': 'steelblue', 'LASSO': 'darkorange', 'RF': 'forestgreen'}
    
    for i, learner in enumerate(learners):
        mask = results_df['Learner'] == learner
        kappa_values = []
        for design in designs:
            val = results_df[(results_df['Design'] == design) & 
                            (results_df['Learner'] == learner)]['κ_DML'].values
            kappa_values.append(val[0] if len(val) > 0 else 0)
        
        color = colors.get(learner, f'C{i}')
        ax.bar(x + i * width, kappa_values, width, label=learner, color=color, alpha=0.8)
    
    ax.set_xlabel('Design')
    ax.set_ylabel('κ_DML')
    ax.set_title('Condition Number κ_DML by Design and Learner')
    ax.set_xticks(x + width)
    ax.set_xticklabels(designs)
    ax.legend(title='Learner')
    
    # Add horizontal reference lines
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='κ = 1')
    ax.axhline(y=2, color='red', linestyle=':', alpha=0.5, label='κ = 2')
    
    plt.tight_layout()
    return fig


def plot_ci_length_vs_kappa(
    results_df: pd.DataFrame,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot CI length vs κ_DML, with points labelled by design/learner.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Combined results table.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    designs = results_df['Design'].unique()
    markers = {'Experimental': 'o', 'Observational': 's', 'Trimmed': '^'}
    colors = {'LIN': 'steelblue', 'LASSO': 'darkorange', 'RF': 'forestgreen'}
    
    for _, row in results_df.iterrows():
        design = row['Design']
        learner = row['Learner']
        kappa = row['κ_DML']
        ci_len = row['CI Length']
        
        marker = markers.get(design, 'o')
        color = colors.get(learner, 'gray')
        
        ax.scatter(kappa, ci_len, marker=marker, color=color, s=100, alpha=0.8,
                  edgecolors='black', linewidth=0.5)
        ax.annotate(f"{design[:3]}-{learner}", (kappa, ci_len), 
                   textcoords="offset points", xytext=(5, 5), fontsize=8)
    
    ax.set_xlabel('κ_DML')
    ax.set_ylabel('95% CI Length')
    ax.set_title('Confidence Interval Length vs. Condition Number')
    
    # Add reference line showing theoretical scaling
    kappa_range = np.linspace(results_df['κ_DML'].min() * 0.9, 
                               results_df['κ_DML'].max() * 1.1, 100)
    # CI length scales roughly as κ/√n
    n_typical = results_df['n'].mean()
    ax.plot(kappa_range, kappa_range / np.sqrt(n_typical) * 2, 
            'k--', alpha=0.3, label=r'$\propto \kappa/\sqrt{n}$')
    
    ax.legend()
    plt.tight_layout()
    return fig


def plot_theta_estimates(
    results_df: pd.DataFrame,
    benchmark: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot θ̂ estimates with confidence intervals by design and learner.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Combined results table.
    benchmark : float, optional
        Benchmark value to show (e.g., experimental estimate).
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        The figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {'LIN': 'steelblue', 'LASSO': 'darkorange', 'RF': 'forestgreen'}
    
    # Create x positions
    designs = results_df['Design'].unique()
    learners = results_df['Learner'].unique()
    
    y_positions = []
    y_labels = []
    
    for i, design in enumerate(designs):
        for j, learner in enumerate(learners):
            mask = (results_df['Design'] == design) & (results_df['Learner'] == learner)
            if mask.sum() > 0:
                row = results_df[mask].iloc[0]
                y_pos = i * (len(learners) + 1) + j
                y_positions.append(y_pos)
                y_labels.append(f"{design} / {learner}")
                
                color = colors.get(learner, 'gray')
                ax.errorbar(row['θ̂'], y_pos, 
                           xerr=[[row['θ̂'] - row['CI Lower']], 
                                 [row['CI Upper'] - row['θ̂']]],
                           fmt='o', color=color, capsize=5, markersize=8)
    
    if benchmark is not None:
        ax.axvline(x=benchmark, color='black', linestyle='--', alpha=0.5,
                  label=f'Benchmark = {benchmark:.2f}')
        ax.legend()
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('θ̂ (Treatment Effect Estimate)')
    ax.set_title('DML Estimates with 95% Confidence Intervals')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


# =============================================================================
# Summary Statistics
# =============================================================================

def print_summary_comparison(
    exp_df: pd.DataFrame,
    obs_df: pd.DataFrame
) -> None:
    """
    Print a comparison of summary statistics between experimental and 
    observational samples.
    
    Parameters
    ----------
    exp_df : pd.DataFrame
        Experimental sample.
    obs_df : pd.DataFrame
        Observational sample.
    """
    print("=" * 70)
    print("SAMPLE COMPARISON")
    print("=" * 70)
    
    print(f"\nExperimental Sample (n = {len(exp_df)}):")
    print(f"  Treated: {exp_df['D'].sum()}, Control: {(1 - exp_df['D']).sum():.0f}")
    
    print(f"\nObservational Sample (n = {len(obs_df)}):")
    print(f"  Treated: {obs_df['D'].sum()}, Control: {(1 - obs_df['D']).sum():.0f}")
    
    print("\n" + "-" * 70)
    print("Covariate Means by Treatment Status")
    print("-" * 70)
    
    covariates = ['age', 'education', 'black', 'hispanic', 'married', 
                  'nodegree', 're74', 're75']
    covariates = [c for c in covariates if c in exp_df.columns]
    
    print("\n{:<12} {:>10} {:>10} {:>10} {:>10}".format(
        "", "Exp-Treat", "Exp-Ctrl", "Obs-Treat", "Obs-Ctrl"))
    
    for cov in covariates:
        exp_treat = exp_df[exp_df['D'] == 1][cov].mean()
        exp_ctrl = exp_df[exp_df['D'] == 0][cov].mean()
        obs_treat = obs_df[obs_df['D'] == 1][cov].mean()
        obs_ctrl = obs_df[obs_df['D'] == 0][cov].mean()
        
        print("{:<12} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(
            cov, exp_treat, exp_ctrl, obs_treat, obs_ctrl))


if __name__ == "__main__":
    # Test plotting functions with dummy data
    print("Testing utility functions...")
    
    # Create dummy data
    np.random.seed(42)
    n = 100
    e_hat = np.random.beta(2, 5, n)
    D = (e_hat > 0.3).astype(int)
    
    # Test propensity plot
    fig, ax = plt.subplots()
    plot_propensity_histogram(e_hat, D, ax=ax)
    plt.close()
    
    print("Utility functions working correctly.")
