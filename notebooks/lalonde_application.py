# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python (dml)
#     language: python
#     name: dml
# ---

# %% [markdown]
# # LaLonde Application: The κ* Diagnostic in Canonical Data
#
# ---
#
# ## Research Goal
#
# This notebook applies the **κ* diagnostic** to the canonical LaLonde (1986) job training 
# dataset. We demonstrate the practical use of κ* as a routine diagnostic for DML estimation.
#
# ## Theoretical Background
#
# From Theorem 3.2, the **exact decomposition**:
#
# $$\hat{\theta} - \theta_0 = \hat{\kappa} \cdot S_n' + \hat{\kappa} \cdot B_n'$$
#
# The standardized condition number $\kappa^* = 1/(1 - R^2(D|X))$ amplifies both:
# - $S_n'$: standardized oracle sampling term (variance inflation)
# - $B_n'$: standardized nuisance bias term (bias amplification)
#
# ---

# %% [markdown]
# ## 1. Setup and Imports

# %%
# Standard imports
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

# Add project root to path
sys.path.insert(0, '..')

# Import from src modules
from src.data import (
    load_lalonde, 
    get_sample_summary,
    EXPERIMENTAL_BENCHMARK,
)
from src.learners import get_learner, LALONDE_LEARNERS
from src.dml import DMLEstimator
from src.tuning import tune_rf_for_data

# Academic matplotlib settings (matching Monte Carlo figures)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 11,
    'mathtext.fontset': 'stix',
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'axes.linewidth': 0.8,
    'legend.fontsize': 9,
    'legend.framealpha': 1.0,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})

# Output paths
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

print("Setup complete.")
print("=" * 60)
print("LALONDE APPLICATION: κ* Diagnostic in Canonical Data")
print("=" * 60)
print(f"Experimental Benchmark ATE: ${EXPERIMENTAL_BENCHMARK:,}")

# %% [markdown]
# ## 2. Load Data

# %%
# Load experimental data (NSW treated vs NSW control)
y_exp, d_exp, X_exp = load_lalonde(mode='experimental', standardize=True)
summary_exp = get_sample_summary(y_exp, d_exp, X_exp)
print(f"\nExperimental Sample: N={len(y_exp)}")
print(f"  Treated: {int(d_exp.sum())}, Control: {int(len(d_exp) - d_exp.sum())}")
print(f"  Naive ATE: ${summary_exp['naive_ate']:,.0f}")

# %%
# Load observational data (NSW treated vs PSID control)
y_obs, d_obs, X_obs = load_lalonde(mode='observational', standardize=True)
summary_obs = get_sample_summary(y_obs, d_obs, X_obs)
print(f"\nObservational Sample: N={len(y_obs)}")
print(f"  Treated: {int(d_obs.sum())}, Control: {int(len(d_obs) - d_obs.sum())}")
print(f"  Naive ATE: ${summary_obs['naive_ate']:,.0f}")
print("\n⚠️ Note: The observational naive ATE is negative due to selection bias!")

# %% [markdown]
# ## 3. Tune RF Hyperparameters

# %%
print("\nTuning RF hyperparameters for LaLonde data...")

print("  Tuning on Experimental sample...", end=" ")
rf_params_exp = tune_rf_for_data(X_exp, d_exp, random_state=RANDOM_STATE)
print(f"Done. Best: {rf_params_exp}")

print("  Tuning on Observational sample...", end=" ")
rf_params_obs = tune_rf_for_data(X_obs, d_obs, random_state=RANDOM_STATE)
print(f"Done. Best: {rf_params_obs}")

# %% [markdown]
# ## 4. Run DML Across Learners

# %%
# Learners to evaluate
LEARNERS = LALONDE_LEARNERS
print(f"\nLearners for comparison: {LEARNERS}")

# DML settings
K_FOLDS = 5
N_REPEATS = 3

# %%
def run_dml_for_sample(y, d, X, sample_name, learners=LEARNERS, rf_params=None):
    """Run DML with multiple learners on a single sample."""
    results = []
    
    for learner_name in tqdm(learners, desc=f"{sample_name}"):
        params = rf_params if learner_name.upper() == 'RF_TUNED' else None
        learner_m = get_learner(learner_name, random_state=RANDOM_STATE, params=params)
        learner_l = get_learner(learner_name, random_state=RANDOM_STATE, params=params)
        
        dml = DMLEstimator(
            learner_m=learner_m,
            learner_l=learner_l,
            K=K_FOLDS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )
        
        result = dml.fit(Y=y, D=d, X=X)
        
        results.append({
            'Sample': sample_name,
            'Learner': learner_name,
            'Estimate': result.theta_hat,
            'SE': result.se,
            'CI_Lower': result.ci_lower,
            'CI_Upper': result.ci_upper,
            'Kappa_Star': result.kappa_star,
            'N': len(y),
            'N_eff': result.effective_sample_size,
        })
    
    return pd.DataFrame(results)

# %%
# Run DML on experimental sample
print("\nRunning DML on Experimental Sample...")
df_exp = run_dml_for_sample(y_exp, d_exp, X_exp, 'Experimental', rf_params=rf_params_exp)
print("\nExperimental Results:")
print(df_exp.round(2).to_string())

# %%
# Run DML on observational sample  
print("\nRunning DML on Observational Sample...")
df_obs = run_dml_for_sample(y_obs, d_obs, X_obs, 'Observational', rf_params=rf_params_obs)
print("\nObservational Results:")
print(df_obs.round(2).to_string())

# %%
# Combine results
df_baseline = pd.concat([df_exp, df_obs], ignore_index=True)

# Save to CSV
baseline_path = RESULTS_DIR / 'lalonde_baseline_results.csv'
df_baseline.to_csv(baseline_path, index=False)
print(f"\nBaseline results saved to: {baseline_path}")

# %% [markdown]
# ## 5. Conditioning Analysis

# %%
print("\n" + "=" * 60)
print("CONDITIONING ANALYSIS")
print("=" * 60)

for sample_name, df_sample in [('Experimental', df_exp), ('Observational', df_obs)]:
    mean_kappa = df_sample['Kappa_Star'].mean()
    n = df_sample['N'].iloc[0]
    n_eff = df_sample['N_eff'].mean()
    
    print(f"\n{sample_name} Sample:")
    print(f"  Mean κ* = {mean_kappa:.2f}")
    print(f"  N = {n:,}")
    print(f"  N_eff = N/κ* = {n_eff:.0f} ({n_eff/n*100:.1f}% of nominal)")

# Estimate dispersion
exp_range = df_exp['Estimate'].max() - df_exp['Estimate'].min()
obs_range = df_obs['Estimate'].max() - df_obs['Estimate'].min()
print(f"\n\nEstimate Dispersion Across Learners:")
print(f"  Experimental: ${exp_range:,.0f}")
print(f"  Observational: ${obs_range:,.0f}")
print(f"  Ratio: {obs_range/exp_range:.1f}x more dispersion in observational sample")

# %% [markdown]
# ---
#
# ## 6. Forest Plot: Experimental vs Observational
#
# **Academic-quality visualization** comparing DML estimates across learners and samples.

# %%
# =============================================================================
# FOREST PLOT - ACADEMIC STYLE (Matching Monte Carlo Figures)
# =============================================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Academic color palette - muted, professional
COLORS = {
    'Experimental': '#2ca02c',     # Muted green
    'Observational': '#d62728',    # Muted red
    'benchmark': '#1f77b4',        # Muted blue
}

fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor('white')

# Prepare data - order by sample then learner
df_plot = df_baseline.copy()
df_plot['y_pos'] = range(len(df_plot))

# Add separator between samples
n_exp = len(df_exp)

# Plot confidence intervals and point estimates
for idx, row in df_plot.iterrows():
    color = COLORS[row['Sample']]
    y = row['y_pos']
    
    # CI line
    ax.hlines(y=y, xmin=row['CI_Lower'], xmax=row['CI_Upper'], 
              color=color, linewidth=2, alpha=0.7)
    
    # Point estimate marker
    ax.scatter(row['Estimate'], y, color=color, s=80, zorder=5, 
               edgecolors='black', linewidth=0.5, marker='o')
    
    # κ* annotation (right side)
    ax.annotate(f"κ*={row['Kappa_Star']:.2f}", 
                xy=(row['CI_Upper'] + 200, y),
                fontsize=8, color='gray', va='center')

# Experimental benchmark reference line
ax.axvline(x=EXPERIMENTAL_BENCHMARK, color=COLORS['benchmark'], 
           linestyle='--', linewidth=1.5, alpha=0.8, zorder=1)

# Zero reference line  
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5, zorder=0)

# Horizontal separator between samples
ax.axhline(y=n_exp - 0.5, color='black', linestyle='-', linewidth=0.5, alpha=0.3)

# Y-axis labels
y_labels = [f"{row['Learner']}" for _, row in df_plot.iterrows()]
ax.set_yticks(df_plot['y_pos'])
ax.set_yticklabels(y_labels, fontsize=10)

# Add sample group labels on left
ax.text(-4500, (n_exp-1)/2, 'Experimental', fontsize=11, fontweight='bold',
        va='center', ha='right', color=COLORS['Experimental'])
ax.text(-4500, n_exp + (len(df_obs)-1)/2, 'Observational', fontsize=11, fontweight='bold',
        va='center', ha='right', color=COLORS['Observational'])

# Axis formatting
ax.set_xlabel('Treatment Effect Estimate ($)', fontsize=11)
ax.set_xlim(-4000, 5000)
ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
ax.invert_yaxis()
ax.tick_params(axis='both', which='major', labelsize=10)

# Simple square legend box (academic style)
legend_elements = [
    Patch(facecolor=COLORS['Experimental'], edgecolor='black', linewidth=0.5,
          label=f'Experimental (κ* ≈ {df_exp["Kappa_Star"].mean():.1f})'),
    Patch(facecolor=COLORS['Observational'], edgecolor='black', linewidth=0.5,
          label=f'Observational (κ* ≈ {df_obs["Kappa_Star"].mean():.1f})'),
    Line2D([0], [0], color=COLORS['benchmark'], linestyle='--', linewidth=1.5,
           label=f'Benchmark (${EXPERIMENTAL_BENCHMARK:,})'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, 
          framealpha=1.0, edgecolor='black', fancybox=False)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lalonde_forest_plot.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(RESULTS_DIR / 'lalonde_forest_plot.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✓ Saved: {RESULTS_DIR / 'lalonde_forest_plot.pdf'}")
plt.show()

# %% [markdown]
# ## 7. Summary Statistics

# %%
print("\n" + "=" * 80)
print("SUMMARY: LALONDE DIAGNOSTIC ANALYSIS")
print("=" * 80)

# Key metrics
kappa_exp_mean = df_exp['Kappa_Star'].mean()
kappa_obs_mean = df_obs['Kappa_Star'].mean()
n_exp = df_exp['N'].iloc[0]
n_obs = df_obs['N'].iloc[0]
n_eff_exp = df_exp['N_eff'].mean()
n_eff_obs = df_obs['N_eff'].mean()

print(f"\n{'Metric':<25} {'Experimental':>15} {'Observational':>15}")
print("-" * 55)
print(f"{'N (sample size)':<25} {n_exp:>15,} {n_obs:>15,}")
print(f"{'Mean κ*':<25} {kappa_exp_mean:>15.2f} {kappa_obs_mean:>15.2f}")
print(f"{'Mean N_eff':<25} {n_eff_exp:>15,.0f} {n_eff_obs:>15,.0f}")
print(f"{'N_eff / N':<25} {n_eff_exp/n_exp*100:>14.1f}% {n_eff_obs/n_obs*100:>14.1f}%")
print(f"{'Estimate dispersion':<25} ${exp_range:>13,.0f} ${obs_range:>13,.0f}")

print(f"\n\n→ Dispersion ratio (Obs/Exp): {obs_range/exp_range:.1f}x")
print(f"→ Higher κ* in observational sample corresponds to greater learner disagreement")

# %%
# Save final results
df_baseline.to_csv(RESULTS_DIR / 'lalonde_results.csv', index=False)

print("\n" + "=" * 60)
print("LALONDE ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nResults saved to: {RESULTS_DIR}")
print("  - lalonde_results.csv")
print("  - lalonde_baseline_results.csv")
print("  - lalonde_forest_plot.pdf / .png")
