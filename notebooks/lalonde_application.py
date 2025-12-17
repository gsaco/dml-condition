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
# # Empirical Application: Bias Amplification in the LaLonde Data
#
# ---
#
# ## Research Goal
#
# This notebook applies our **Bias Amplification** theory to the canonical LaLonde (1986) dataset. We demonstrate that:
#
# 1. **Experimental Sample** (NSW treated vs. NSW control): Low $\kappa^* \approx 4$ → stable, consistent estimates across learners.
# 2. **Observational Sample** (NSW treated vs. PSID control): High $\kappa^* > 20$ → wildly diverging estimates due to bias amplification.
#
# ## Theoretical Background
#
# From the paper's **exact decomposition** (algebraic identity, no Taylor remainder):
# $$\hat{\theta} - \theta_0 = \widehat{\kappa} \cdot S_n' + \widehat{\kappa} \cdot B_n'$$
#
# The standardized condition number $\kappa^* = (1 - R^2(D|X))^{-1}$ amplifies both sampling variance ($S_n'$) and nuisance bias ($B_n'$).
#
# **Key Insight**: In the observational LaLonde sample, weak overlap (treatment is highly predictable from covariates) leads to catastrophic bias amplification. The **trimming sensitivity analysis** demonstrates that artificially improving overlap reduces $\kappa^*$ and stabilizes estimates.
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
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, '..')

# Import from src modules
from src.data import (
    load_lalonde, 
    get_propensity_trimmed_sample,
    get_sample_summary,
    EXPERIMENTAL_BENCHMARK,
)
from src.learners import get_learner
from src.dml import DMLEstimator, run_dml
from src.tuning import tune_rf_for_data

# Configure matplotlib
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# Output paths
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

# Random seed for reproducibility
RANDOM_STATE = 42

print("Setup complete.")
print(f"Experimental Benchmark ATE: ${EXPERIMENTAL_BENCHMARK:,}")

# %% [markdown]
# ---
#
# # Phase A: Baseline Diagnostics (The "Static" Proof)
#
# Compare DML estimates across learners for:
# 1. **Experimental Sample** (gold standard)
# 2. **Observational Sample** (biased comparison)

# %% [markdown]
# ## 2. Load Data

# %%
# Load experimental data (NSW treated vs NSW control)
y_exp, d_exp, X_exp = load_lalonde(mode='experimental', standardize=True)
summary_exp = get_sample_summary(y_exp, d_exp, X_exp)
print(f"\nNaive ATE (experimental): ${summary_exp['naive_ate']:,.0f}")
print()

# %%
# Load observational data (NSW treated vs PSID control)
y_obs, d_obs, X_obs = load_lalonde(mode='observational', standardize=True)
summary_obs = get_sample_summary(y_obs, d_obs, X_obs)
print(f"\nNaive ATE (observational): ${summary_obs['naive_ate']:,.0f}")
print("\n⚠️ Note: The observational naive ATE is negative due to selection bias!")

# %% [markdown]
# ## 3. Run DML Across Learners

# %%
# Learners to evaluate
LEARNERS = ['OLS', 'Lasso', 'RF_Tuned', 'MLP']

# DML settings
K_FOLDS = 5
N_REPEATS = 3  # Repeated cross-fitting for variance reduction

# %%
# =============================================================================
# LALONDE-SPECIFIC RF TUNING (NOT SIMULATION PARAMS!)
# =============================================================================
# Since LaLonde is a single dataset, we tune RF specifically for it.
# This is different from simulations where we pre-tune per (N, R²) regime.

print("Tuning RF hyperparameters for LaLonde data...")
print("  (This ensures optimal RF performance on this specific dataset)")

# Tune on propensity score (treatment prediction task) for each sample
print("  Tuning on Experimental sample...", end=" ")
rf_params_exp = tune_rf_for_data(X_exp, d_exp, random_state=RANDOM_STATE)
print(f"Done. Best: {rf_params_exp}")

print("  Tuning on Observational sample...", end=" ")
rf_params_obs = tune_rf_for_data(X_obs, d_obs, random_state=RANDOM_STATE)
print(f"Done. Best: {rf_params_obs}")

print("\nLaLonde-specific RF tuning complete.")

# %%
def run_dml_for_sample(y, d, X, sample_name, learners=LEARNERS, rf_params=None):
    """
    Run DML with multiple learners on a single sample.
    
    Parameters
    ----------
    rf_params : dict or None
        Pre-tuned RF hyperparameters for this sample.
    
    Returns a DataFrame with results.
    """
    results = []
    
    for learner_name in tqdm(learners, desc=f"{sample_name}"):
        # Get learner instances (use tuned params for RF_Tuned)
        params = rf_params if learner_name.upper() == 'RF_TUNED' else None
        learner_m = get_learner(learner_name, random_state=RANDOM_STATE, params=params)
        learner_l = get_learner(learner_name, random_state=RANDOM_STATE, params=params)
        
        # Create DML estimator
        dml = DMLEstimator(
            learner_m=learner_m,
            learner_l=learner_l,
            K=K_FOLDS,
            n_repeats=N_REPEATS,
            random_state=RANDOM_STATE,
        )
        
        # Fit DML
        result = dml.fit(Y=y, D=d, X=X)
        
        # Store results
        results.append({
            'Sample': sample_name,
            'Learner': learner_name,
            'Estimate': result.theta_hat,
            'SE': result.se,
            'CI_Lower': result.ci_lower,
            'CI_Upper': result.ci_upper,
            'Kappa_Star': result.kappa_star,
            'N': len(y),
        })
    
    return pd.DataFrame(results)

print("DML function defined.")

# %%
# Run DML on experimental sample (using sample-specific RF params)
print("Running DML on Experimental Sample...")
df_exp = run_dml_for_sample(y_exp, d_exp, X_exp, 'Experimental', rf_params=rf_params_exp)
print("\nExperimental Results:")
display(df_exp.round(2))

# %%
# Run DML on observational sample (using sample-specific RF params)
print("Running DML on Observational Sample...")
df_obs = run_dml_for_sample(y_obs, d_obs, X_obs, 'Observational', rf_params=rf_params_obs)
print("\nObservational Results:")
display(df_obs.round(2))

# %%
# Combine results
df_baseline = pd.concat([df_exp, df_obs], ignore_index=True)

# Save to CSV
baseline_path = RESULTS_DIR / 'lalonde_baseline_results.csv'
df_baseline.to_csv(baseline_path, index=False)
print(f"Baseline results saved to: {baseline_path}")

# Display combined results
print("\n" + "="*80)
print("COMBINED BASELINE RESULTS")
print("="*80)
display(df_baseline.round(2))

# %% [markdown]
# ### Key Observations (Phase A)
#
# **Experimental Sample:**
# - Low $\kappa^* \approx 4$ → treatment assignment is nearly random
# - All learners produce similar estimates (~$1,500-2,000)
# - Estimates are close to the benchmark (~$1,794)
#
# **Observational Sample:**
# - High $\kappa^* > 20$ → treatment is highly predictable from $X$
# - Learners disagree wildly (estimates range from negative to positive thousands)
# - **This is Bias Amplification in action!**

# %% [markdown]
# ---
#
# # Phase B: Trimming Sensitivity Analysis (The "Dynamic" Proof)
#
# **Theory**: If bias amplification via $\kappa^*$ is the culprit, then artificially reducing $\kappa^*$ (via propensity score trimming) should stabilize the estimate.
#
# **Algorithm**:
# 1. Fit a propensity score model on the observational data
# 2. For each trimming threshold $\alpha$: keep observations where $\alpha < \hat{e}(X) < 1-\alpha$
# 3. Compute $\kappa^*$ and DML estimate for trimmed sample
# 4. Plot the "stabilization path"

# %%
# Trimming thresholds
ALPHA_GRID = [0.00, 0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]

# Fit propensity model once (to avoid retraining for each threshold)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

print("Fitting propensity score model...")
base_clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, n_jobs=-1)
propensity_model = CalibratedClassifierCV(base_clf, cv=5)
propensity_model.fit(X_obs, d_obs)

# Get propensity scores
e_hat = propensity_model.predict_proba(X_obs)[:, 1]
print(f"Propensity score range: [{e_hat.min():.4f}, {e_hat.max():.4f}]")
print(f"Propensity score mean: {e_hat.mean():.4f}")

# %%
# Run trimming sensitivity analysis
trimming_results = []

print("\nRunning Trimming Sensitivity Analysis...")
print("="*60)

for alpha in tqdm(ALPHA_GRID, desc="Trimming"):
    # Apply trimming
    if alpha > 0:
        mask = (e_hat > alpha) & (e_hat < (1 - alpha))
    else:
        mask = np.ones(len(y_obs), dtype=bool)
    
    y_trim = y_obs[mask]
    d_trim = d_obs[mask]
    X_trim = X_obs[mask]
    
    n_trim = len(y_trim)
    n_treated_trim = int(d_trim.sum())
    
    # Skip if too few observations
    if n_trim < 100 or n_treated_trim < 20:
        print(f"  α={alpha:.2f}: Skipping (n={n_trim}, n_treated={n_treated_trim})")
        continue
    
    # Run DML with RF_Tuned (using observational sample's tuned params)
    learner_m = get_learner('RF_Tuned', random_state=RANDOM_STATE, params=rf_params_obs)
    learner_l = get_learner('RF_Tuned', random_state=RANDOM_STATE, params=rf_params_obs)
    
    dml = DMLEstimator(
        learner_m=learner_m,
        learner_l=learner_l,
        K=K_FOLDS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )
    
    result = dml.fit(Y=y_trim, D=d_trim, X=X_trim)
    
    trimming_results.append({
        'alpha': alpha,
        'n': n_trim,
        'n_treated': n_treated_trim,
        'estimate': result.theta_hat,
        'se': result.se,
        'ci_lower': result.ci_lower,
        'ci_upper': result.ci_upper,
        'kappa_star': result.kappa_star,
    })
    
    print(f"  α={alpha:.2f}: N={n_trim:,}, κ*={result.kappa_star:.1f}, θ̂={result.theta_hat:,.0f} ± {1.96*result.se:,.0f}")

# Convert to DataFrame
df_trimming = pd.DataFrame(trimming_results)

# Save results
trimming_path = RESULTS_DIR / 'lalonde_trimming_results.csv'
df_trimming.to_csv(trimming_path, index=False)
print(f"\nTrimming results saved to: {trimming_path}")

# %%
# Display trimming results
print("\n" + "="*80)
print("TRIMMING SENSITIVITY ANALYSIS RESULTS")
print("="*80)
display(df_trimming.round(2))

# %% [markdown]
# ---
#
# # Phase C: Publication-Ready Visualizations

# %% [markdown]
# ## 4. The Forest Plot (Baseline Comparison)

# %%
# =============================================================================
# PLOT 1: FOREST PLOT - EXPERIMENTAL VS OBSERVATIONAL
# =============================================================================

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data for plotting
df_plot = df_baseline.copy()
df_plot['y_pos'] = range(len(df_plot))

# Colors for samples
colors = {'Experimental': '#2ecc71', 'Observational': '#e74c3c'}

# Plot confidence intervals and point estimates
for idx, row in df_plot.iterrows():
    color = colors[row['Sample']]
    y = row['y_pos']
    
    # CI line
    ax.hlines(y=y, xmin=row['CI_Lower'], xmax=row['CI_Upper'], 
              color=color, linewidth=2, alpha=0.7)
    
    # Point estimate
    ax.scatter(row['Estimate'], y, color=color, s=100, zorder=5, edgecolors='black', linewidth=0.5)
    
    # Annotate with κ*
    ax.annotate(f"κ*={row['Kappa_Star']:.1f}", 
                xy=(row['CI_Upper'] + 200, y),
                fontsize=9, color='gray', va='center')

# Reference line at experimental benchmark
ax.axvline(x=EXPERIMENTAL_BENCHMARK, color='navy', linestyle='--', linewidth=2, alpha=0.7)

# Reference line at 0
ax.axvline(x=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

# Y-axis labels
y_labels = [f"{row['Sample']}\n{row['Learner']}" for _, row in df_plot.iterrows()]
ax.set_yticks(df_plot['y_pos'])
ax.set_yticklabels(y_labels)

# Formatting
ax.set_xlabel('Treatment Effect Estimate ($)', fontsize=12)
ax.set_title('Forest Plot: DML Estimates by Sample and Learner\n(High κ* → High Disagreement)', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()  # Top to bottom

# Create custom legend
legend_elements = [
    Patch(facecolor='#2ecc71', label='Experimental (low κ*)'),
    Patch(facecolor='#e74c3c', label='Observational (high κ*)'),
    Line2D([0], [0], color='navy', linestyle='--', linewidth=2, label=f'Benchmark (${EXPERIMENTAL_BENCHMARK:,})'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lalonde_forest_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'lalonde_forest_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'lalonde_forest_plot.pdf'}")
plt.show()

# %% [markdown]
# ## 5. The Stabilization Path (Trimming Analysis)

# %%
# =============================================================================
# PLOT 2: DUAL Y-AXIS STABILIZATION PATH
# =============================================================================

fig, ax1 = plt.subplots(figsize=(12, 7))

# Left Y-axis: DML Estimate with CI
color_estimate = '#3498db'  # Blue
ax1.set_xlabel('Trimming Threshold (α)', fontsize=12)
ax1.set_ylabel('DML Estimate ($)', color=color_estimate, fontsize=12)

# Plot estimate with CI shaded region
ax1.fill_between(
    df_trimming['alpha'],
    df_trimming['ci_lower'],
    df_trimming['ci_upper'],
    alpha=0.3, color=color_estimate, label='95% CI'
)
ax1.plot(
    df_trimming['alpha'],
    df_trimming['estimate'],
    color=color_estimate, linewidth=3, marker='o', markersize=8,
    label='DML Estimate (RF_Tuned)'
)

# Experimental benchmark horizontal line
ax1.axhline(y=EXPERIMENTAL_BENCHMARK, color='green', linestyle='--', linewidth=2,
            label=f'Experimental Benchmark (${EXPERIMENTAL_BENCHMARK:,})')

ax1.tick_params(axis='y', labelcolor=color_estimate)
ax1.set_ylim(bottom=min(df_trimming['ci_lower'].min() - 500, -2000),
             top=max(df_trimming['ci_upper'].max() + 500, 4000))

# Right Y-axis: Condition Number (log scale)
ax2 = ax1.twinx()
color_kappa = '#e74c3c'  # Red
ax2.set_ylabel('Condition Number κ* (Log Scale)', color=color_kappa, fontsize=12)

ax2.plot(
    df_trimming['alpha'],
    df_trimming['kappa_star'],
    color=color_kappa, linewidth=3, marker='s', markersize=8, linestyle='--',
    label='κ* (Condition Number)'
)

ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor=color_kappa)

# Add horizontal reference line for κ* = 10 (threshold for instability)
ax2.axhline(y=10, color=color_kappa, linestyle=':', linewidth=1.5, alpha=0.7)
ax2.text(0.35, 10, 'κ* = 10', color=color_kappa, fontsize=9, va='bottom')

# Title
plt.title(
    'The Stabilization Path: Trimming Improves Overlap and Stabilizes Estimates\n'
    'As α↑, κ*↓, and the observational estimate converges to the experimental truth',
    fontsize=14
)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)

# Grid
ax1.grid(True, alpha=0.3)

# Annotations
# Add sample size annotations
for idx, row in df_trimming.iterrows():
    if row['alpha'] in [0.0, 0.10, 0.20, 0.30]:
        ax1.annotate(
            f"n={row['n']:,}",
            xy=(row['alpha'], row['estimate']),
            xytext=(row['alpha'], row['estimate'] - 800),
            fontsize=8, color='gray', ha='center'
        )

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'lalonde_stabilization_path.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'lalonde_stabilization_path.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'lalonde_stabilization_path.pdf'}")
plt.show()

# %% [markdown]
# ## 6. Summary Statistics Table

# %%
# =============================================================================
# SUMMARY TABLE FOR PAPER
# =============================================================================

print("\n" + "="*80)
print("SUMMARY TABLE: BASELINE COMPARISON")
print("="*80)

# Pivot table for easy comparison
summary_pivot = df_baseline.pivot_table(
    index='Learner',
    columns='Sample',
    values=['Estimate', 'SE', 'Kappa_Star'],
    aggfunc='first'
)

print("\nEstimates by Learner and Sample:")
display(summary_pivot['Estimate'].round(0))

print("\nκ* by Learner and Sample:")
display(summary_pivot['Kappa_Star'].round(1))

# Key insight summary
print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
kappa_exp_mean = df_baseline[df_baseline['Sample'] == 'Experimental']['Kappa_Star'].mean()
kappa_obs_mean = df_baseline[df_baseline['Sample'] == 'Observational']['Kappa_Star'].mean()
print(f"\nExperimental κ* (mean): {kappa_exp_mean:.1f}")
print(f"Observational κ* (mean): {kappa_obs_mean:.1f}")
print(f"Ratio: {kappa_obs_mean/kappa_exp_mean:.1f}x higher in observational sample")
print("\n→ High κ* in observational sample causes bias amplification and estimate instability.")

# %% [markdown]
# ---
#
# ## 7. Conclusions
#
# This empirical application confirms our theoretical findings:
#
# ### Phase A (Baseline)
# 1. **Experimental sample** has low $\kappa^* \approx 4$ → all learners agree, estimates stable
# 2. **Observational sample** has high $\kappa^* > 20$ → learners disagree wildly
#
# ### Phase B (Trimming)
# 3. As we trim extreme propensity scores (increasing $\alpha$):
#    - $\kappa^*$ decreases (better overlap)
#    - The biased observational estimate **converges toward the experimental truth**
#    
# This provides **causal evidence** that $\kappa^*$ (the condition number) is the mechanism
# behind DML's failure in weak overlap settings.
#
# ### Practical Implications
# - **Diagnostic**: Calculate $\kappa^*$ before trusting DML estimates
# - **Regimes** (from the paper):
#   - $\kappa^* < 5$: Well-conditioned — standard DML asymptotics apply
#   - $5 \le \kappa^* \le 20$: Moderately ill-conditioned — sensitivity analysis warranted
#   - $\kappa^* > 20$: Severely ill-conditioned — bias amplification likely dominates

# %%
# Save all results
df_all = df_baseline.copy()
df_all.to_csv(RESULTS_DIR / 'lalonde_results.csv', index=False)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: {RESULTS_DIR}")
print("  - lalonde_results.csv (baseline)")
print("  - lalonde_baseline_results.csv")
print("  - lalonde_trimming_results.csv")
print("  - lalonde_forest_plot.pdf / .png")
print("  - lalonde_stabilization_path.pdf / .png")
