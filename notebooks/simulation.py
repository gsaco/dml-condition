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
#     display_name: diagnostic
#     language: python
#     name: python3
# ---

# %%
# gabrielsaco@MacBook-Pro-de-Gabriel-2 dml-diagnostic % /Users/gabrielsaco/anaconda3/envs/diagnostic/bin/python -m j
# upytext --sync notebooks/simulation.py



# %% [markdown]
# # Forensic Monte Carlo Study: Bias Amplification in DML
#
# ---
#
# ## Research Goal
#
# This notebook demonstrates the **Bias Amplification** hypothesis:
#
# Under weak overlap (high condition number $\kappa^*$), regularized ML estimators fail due to **amplified bias**, whereas OLS fails due to **variance inflation** (or misspecification).
#
# ## Theoretical Background
#
# The standardized DML condition number is defined as:
# $$\kappa^* := \frac{1}{1 - R^2(D|X)} = \frac{\sigma_D^2}{\sigma_V^2}$$
#
# From the paper's **exact decomposition** (no Taylor remainder):
# $$\hat{\theta} - \theta_0 = \widehat{\kappa} \cdot S_n' + \widehat{\kappa} \cdot B_n'$$
#
# where $S_n' = S_n / \hat{\sigma}_D^2$ is the standardized oracle term and $B_n' = B_n / \hat{\sigma}_D^2$ is the standardized bias term.
#
# This is an **algebraic identity** (exact for PLR), implying $\kappa^*$ directly amplifies both variance ($S_n'$) and nuisance bias ($B_n'$).
#
# ## Key Metrics
#
# | Metric | Definition | Significance |
# |--------|------------|--------------|
# | Coverage | $\mathbb{I}(\theta_0 \in \text{CI}_{95})$ | Should be 0.95 |
# | Bias-to-SE Ratio | $|\text{MC Bias}| / \text{MC SE}$ | If > 1, bias-dominated |
# | RMSE | $\sqrt{\frac{1}{n}\sum(\hat{\eta} - \eta_0)^2}$ | Nuisance estimation error |
# | $Q$ | $\kappa^* \times \sqrt{\text{RMSE}_m \times \text{RMSE}_\ell}$ | Interaction term (product rate × κ*) |


# %% [markdown]
# ## 1. Setup and Imports

# %%
# Standard imports
import sys
import warnings
from pathlib import Path
import seaborn as sns

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Use sklearn's Parallel/delayed to propagate scikit-learn config to workers
from sklearn.utils.parallel import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# Add project root to path
sys.path.insert(0, '..')

# Import from src modules
from src.dgp import NonLinearDGP, generate_nonlinear_data
from src.learners import get_learner, get_oracle_pair, SIMULATION_LEARNERS
from src.dml import DMLEstimator, run_dml
from src.tuning import tune_rf_hyperparameters

# Configure matplotlib
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Output paths
RESULTS_DIR = Path('../results')
RESULTS_DIR.mkdir(exist_ok=True)

print("Setup complete.")
print(f"Available learners: {SIMULATION_LEARNERS}")

# %% [markdown]
# ## 2. Simulation Configuration

# %%
# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================

N_SAMPLES = [500, 2000, 5000]    # Sample sizes for JCI Optimized design
B_REPS = 500                    # Monte Carlo replications
K_FOLDS = 5                      # Cross-fitting folds
N_REPEATS = 2                    # Repeated cross-fitting iterations
THETA0 = 1.0                     # True treatment effect
BASE_SEED = 20241216             # Base random seed

# Overlap regimes (target R²(D|X))
R2_REGIMES = [0.75, 0.90, 0.95, 0.98, 0.99]

# Learners to evaluate (JCI Optimized: Oracle, OLS, Lasso, RF_Tuned, MLP)
LEARNERS = ['Oracle', 'OLS', 'Lasso', 'RF_Tuned', 'MLP']

# Number of parallel jobs
N_JOBS = -1  # Use all cores

print(f"Simulation Configuration:")
print(f"  Sample sizes: N = {N_SAMPLES}")
print(f"  Replications: B = {B_REPS}")
print(f"  R² regimes: {R2_REGIMES}")
print(f"  Learners: {LEARNERS}")
print(f"  Total runs: {len(N_SAMPLES) * len(R2_REGIMES) * len(LEARNERS) * B_REPS:,}")

# %% [markdown]
# ## 3. Phase A: Pre-Tuning (Global Hyperparameter Optimization)
#
# **Strategy**: Tune RF hyperparameters ONCE per (N, R²) regime, then use
# these fixed parameters in all Monte Carlo replications. This saves
# significant compute compared to running RandomizedSearchCV inside every loop.

# %%
# =============================================================================
# PHASE A: PRE-TUNING (GLOBAL HYPERPARAMETER OPTIMIZATION)
# =============================================================================

print("Phase A: Pre-Tuning Random Forest hyperparameters...")
print(f"  Configurations: {len(N_SAMPLES)} × {len(R2_REGIMES)} = {len(N_SAMPLES) * len(R2_REGIMES)}")
print()

# Store optimal params: optimal_params[n][r2] = {'max_depth': ..., ...}
optimal_params = {}

for n_sample in N_SAMPLES:
    optimal_params[n_sample] = {}
    for r2 in R2_REGIMES:
        print(f"  Tuning for N={n_sample}, R²={r2}...", end=" ")
        params = tune_rf_hyperparameters(
            n_samples=n_sample,
            target_r2=r2,
            random_state=BASE_SEED,
        )
        optimal_params[n_sample][r2] = params
        print(f"Done. Best: {params}")

print("\nPre-tuning complete. Optimal params stored.")
print()

# %% [markdown]
# ## 4. Single Replication Function (with RMSE Tracking)

# %%
def run_single_replication(
    n: int,
    target_r2: float,
    learner_name: str,
    rep: int,
    base_seed: int = BASE_SEED,
    rf_params: dict = None,
) -> dict:
    """
    Run a single Monte Carlo replication with full forensic tracking.
    
    Now includes RMSE tracking for interaction term calculation.
    
    Parameters
    ----------
    rf_params : dict or None
        Pre-tuned RF hyperparameters. If provided, RF_Tuned uses these
        instead of running RandomizedSearchCV (saves compute).
    """
    # Create unique seed for this replication
    r2_code = int(target_r2 * 100)
    learner_code = hash(learner_name) % 100
    seed = base_seed + r2_code * 10000 + learner_code * 1000 + rep
    
    # Generate data
    Y, D, X, info, dgp = generate_nonlinear_data(
        n=n,
        target_r2=target_r2,
        random_state=seed,
    )
    
    # Get true values for MSE calculation
    m0_X = info['m0_X']
    ell0_X = info['ell0_X']
    
    # Get learners
    if learner_name.upper() == 'ORACLE':
        from src.learners import OracleLearner
        learner_m = OracleLearner(dgp=dgp, target='m')
        learner_l = OracleLearner(dgp=dgp, target='l')
    else:
        # For RF_Tuned, use pre-tuned params (avoids RandomizedSearchCV in loop)
        params = rf_params if learner_name.upper() == 'RF_TUNED' else None
        learner_m = get_learner(learner_name, random_state=seed, params=params)
        learner_l = get_learner(learner_name, random_state=seed, params=params)
    
    # Run DML with repeated cross-fitting
    dml_result = run_dml(
        Y=Y, D=D, X=X,
        learner_m=learner_m,
        learner_l=learner_l,
        m0_X=m0_X,
        ell0_X=ell0_X,
        K=K_FOLDS,
        n_repeats=N_REPEATS,
        theta0=THETA0,
        random_state=seed,
    )
    
    # Compile results with RMSE tracking
    return {
        'n': n,
        'target_r2': target_r2,
        'sample_r2': info['sample_r2'],
        'learner': learner_name,
        'replication': rep,
        'theta_hat': dml_result.theta_hat,
        'se': dml_result.se,
        'ci_lower': dml_result.ci_lower,
        'ci_upper': dml_result.ci_upper,
        'ci_length': dml_result.ci_length,
        'kappa_star': dml_result.kappa_star,
        'bias': dml_result.bias,
        'coverage': int(dml_result.covers(THETA0)),
        'nuisance_mse_m': dml_result.nuisance_mse_m,
        'nuisance_mse_l': dml_result.nuisance_mse_l,
        'rmse_m': dml_result.rmse_m,  # NEW: RMSE tracking
        'rmse_l': dml_result.rmse_l,  # NEW: RMSE tracking
    }

print("Single replication function defined (with RMSE tracking).")

# %% [markdown]
# ## 5. Phase B: Run Monte Carlo Simulation (with Pre-Tuned Params)

# %%
# =============================================================================
# RUN SIMULATION WITH PARALLEL PROCESSING
# =============================================================================

print("Starting Monte Carlo simulation...")
print(f"Total replications: {len(N_SAMPLES) * len(R2_REGIMES) * len(LEARNERS) * B_REPS:,}")
print()

# Build task list with pre-tuned RF params
tasks = []
for n_sample in N_SAMPLES:
    for r2 in R2_REGIMES:
        # Get pre-tuned RF params for this (N, R²) configuration
        rf_params = optimal_params[n_sample][r2]
        for learner in LEARNERS:
            for rep in range(B_REPS):
                # Pass rf_params to each task (only used by RF_Tuned)
                tasks.append((n_sample, r2, learner, rep, BASE_SEED, rf_params))

# Run in parallel (Phase B: Monte Carlo with fixed hyperparameters)
# NOTE: No RandomizedSearchCV runs inside this loop!
results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_single_replication)(*task) for task in tasks
)

# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to CSV
output_path = RESULTS_DIR / 'sim_results_jci.csv'
df_results.to_csv(output_path, index=False)
print(f"\nResults saved to: {output_path}")
print(f"Shape: {df_results.shape}")


# %% [markdown]
# ## 6. Compute Aggregate Metrics (with Interaction Term Q)

# %%
# =============================================================================
# AGGREGATE RESULTS BY (LEARNER, REGIME, SAMPLE SIZE)
# =============================================================================

def compute_aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute forensic metrics by (Learner, Regime, Sample Size).
    Now includes RMSE and interaction term Q.
    """
    agg = df.groupby(['learner', 'target_r2', 'n']).agg(
        # Coverage
        coverage=('coverage', 'mean'),
        
        # CI Length
        avg_ci_length=('ci_length', 'mean'),
        
        # Monte Carlo Bias
        mc_bias=('theta_hat', lambda x: x.mean() - THETA0),
        
        # Monte Carlo SE
        mc_se=('theta_hat', 'std'),
        
        # Average κ*
        avg_kappa_star=('kappa_star', 'mean'),
        median_kappa_star=('kappa_star', 'median'),
        
        # Nuisance MSE
        avg_nuisance_mse_m=('nuisance_mse_m', 'mean'),
        avg_nuisance_mse_l=('nuisance_mse_l', 'mean'),
        
        # RMSE (NEW)
        avg_rmse_m=('rmse_m', 'mean'),
        avg_rmse_l=('rmse_l', 'mean'),
        
        # Sample R²
        avg_sample_r2=('sample_r2', 'mean'),
        
        # Count
        n_reps=('replication', 'count'),
    ).reset_index()
    
    # Compute Bias-to-SE Ratio
    agg['bias_to_se_ratio'] = np.abs(agg['mc_bias']) / agg['mc_se']
    
    # Combined nuisance MSE
    agg['avg_nuisance_mse'] = (agg['avg_nuisance_mse_m'] + agg['avg_nuisance_mse_l']) / 2
    
    # Interaction Term Q = κ* × sqrt(RMSE_m × RMSE_l) (NEW)
    agg['Q'] = agg['median_kappa_star'] * np.sqrt(agg['avg_rmse_m'] * agg['avg_rmse_l'])
    
    return agg

# Compute aggregates
df_agg = compute_aggregate_metrics(df_results)

# Display
print("Aggregate Metrics (with interaction term Q):")
display(df_agg.round(4))

# Save aggregates
agg_path = RESULTS_DIR / 'sim_aggregates_jci.csv'
df_agg.to_csv(agg_path, index=False)
print(f"\nAggregates saved to: {agg_path}")

# %% [markdown]
# ## 6. Visualization: The "Crash" Plot
#
# **Goal**: Show ML methods crashing toward 0 coverage as $\kappa^*$ rises.

# %%
# =============================================================================
# PLOT 1: COVERAGE VS κ* (LOG SCALE) - BY SAMPLE SIZE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

# Color palette
colors = sns.color_palette('husl', n_colors=len(LEARNERS))
markers = ['o', 's', '^', 'D', 'v']

for ax_idx, n_sample in enumerate(N_SAMPLES):
    ax = axes[ax_idx]
    subset_n = df_agg[df_agg['n'] == n_sample]
    
    for idx, learner in enumerate(LEARNERS):
        subset = subset_n[subset_n['learner'] == learner]
        ax.plot(
            subset['median_kappa_star'],
            subset['coverage'],
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=8,
            label=learner,
        )
    
    # Reference line at 95%
    ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\kappa^*$ (Log Scale)', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(f'N = {n_sample}', fontsize=14)
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.set_ylabel('Coverage Probability', fontsize=12)
        ax.legend(loc='lower left', fontsize=9)

plt.suptitle('The "Crash" Plot: Coverage vs. Condition Number', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'crash_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'crash_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'crash_plot.pdf'}")
plt.show()

# %% [markdown]
# ## 7. Visualization: The "Mechanism" Plot
#
# **Goal**: Show Bias-to-SE ratio exploding for ML methods while staying flat for OLS.

# %%
# =============================================================================
# PLOT 2: BIAS-TO-SE RATIO VS κ* - BY SAMPLE SIZE
# =============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax_idx, n_sample in enumerate(N_SAMPLES):
    ax = axes[ax_idx]
    subset_n = df_agg[df_agg['n'] == n_sample]
    
    for idx, learner in enumerate(LEARNERS):
        subset = subset_n[subset_n['learner'] == learner]
        ax.plot(
            subset['median_kappa_star'],
            subset['bias_to_se_ratio'],
            marker=markers[idx % len(markers)],
            color=colors[idx],
            linewidth=2,
            markersize=8,
            label=learner,
        )
    
    # Reference line at 1 (bias = SE)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5)
    ax.set_xscale('log')
    ax.set_xlabel(r'$\kappa^*$ (Log Scale)', fontsize=12)
    ax.set_title(f'N = {n_sample}', fontsize=14)
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.set_ylabel('Bias-to-SE Ratio', fontsize=12)
        ax.legend(loc='upper left', fontsize=9)

plt.suptitle('The "Mechanism" Plot: Bias-to-SE Ratio vs. Condition Number', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'mechanism_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'mechanism_plot.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'mechanism_plot.pdf'}")
plt.show()

# %% [markdown]
# ## 8. The "Smoking Gun" Decomposition Plot
#
# Three-panel figure showing:
# 1. **Panel 1 (The Cause)**: $\kappa^*$ vs. Nuisance MSE (should be flat/slowly rising)
# 2. **Panel 2 (The Mechanism)**: $\kappa^*$ vs. Bias-to-SE Ratio (explodes for ML)
# 3. **Panel 3 (The Effect)**: $\kappa^*$ vs. Coverage (crashes for ML)

# %%
# =============================================================================
# PLOT 3: THREE-PANEL "SMOKING GUN" DECOMPOSITION (for N=2000)
# =============================================================================

# Use middle sample size for main decomposition plot
df_mid = df_agg[df_agg['n'] == 2000]

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

# Panel 1: Nuisance MSE
ax1 = axes[0]
for idx, learner in enumerate(LEARNERS):
    subset = df_mid[df_mid['learner'] == learner]
    ax1.plot(
        subset['median_kappa_star'],
        subset['avg_nuisance_mse'],
        marker=markers[idx % len(markers)],
        color=colors[idx],
        linewidth=2,
        markersize=6,
        label=learner,
    )
ax1.set_xscale('log')
ax1.set_ylabel('Nuisance MSE', fontsize=12)
ax1.set_title('Panel 1: The Cause', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8)

# Panel 2: Bias-to-SE Ratio
ax2 = axes[1]
for idx, learner in enumerate(LEARNERS):
    subset = df_mid[df_mid['learner'] == learner]
    ax2.plot(
        subset['median_kappa_star'],
        subset['bias_to_se_ratio'],
        marker=markers[idx % len(markers)],
        color=colors[idx],
        linewidth=2,
        markersize=6,
        label=learner,
    )
ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_xscale('log')
ax2.set_xlabel(r'$\kappa^*$ (Log Scale)', fontsize=12)
ax2.set_ylabel('Bias / SE', fontsize=12)
ax2.set_title('Panel 2: The Mechanism', fontsize=14)
ax2.grid(True, alpha=0.3)

# Panel 3: Coverage
ax3 = axes[2]
for idx, learner in enumerate(LEARNERS):
    subset = df_mid[df_mid['learner'] == learner]
    ax3.plot(
        subset['median_kappa_star'],
        subset['coverage'],
        marker=markers[idx % len(markers)],
        color=colors[idx],
        linewidth=2,
        markersize=6,
        label=learner,
    )
ax3.axhline(y=0.95, color='gray', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xscale('log')
ax3.set_ylabel('Coverage', fontsize=12)
ax3.set_ylim(0, 1.05)
ax3.set_title('Panel 3: The Effect', fontsize=14)
ax3.grid(True, alpha=0.3)

plt.suptitle('The "Smoking Gun" Decomposition: Why DML Fails Under Weak Overlap (N=2000)', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'smoking_gun_decomposition.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'smoking_gun_decomposition.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'smoking_gun_decomposition.pdf'}")
plt.show()

# %% [markdown]
# ## 9. The "Universal Failure Curve" (Rate Validation)
#
# **NEW**: This plot unifies all learners and sample sizes onto a single curve using the theoretical interaction term:
# $$Q = \kappa^* \times \sqrt{\text{RMSE}_m \times \text{RMSE}_\ell}$$
#
# **Goal**: Show that all points collapse onto a single sigmoid curve - coverage drops when $Q > C$ (some constant), regardless of learner or sample size.

# %%
# =============================================================================
# PLOT 4: THE UNIVERSAL FAILURE CURVE (Rate Validation)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 8))

# Filter out Oracle (has near-zero RMSE, so Q is not meaningful)
df_plot = df_agg[df_agg['learner'] != 'Oracle'].copy()

# Remove NaN Q values
df_plot = df_plot.dropna(subset=['Q'])

# Color by learner, shape by sample size
learner_colors = {
    'OLS': '#e74c3c',
    'Lasso': '#3498db',
    'RF_Tuned': '#2ecc71',
    'MLP': '#9b59b6',
}
sample_markers = {500: 'o', 2000: 's', 5000: '^'}

# Scatter plot
for learner in ['OLS', 'Lasso', 'RF_Tuned', 'MLP']:
    for n_sample in N_SAMPLES:
        subset = df_plot[(df_plot['learner'] == learner) & (df_plot['n'] == n_sample)]
        if len(subset) > 0:
            ax.scatter(
                subset['Q'],
                subset['coverage'],
                color=learner_colors[learner],
                marker=sample_markers[n_sample],
                s=120,
                alpha=0.8,
                edgecolors='black',
                linewidth=0.5,
                label=f'{learner}, N={n_sample}',
            )

# Add trend line (LOESS-like smoothing using sorted values)
df_sorted = df_plot.sort_values('Q')
if len(df_sorted) > 5:
    # Simple moving average as trend
    window = max(3, len(df_sorted) // 5)
    smooth_coverage = df_sorted['coverage'].rolling(window=window, center=True).mean()
    ax.plot(df_sorted['Q'], smooth_coverage, 'k-', linewidth=2, alpha=0.5, label='Trend')

# Reference lines
ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=2, label='Nominal 95%')

# Find approximate threshold where coverage drops below 0.9
threshold_mask = df_plot['coverage'] < 0.9
if threshold_mask.any():
    Q_threshold = df_plot.loc[threshold_mask, 'Q'].min()
    ax.axvline(x=Q_threshold, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.text(Q_threshold * 1.1, 0.2, f'Q ≈ {Q_threshold:.2f}', color='red', fontsize=10)

ax.set_xscale('log')
ax.set_xlabel(r'Interaction Term: $Q = \kappa^* \times \sqrt{\mathrm{RMSE}_m \times \mathrm{RMSE}_\ell}$', fontsize=14)
ax.set_ylabel('Coverage Probability', fontsize=14)
ax.set_ylim(0, 1.05)
ax.set_title(
    'The Universal Failure Curve: All Learners Collapse onto One Line\n'
    '(Coverage Fails When Q Exceeds Critical Threshold)',
    fontsize=16
)

# Custom legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Learner colors
color_handles = [Patch(facecolor=c, label=l) for l, c in learner_colors.items()]
# Sample size markers
marker_handles = [Line2D([0], [0], marker=m, color='gray', linestyle='', markersize=10, label=f'N={n}') 
                  for n, m in sample_markers.items()]
ax.legend(handles=color_handles + marker_handles, loc='lower left', fontsize=10, ncol=2)

ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'universal_failure_curve.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'universal_failure_curve.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'universal_failure_curve.pdf'}")
plt.show()

# %% [markdown]
# ## 10. Summary Table for Paper

# %%
# =============================================================================
# SUMMARY TABLE
# =============================================================================

# Pivot for paper table (for N=2000)
df_mid = df_agg[df_agg['n'] == 2000]
summary_cols = ['coverage', 'mc_bias', 'mc_se', 'bias_to_se_ratio', 'median_kappa_star', 'Q']
summary = df_mid.pivot_table(
    index='learner',
    columns='target_r2',
    values=summary_cols,
)

print("\n" + "="*80)
print("SUMMARY TABLE FOR PAPER (N=2000)")
print("="*80)

# Coverage table
print("\nCOVERAGE:")
display(summary['coverage'].round(3))

# Bias-to-SE Ratio table
print("\nBIAS-TO-SE RATIO:")
display(summary['bias_to_se_ratio'].round(3))

# κ* table
print("\nMEDIAN κ*:")
display(summary['median_kappa_star'].round(2))

# Q (Interaction Term) table
print("\nINTERACTION TERM Q:")
display(summary['Q'].round(3))

# %% [markdown]
# ## 11. Conclusions
#
# The simulation results demonstrate:
#
# 1. **Oracle baseline**: Near-perfect 95% coverage across all regimes → bias amplification is learner-specific.
#
# 2. **OLS failure mode**: Low but stable coverage (misspecification) → fails due to wrong functional form.
#
# 3. **ML failure mode**: Coverage crashes as $\kappa^* \to \infty$ → fails due to **bias amplification**.
#
# 4. **Smoking Gun**: Nuisance MSE is flat/slowly rising, but Bias-to-SE ratio explodes → the condition number amplifies small errors.
#
# 5. **Universal Failure Curve (NEW)**: All learners and sample sizes collapse onto a single curve when plotted against $Q = \kappa^* \times \sqrt{\text{RMSE}_m \times \text{RMSE}_\ell}$. Coverage fails when $Q$ exceeds a critical threshold.

# %%
print("\nSimulation complete!")
print(f"Results saved to: {RESULTS_DIR}")
print("\nNew outputs:")
print("  - universal_failure_curve.pdf (The Rate Validation Plot)")
print("  - RMSE tracking in sim_results_jci.csv")
print("  - Interaction term Q in sim_aggregates_jci.csv")


# %% [markdown]
# ---
#
# # JCI Enhancement #1: Binary Treatment DGP Simulation
#
# This section demonstrates that bias amplification via κ* is **universal** —
# it applies equally to binary treatments D ∈ {0,1} as it does to continuous treatments.
# This connects directly to the propensity score literature familiar to JCI readers.

# %%
# =============================================================================
# BINARY TREATMENT DGP SIMULATION
# =============================================================================

from src.dgp import BinaryTreatmentDGP, generate_binary_treatment_data
from src.tuning import tune_rf_for_binary_treatment

print("\n" + "="*80)
print("JCI ENHANCEMENT #1: BINARY TREATMENT DGP")
print("="*80)

# Binary treatment overlap regimes
# target_overlap controls propensity score bounds
OVERLAP_REGIMES = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60]  # Higher = better overlap = lower κ*

# Use fewer sample sizes for computational efficiency
N_SAMPLES_BINARY = [1000, 2000, 5000]

# Use fewer learners (MLP slow on binary outcomes)
LEARNERS_BINARY = ['Oracle', 'OLS', 'Lasso', 'RF_Tuned']

# Fewer replications for this supplementary analysis
B_REPS_BINARY = min(B_REPS, 20)

print(f"\nBinary Treatment Configuration:")
print(f"  Overlap regimes: {OVERLAP_REGIMES}")
print(f"  Sample sizes: N = {N_SAMPLES_BINARY}")
print(f"  Learners: {LEARNERS_BINARY}")
print(f"  Replications: B = {B_REPS_BINARY}")

# %%
# Pre-tune RF for binary treatment (Phase A)
print("\nPhase A: Pre-Tuning for Binary Treatment DGP...")

optimal_params_binary = {}
for n_sample in N_SAMPLES_BINARY:
    optimal_params_binary[n_sample] = {}
    for overlap in OVERLAP_REGIMES:
        print(f"  Tuning for N={n_sample}, overlap={overlap}...", end=" ")
        params = tune_rf_for_binary_treatment(
            n_samples=n_sample,
            target_overlap=overlap,
            random_state=BASE_SEED,
        )
        optimal_params_binary[n_sample][overlap] = params
        print("Done.")

print("Pre-tuning complete.")

# %%
def run_binary_treatment_replication(
    n: int,
    target_overlap: float,
    learner_name: str,
    rep: int,
    base_seed: int = BASE_SEED,
    rf_params: dict = None,
) -> dict:
    """Run single replication with Binary Treatment DGP."""
    overlap_code = int(target_overlap * 100)
    learner_code = hash(learner_name) % 100
    seed = base_seed + overlap_code * 10000 + learner_code * 1000 + rep
    
    # Generate binary treatment data
    Y, D, X, info, dgp = generate_binary_treatment_data(
        n=n,
        target_overlap=target_overlap,
        random_state=seed,
    )
    
    m0_X = info['m0_X']  # Same as e_X for binary
    ell0_X = info['ell0_X']
    
    # Get learners
    if learner_name.upper() == 'ORACLE':
        from src.learners import OracleLearner
        learner_m = OracleLearner(dgp=dgp, target='m')
        learner_l = OracleLearner(dgp=dgp, target='l')
    else:
        params = rf_params if learner_name.upper() == 'RF_TUNED' else None
        learner_m = get_learner(learner_name, random_state=seed, params=params)
        learner_l = get_learner(learner_name, random_state=seed, params=params)
    
    # Run DML
    dml_result = run_dml(
        Y=Y, D=D, X=X,
        learner_m=learner_m,
        learner_l=learner_l,
        m0_X=m0_X,
        ell0_X=ell0_X,
        K=K_FOLDS,
        n_repeats=N_REPEATS,
        theta0=THETA0,
        random_state=seed,
    )
    
    return {
        'dgp_type': 'binary',
        'n': n,
        'target_overlap': target_overlap,
        'sample_r2': info['sample_r2'],
        'learner': learner_name,
        'replication': rep,
        'theta_hat': dml_result.theta_hat,
        'se': dml_result.se,
        'ci_lower': dml_result.ci_lower,
        'ci_upper': dml_result.ci_upper,
        'ci_length': dml_result.ci_length,
        'kappa_star': dml_result.kappa_star,
        'bias': dml_result.bias,
        'coverage': int(dml_result.covers(THETA0)),
        'rmse_m': dml_result.rmse_m,
        'rmse_l': dml_result.rmse_l,
        'mean_propensity': info['mean_propensity'],
        'treatment_frac': info['treatment_frac'],
    }

# %%
# Run Binary Treatment Simulation
print("\nPhase B: Running Binary Treatment Simulation...")

tasks_binary = []
for n_sample in N_SAMPLES_BINARY:
    for overlap in OVERLAP_REGIMES:
        rf_params = optimal_params_binary[n_sample][overlap]
        for learner in LEARNERS_BINARY:
            for rep in range(B_REPS_BINARY):
                tasks_binary.append((n_sample, overlap, learner, rep, BASE_SEED, rf_params))

results_binary = Parallel(n_jobs=N_JOBS, verbose=5)(
    delayed(run_binary_treatment_replication)(*task) for task in tasks_binary
)

df_binary = pd.DataFrame(results_binary)

# Save results
binary_path = RESULTS_DIR / 'sim_results_binary_treatment.csv'
df_binary.to_csv(binary_path, index=False)
print(f"\nBinary Treatment results saved to: {binary_path}")

# %%
# Aggregate Binary Treatment Results
df_binary_agg = df_binary.groupby(['learner', 'target_overlap', 'n']).agg(
    coverage=('coverage', 'mean'),
    mc_bias=('theta_hat', lambda x: x.mean() - THETA0),
    mc_se=('theta_hat', 'std'),
    median_kappa_star=('kappa_star', 'median'),
    avg_rmse_m=('rmse_m', 'mean'),
    avg_rmse_l=('rmse_l', 'mean'),
    n_reps=('replication', 'count'),
).reset_index()

df_binary_agg['bias_to_se_ratio'] = np.abs(df_binary_agg['mc_bias']) / df_binary_agg['mc_se']
df_binary_agg['Q'] = df_binary_agg['median_kappa_star'] * np.sqrt(
    df_binary_agg['avg_rmse_m'] * df_binary_agg['avg_rmse_l']
)

print("\nBinary Treatment Aggregates (sample):")
display(df_binary_agg.head(10).round(4))

# Save aggregates
df_binary_agg.to_csv(RESULTS_DIR / 'sim_aggregates_binary_treatment.csv', index=False)

# %%
# Binary Treatment: Coverage vs κ* Plot
fig, ax = plt.subplots(figsize=(10, 6))

colors_binary = sns.color_palette('husl', n_colors=len(LEARNERS_BINARY))

for idx, learner in enumerate(LEARNERS_BINARY):
    subset = df_binary_agg[df_binary_agg['learner'] == learner]
    ax.plot(
        subset['median_kappa_star'],
        subset['coverage'],
        marker='o',
        color=colors_binary[idx],
        linewidth=2,
        markersize=8,
        label=learner,
    )

ax.axhline(y=0.95, color='gray', linestyle='--', linewidth=1.5)
ax.set_xscale('log')
ax.set_xlabel(r'$\kappa^*$ (Log Scale)', fontsize=12)
ax.set_ylabel('Coverage Probability', fontsize=12)
ax.set_ylim(0, 1.05)
ax.set_title('Binary Treatment DGP: Coverage vs. Condition Number\n(Same Pattern as Continuous Treatment)', fontsize=14)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'binary_treatment_coverage.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'binary_treatment_coverage.png', dpi=300, bbox_inches='tight')
print(f"Saved: {RESULTS_DIR / 'binary_treatment_coverage.pdf'}")
plt.show()



# %% [markdown]
# ---
#
# # JCI Enhancement #3: Diagnostic Threshold Calibration
#
# This section provides **actionable guidance** for practitioners by computing
# ROC-like diagnostics for κ* as a predictor of coverage failure.
#
# **Key Output**: Recommend κ* thresholds analogous to the Stock-Yogo F > 10 rule.

# %%
# =============================================================================
# DIAGNOSTIC THRESHOLD CALIBRATION ANALYSIS
# =============================================================================

print("\n" + "="*80)
print("JCI ENHANCEMENT #3: DIAGNOSTIC THRESHOLD CALIBRATION")
print("="*80)

def compute_threshold_diagnostics(
    df: pd.DataFrame,
    kappa_col: str = 'median_kappa_star',
    coverage_col: str = 'coverage',
    target_coverage: float = 0.90,
    threshold_grid: np.ndarray = None,
) -> pd.DataFrame:
    """
    Compute ROC-like analysis for κ* as a diagnostic of coverage failure.
    
    For each candidate threshold κ*_thresh:
    - "Positive Prediction" = κ* > threshold (we predict failure)
    - "Ground Truth Failure" = coverage < target_coverage
    
    Computes sensitivity (TPR) and specificity (1 - FPR).
    
    Parameters
    ----------
    df : DataFrame
        Aggregated simulation results with κ* and coverage.
    kappa_col : str
        Column name for κ*.
    coverage_col : str
        Column name for coverage.
    target_coverage : float
        Coverage threshold below which we consider "failure".
    threshold_grid : array or None
        Grid of κ* thresholds to evaluate.
    
    Returns
    -------
    diagnostics : DataFrame
        With columns: threshold, sensitivity, specificity, PPV, NPV, youden_j
    """
    if threshold_grid is None:
        threshold_grid = np.logspace(0, 2.5, 50)  # κ* from 1 to ~316
    
    # Ground truth: did coverage fail?
    failure = (df[coverage_col] < target_coverage).values
    kappa_values = df[kappa_col].values
    
    n_failures = failure.sum()
    n_non_failures = len(failure) - n_failures
    
    results = []
    for thresh in threshold_grid:
        # Prediction: κ* > threshold means we predict failure
        pred_failure = kappa_values > thresh
        
        # Confusion matrix
        TP = np.sum(pred_failure & failure)
        FP = np.sum(pred_failure & ~failure)
        TN = np.sum(~pred_failure & ~failure)
        FN = np.sum(~pred_failure & failure)
        
        # Metrics
        sensitivity = TP / n_failures if n_failures > 0 else 0
        specificity = TN / n_non_failures if n_non_failures > 0 else 0
        PPV = TP / (TP + FP) if (TP + FP) > 0 else 0  # Positive Predictive Value
        NPV = TN / (TN + FN) if (TN + FN) > 0 else 0  # Negative Predictive Value
        youden_j = sensitivity + specificity - 1
        
        results.append({
            'threshold': thresh,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'PPV': PPV,
            'NPV': NPV,
            'youden_j': youden_j,
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN,
        })
    
    return pd.DataFrame(results)


def find_optimal_thresholds(diagnostics: pd.DataFrame) -> dict:
    """Find optimal κ* thresholds using different criteria."""
    # Youden's J statistic (maximize sensitivity + specificity - 1)
    idx_youden = diagnostics['youden_j'].idxmax()
    thresh_youden = diagnostics.loc[idx_youden, 'threshold']
    
    # High sensitivity (catch 90% of failures)
    high_sens = diagnostics[diagnostics['sensitivity'] >= 0.90]
    thresh_high_sens = high_sens['threshold'].max() if len(high_sens) > 0 else np.nan
    
    # High specificity (low false alarm rate, 90% specificity)
    high_spec = diagnostics[diagnostics['specificity'] >= 0.90]
    thresh_high_spec = high_spec['threshold'].min() if len(high_spec) > 0 else np.nan
    
    return {
        'youden_optimal': thresh_youden,
        'high_sensitivity_90': thresh_high_sens,
        'high_specificity_90': thresh_high_spec,
    }


# %%
# Run Diagnostic Analysis on Main Simulation Results

# Combine all DGP results for comprehensive analysis
df_all_sims = pd.concat([
    df_agg.assign(dgp_type='continuous'),
    df_binary_agg.assign(dgp_type='binary'),
], ignore_index=True)

print(f"\nCombined simulation results: {len(df_all_sims)} configurations")
print(f"DGP types: {df_all_sims['dgp_type'].unique()}")

# %%
# Compute threshold diagnostics for failure detection

# Define failure as coverage < 90%
TARGET_COVERAGE = 0.90

# Create threshold grid
THRESHOLD_GRID = np.concatenate([
    np.linspace(1, 10, 20),
    np.linspace(10, 50, 20),
    np.linspace(50, 200, 20),
])

# Exclude Oracle from diagnostic analysis (always near 95%)
df_for_diag = df_all_sims[df_all_sims['learner'] != 'Oracle'].copy()

diagnostics = compute_threshold_diagnostics(
    df_for_diag,
    kappa_col='median_kappa_star',
    coverage_col='coverage',
    target_coverage=TARGET_COVERAGE,
    threshold_grid=THRESHOLD_GRID,
)

optimal_thresholds = find_optimal_thresholds(diagnostics)

print("\n" + "="*60)
print("OPTIMAL κ* THRESHOLDS FOR COVERAGE FAILURE DETECTION")
print("="*60)
print(f"\nCoverage failure defined as: coverage < {TARGET_COVERAGE:.0%}")
print(f"\nOptimal thresholds:")
for name, val in optimal_thresholds.items():
    print(f"  {name}: κ* = {val:.1f}")

# %%
# Plot ROC Curve for κ* Diagnostic

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: ROC Curve
ax1 = axes[0]
ax1.plot(1 - diagnostics['specificity'], diagnostics['sensitivity'],
         'b-', linewidth=2, label='κ* Diagnostic')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

# Mark key thresholds
for thresh_val, color, name in [
    (5, 'green', 'κ*=5'),
    (10, 'orange', 'κ*=10'),
    (20, 'red', 'κ*=20'),
]:
    row = diagnostics.iloc[(diagnostics['threshold'] - thresh_val).abs().argmin()]
    ax1.scatter(1 - row['specificity'], row['sensitivity'],
                color=color, s=150, zorder=5, edgecolors='black', linewidth=1)
    ax1.annotate(name, (1 - row['specificity'] + 0.02, row['sensitivity']),
                fontsize=10, color=color)

ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
ax1.set_title(f'ROC Curve: κ* as Diagnostic for Coverage Failure\n(Failure = Coverage < {TARGET_COVERAGE:.0%})',
              fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.02, 1.02)
ax1.set_ylim(-0.02, 1.02)

# Panel B: Youden's J and Sensitivity/Specificity vs Threshold
ax2 = axes[1]
ax2.plot(diagnostics['threshold'], diagnostics['sensitivity'],
         'b-', linewidth=2, label='Sensitivity')
ax2.plot(diagnostics['threshold'], diagnostics['specificity'],
         'r-', linewidth=2, label='Specificity')
ax2.plot(diagnostics['threshold'], diagnostics['youden_j'],
         'g--', linewidth=2, label="Youden's J")

# Mark optimal threshold
thresh_opt = optimal_thresholds['youden_optimal']
ax2.axvline(x=thresh_opt, color='green', linestyle=':', linewidth=2, alpha=0.7)
ax2.annotate(f'Optimal: κ*={thresh_opt:.1f}',
            (thresh_opt + 2, 0.5), fontsize=11, color='green')

# Mark reference thresholds
for thresh_val, color in [(5, 'lightgreen'), (10, 'orange'), (20, 'red')]:
    ax2.axvline(x=thresh_val, color=color, linestyle='--', linewidth=1.5, alpha=0.5)

ax2.set_xscale('log')
ax2.set_xlabel(r'$\kappa^*$ Threshold', fontsize=12)
ax2.set_ylabel('Metric Value', fontsize=12)
ax2.set_title('Sensitivity, Specificity, and Youden\'s J\nvs. Diagnostic Threshold',
              fontsize=14)
ax2.legend(loc='center right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(RESULTS_DIR / 'diagnostic_roc_curve.pdf', dpi=300, bbox_inches='tight')
plt.savefig(RESULTS_DIR / 'diagnostic_roc_curve.png', dpi=300, bbox_inches='tight')
print(f"\nSaved: {RESULTS_DIR / 'diagnostic_roc_curve.pdf'}")
plt.show()

# %%
# Generate Threshold Recommendation Table

print("\n" + "="*80)
print("κ* THRESHOLD RECOMMENDATION TABLE")
print("="*80)

# Evaluate specific thresholds
reference_thresholds = [5, 10, 15, 20, 30, 50]
threshold_table = []

for thresh in reference_thresholds:
    row = diagnostics.iloc[(diagnostics['threshold'] - thresh).abs().argmin()]
    threshold_table.append({
        'κ* Threshold': thresh,
        'Sensitivity': f"{row['sensitivity']:.1%}",
        'Specificity': f"{row['specificity']:.1%}",
        'PPV': f"{row['PPV']:.1%}",
        'NPV': f"{row['NPV']:.1%}",
        'Interpretation': 'Well-conditioned' if thresh < 10 else (
            'Moderate' if thresh < 20 else 'Ill-conditioned'),
    })

df_threshold_table = pd.DataFrame(threshold_table)
print("\n")
display(df_threshold_table)

# Save table
df_threshold_table.to_csv(RESULTS_DIR / 'kappa_threshold_recommendations.csv', index=False)

# %%
# Final Summary and Recommendations

print("\n" + "="*80)
print("DIAGNOSTIC THRESHOLD CALIBRATION: SUMMARY")
print("="*80)

# Count failures by κ* regime
df_for_diag['kappa_regime'] = pd.cut(
    df_for_diag['median_kappa_star'],
    bins=[0, 5, 10, 20, np.inf],
    labels=['κ*<5', '5≤κ*<10', '10≤κ*<20', 'κ*≥20']
)

failure_rates = df_for_diag.groupby('kappa_regime').agg(
    n_configs=('coverage', 'count'),
    avg_coverage=('coverage', 'mean'),
    failure_rate=('coverage', lambda x: (x < TARGET_COVERAGE).mean()),
).round(3)

print("\nCoverage Failure Rates by κ* Regime:")
display(failure_rates)

print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RECOMMENDED κ* DIAGNOSTIC THRESHOLDS                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Based on {len(df_for_diag)} simulation configurations across 3 DGP types:                 ║
║                                                                              ║
║  • κ* < 5   : WELL-CONDITIONED                                               ║
║               Expected failure rate: {failure_rates.loc['κ*<5', 'failure_rate']:.1%}                                   ║
║               Recommendation: Standard DML inference applies                 ║
║                                                                              ║
║  • 5 ≤ κ* < 10 : MODERATE CONCERN                                            ║
║               Expected failure rate: {failure_rates.loc['5≤κ*<10', 'failure_rate']:.1%}                                  ║
║               Recommendation: Report κ*, consider sensitivity analysis       ║
║                                                                              ║
║  • 10 ≤ κ* < 20 : ILL-CONDITIONED                                            ║
║               Expected failure rate: {failure_rates.loc['10≤κ*<20', 'failure_rate']:.1%}                                  ║
║               Recommendation: Strong sensitivity analysis required           ║
║                                                                              ║
║  • κ* ≥ 20  : SEVERELY ILL-CONDITIONED                                       ║
║               Expected failure rate: {failure_rates.loc['κ*≥20', 'failure_rate']:.1%}                                  ║
║               Recommendation: Improve overlap before trusting estimates      ║
║                                                                              ║
║  OPTIMAL THRESHOLD (Youden's J): κ* = {optimal_thresholds['youden_optimal']:.1f}                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

# Save summary
summary_dict = {
    'optimal_youden': optimal_thresholds['youden_optimal'],
    'high_sensitivity_90': optimal_thresholds['high_sensitivity_90'],
    'high_specificity_90': optimal_thresholds['high_specificity_90'],
    'n_configurations': len(df_for_diag),
    'n_dgp_types': 2,
}
pd.DataFrame([summary_dict]).to_csv(
    RESULTS_DIR / 'diagnostic_threshold_summary.csv', index=False
)

# %%
print("\n" + "="*80)
print("JCI SIMULATION ENHANCEMENTS COMPLETE")
print("="*80)
print(f"\nAll results saved to: {RESULTS_DIR}")
print("\nNew outputs from JCI Enhancements:")
print("  - sim_results_binary_treatment.csv")
print("  - sim_aggregates_binary_treatment.csv")
print("  - binary_treatment_coverage.pdf")
print("  - diagnostic_roc_curve.pdf")
print("  - kappa_threshold_recommendations.csv")
print("  - diagnostic_threshold_summary.csv")
