"""
Core simulation module for the DML condition number study.

This module implements Monte Carlo simulations for Double Machine Learning (DML)
in the Partially Linear Regression (PLR) model, with emphasis on the condition
number κ_DML as a diagnostic for inference reliability.

Theoretical Foundation (from the paper):
    The DML condition number is defined as:
        κ_DML := 1 / |Ĵ_θ| = n / Σᵢ Ûᵢ²
    
    where Ĵ_θ = -(1/n) Σᵢ Ûᵢ² is the empirical Jacobian of the orthogonal score.
    
    The refined linearization shows:
        θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ
    
    This implies the condition number directly amplifies both variance and bias:
        |θ̂ − θ₀| = O_P(κ_DML/√n + κ_DML · rₙ)
    
    Three conditioning regimes:
        (i)   Well-conditioned:        κₙ = O_P(1)       → CI length O_P(1/√n)
        (ii)  Moderately ill-cond:     κₙ = O_P(n^β)     → CI length O_P(n^{β-1/2})
        (iii) Severely ill-cond:       κₙ ≍ c√n          → CI length O_P(1)

References:
    - Robinson (1988), Root-N-Consistent Semiparametric Regression, Econometrica
    - Chernozhukov et al. (2018), Double/Debiased ML, Econometrics Journal
    - Chernozhukov, Newey & Singh (2023), Automatic Debiased ML, Biometrika
    - Bach et al. (2022), DoubleML, Journal of Statistical Software

Author: DML Condition Number Study
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, Ridge, RidgeCV
from sklearn.model_selection import KFold

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================

# True treatment effect (used throughout the study)
THETA0: float = 1.0

# Fixed design parameters
P_DEFAULT: int = 10          # Covariate dimension
RHO_DEFAULT: float = 0.5     # Toeplitz correlation (fixed per spec)
K_FOLDS: int = 5             # Cross-fitting folds

# Target R²(D|X) levels for overlap calibration
R2_TARGETS: Dict[str, float] = {
    "high": 0.75,      # Well-conditioned (good overlap)
    "moderate": 0.90,  # Moderately ill-conditioned
    "low": 0.97,       # Severely ill-conditioned (poor overlap)
}

# Default random seed for reproducibility
DEFAULT_SEED: int = 20241205

# Monte Carlo configuration
B_DEFAULT: int = 500  # Number of replications


# =============================================================================
# TOEPLITZ COVARIANCE STRUCTURE
# =============================================================================

def make_toeplitz_cov(p: int, rho: float) -> NDArray:
    """
    Construct Toeplitz covariance matrix Σ(ρ) with Σ_{jk} = ρ^{|j-k|}.
    
    This is an AR(1) correlation structure commonly used in simulation studies.
    
    Parameters
    ----------
    p : int
        Dimension of the covariance matrix.
    rho : float
        Correlation decay parameter, must be in [0, 1).
    
    Returns
    -------
    Sigma : ndarray of shape (p, p)
        Symmetric positive-definite Toeplitz covariance matrix.
    """
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


# =============================================================================
# TREATMENT EQUATION COEFFICIENTS
# =============================================================================

def get_gamma_coeffs(p: int = P_DEFAULT) -> NDArray:
    """
    Coefficient vector γ for treatment equation: S = X @ γ.
    
    Uses geometric decay γ_j = 0.7^{j-1} as specified.
    
    Parameters
    ----------
    p : int, default=10
        Dimension of covariates.
    
    Returns
    -------
    gamma : ndarray of shape (p,)
        Coefficient vector with γ_j = 0.7^{j-1}.
    """
    return 0.7 ** np.arange(p)


def compute_V_gamma(rho: float, p: int = P_DEFAULT) -> float:
    """
    Compute Var(S) = Var(X @ γ) = γ' Σ(ρ) γ analytically.
    
    Parameters
    ----------
    rho : float
        Toeplitz correlation parameter.
    p : int, default=10
        Dimension of covariates.
    
    Returns
    -------
    V_gamma : float
        Theoretical variance of the linear index S = X @ γ.
    """
    Sigma = make_toeplitz_cov(p, rho)
    gamma = get_gamma_coeffs(p)
    return float(gamma @ Sigma @ gamma)


# =============================================================================
# OVERLAP CALIBRATION VIA R²(D|X)
# =============================================================================

def calibrate_sigma_xi_sq(
    R2_target: float,
    rho: float = RHO_DEFAULT,
    p: int = P_DEFAULT,
) -> Tuple[float, float]:
    """
    Calibrate σ_ξ² to achieve target R²(D|X).
    
    For the treatment equation D = S + ξ where S = X @ γ and ξ ~ N(0, σ_ξ²):
        Var(D) = V_γ + σ_ξ²
        R²(D|X) = V_γ / Var(D) = V_γ / (V_γ + σ_ξ²)
    
    Solving for σ_ξ²:
        σ_ξ² = V_γ · (1 - R²) / R²
    
    Higher R²(D|X) means less residual variation in treatment after partialling
    out X, leading to larger κ_DML (worse conditioning).
    
    Parameters
    ----------
    R2_target : float
        Target R²(D|X), must be in (0, 1).
    rho : float, default=0.5
        Toeplitz correlation parameter.
    p : int, default=10
        Covariate dimension.
    
    Returns
    -------
    sigma_xi_sq : float
        Calibrated variance of ξ.
    V_gamma : float
        Theoretical variance of S = X @ γ.
    """
    if not 0 < R2_target < 1:
        raise ValueError(f"R2_target must be in (0, 1), got {R2_target}")
    
    V_gamma = compute_V_gamma(rho, p)
    sigma_xi_sq = V_gamma * (1 - R2_target) / R2_target
    
    return sigma_xi_sq, V_gamma


# =============================================================================
# OUTCOME EQUATION: NONLINEAR NUISANCE FUNCTION
# =============================================================================

def g0_function(X: NDArray) -> NDArray:
    """
    Nonlinear nuisance function g₀(X) for the outcome equation.
    
    g₀(X) = 0.5 * X[:,0]² + 0.5 * sin(X[:,1]) + 0.3 * X[:,2] * X[:,3]
    
    This function is smooth and bounded, requiring flexible methods to
    estimate accurately, but not pathologically difficult.
    
    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate matrix with p >= 4.
    
    Returns
    -------
    g0 : ndarray of shape (n,)
        Evaluated nuisance function.
    """
    return (
        0.5 * X[:, 0] ** 2 
        + 0.5 * np.sin(X[:, 1]) 
        + 0.3 * X[:, 2] * X[:, 3]
    )


# =============================================================================
# DATA GENERATING PROCESS
# =============================================================================

@dataclass
class DGPInfo:
    """Container for DGP calibration information."""
    R2_target: float
    sigma_xi_sq: float
    V_gamma: float
    sample_R2: float
    var_D: float
    var_xi: float


def generate_plr_data(
    n: int,
    R2_target: float,
    rho: float = RHO_DEFAULT,
    p: int = P_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, DGPInfo]:
    """
    Generate data from the PLR model with overlap calibrated via R²(D|X).
    
    Model (Robinson 1988; Chernozhukov et al. 2018):
        Y = D · θ₀ + g₀(X) + ε,    ε ~ N(0, 1)
        D = S + ξ,                  ξ ~ N(0, σ_ξ²)
        S = X @ γ                   (linear index)
        X ~ N(0, Σ(ρ)),            Σ_{jk} = ρ^|j-k| (Toeplitz)
    
    where:
        - γ_j = 0.7^{j-1} (geometric decay)
        - g₀(X) = 0.5·X₀² + 0.5·sin(X₁) + 0.3·X₂·X₃
        - σ_ξ² calibrated to achieve target R²(D|X)
    
    Parameters
    ----------
    n : int
        Sample size.
    R2_target : float
        Target R²(D|X) in (0, 1). Higher values → worse conditioning.
    rho : float, default=0.5
        Toeplitz correlation parameter (fixed per spec).
    p : int, default=10
        Dimension of covariates X.
    theta0 : float, default=1.0
        True treatment effect parameter.
    random_state : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    Y : ndarray of shape (n,)
        Outcome variable.
    D : ndarray of shape (n,)
        Treatment variable.
    X : ndarray of shape (n, p)
        Covariate matrix.
    info : DGPInfo
        Calibration and diagnostic information.
    """
    rng = np.random.default_rng(random_state)
    
    # Calibrate σ_ξ² for target R²(D|X)
    sigma_xi_sq, V_gamma = calibrate_sigma_xi_sq(R2_target, rho, p)
    sigma_xi = np.sqrt(sigma_xi_sq)
    
    # Construct Toeplitz covariance matrix
    Sigma = make_toeplitz_cov(p, rho)
    
    # Generate covariates X ~ N(0, Σ(ρ))
    X = rng.multivariate_normal(np.zeros(p), Sigma, size=n)
    
    # Treatment equation: D = S + ξ where S = X @ γ
    gamma = get_gamma_coeffs(p)
    S = X @ gamma  # Linear index (propensity score component)
    xi = rng.normal(0, sigma_xi, size=n)
    D = S + xi
    
    # Outcome equation: Y = D·θ₀ + g₀(X) + ε
    g0_X = g0_function(X)
    eps = rng.normal(0, 1.0, size=n)
    Y = D * theta0 + g0_X + eps
    
    # Compute sample R²(D|X) for diagnostics
    var_D = np.var(D, ddof=0)
    var_S = np.var(S, ddof=0)
    sample_R2 = var_S / var_D if var_D > 0 else 0.0
    
    info = DGPInfo(
        R2_target=R2_target,
        sigma_xi_sq=sigma_xi_sq,
        V_gamma=V_gamma,
        sample_R2=sample_R2,
        var_D=var_D,
        var_xi=np.var(xi, ddof=0),
    )
    
    return Y, D, X, info


# =============================================================================
# NUISANCE LEARNER FACTORIES
# =============================================================================

LearnerType = Literal["LIN", "LAS", "RF"]


def get_nuisance_model(learner: LearnerType, random_state: int = 42) -> BaseEstimator:
    """
    Factory function to create nuisance regression models.
    
    Implements three learner types as specified:
    
    1. LIN (Ridge/OLS): Ridge regression with CV-tuned alpha.
       - Uses RidgeCV with alphas = [0.0, 0.01, 0.1, 1.0]
       - alpha=0 is equivalent to OLS
    
    2. LAS (Lasso): Lasso with CV-tuned regularization.
       - Uses LassoCV with 5-fold CV
       - Post-Lasso optional (using direct predictions here)
    
    3. RF (Random Forest): Conservative fixed hyperparameters.
       - n_estimators = 200
       - max_depth = 5
       - min_samples_leaf = 10
       - max_features = "sqrt"
    
    Parameters
    ----------
    learner : {"LIN", "LAS", "RF"}
        Learner type label.
    random_state : int, default=42
        Random state for reproducibility.
    
    Returns
    -------
    model : sklearn estimator
        Configured regression model.
    """
    if learner == "LIN":
        # Ridge with CV over alphas including 0 (OLS)
        # Note: RidgeCV with alpha=0 can be numerically unstable,
        # so we use small positive values
        return RidgeCV(
            alphas=[1e-6, 0.01, 0.1, 1.0, 10.0],
            cv=5,
            fit_intercept=True,
        )
    
    elif learner == "LAS":
        # Lasso with 5-fold CV
        return LassoCV(
            cv=5,
            fit_intercept=True,
            max_iter=10000,
            random_state=random_state,
            n_jobs=-1,
        )
    
    elif learner == "RF":
        # Random Forest with conservative fixed hyperparameters
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
        )
    
    else:
        raise ValueError(f"Unknown learner: {learner}. Must be 'LIN', 'LAS', or 'RF'.")


# =============================================================================
# CROSS-FITTED DML ESTIMATOR
# =============================================================================

@dataclass
class DMLResult:
    """
    Container for DML estimation results with conditioning diagnostics.
    
    Key diagnostic:
        κ_DML = n / Σ Ûᵢ²  (condition number)
    
    The parameter-scale expansion shows:
        |θ̂ - θ₀| = O_P(κ_DML / √n + κ_DML · r_n)
    so κ_DML directly amplifies both variance and nuisance bias.
    """
    theta_hat: float
    kappa_dml: float
    se_dml: float
    ci_lower: float
    ci_upper: float
    J_hat: float
    U_hat: NDArray  # Residualized treatment
    V_hat: NDArray  # Residualized outcome
    
    @property
    def ci_length(self) -> float:
        """Length of 95% confidence interval."""
        return self.ci_upper - self.ci_lower
    
    def covers(self, theta0: float = THETA0) -> bool:
        """Check if CI covers true parameter."""
        return self.ci_lower <= theta0 <= self.ci_upper


def run_dml_plr(
    Y: NDArray,
    D: NDArray,
    X: NDArray,
    learner_label: LearnerType,
    K: int = K_FOLDS,
    random_state: Optional[int] = None,
) -> DMLResult:
    """
    Cross-fitted DML estimator for the PLR model.
    
    Implements the partialling-out approach with K-fold cross-fitting 
    (Chernozhukov et al. 2018).
    
    Algorithm:
        1. Split data into K folds with shuffle=True
        2. For each fold k, fit on training folds I^c_k:
           - m̂(X) ≈ E[D|X]   (treatment model)
           - ĝ(X) ≈ E[Y|X]   (outcome model, we use g* = ℓ₀)
        3. Compute out-of-fold residuals:
           - Ûᵢ = Dᵢ − m̂(Xᵢ)  (residualized treatment)
           - V̂ᵢ = Yᵢ − ĝ(Xᵢ)  (residualized outcome)
        4. DML estimate:
           θ̂ = Σᵢ(ÛᵢV̂ᵢ) / Σᵢ(Ûᵢ²)
        5. Condition number:
           κ_DML = n / Σᵢ Ûᵢ² = 1/|Ĵ_θ|
        6. Standard error (heteroskedasticity-robust):
           ε̂ᵢ = Yᵢ − ĝ(Xᵢ) − θ̂·Ûᵢ
           SE = (κ_DML/√n) · √[(1/n)Σᵢ Ûᵢ²·ε̂ᵢ²]
    
    Parameters
    ----------
    Y : ndarray of shape (n,)
        Outcome variable.
    D : ndarray of shape (n,)
        Treatment variable.
    X : ndarray of shape (n, p)
        Covariate matrix.
    learner_label : {"LIN", "LAS", "RF"}
        Nuisance learner type.
    K : int, default=5
        Number of cross-fitting folds.
    random_state : int or None
        Random state for fold splitting and learners.
    
    Returns
    -------
    result : DMLResult
        Contains theta_hat, kappa_dml, se_dml, ci_lower, ci_upper, J_hat,
        U_hat, V_hat.
    
    Notes
    -----
    The condition number κ_DML = n / Σᵢ Ûᵢ² is the key diagnostic.
    From the paper's linearization:
        θ̂ − θ₀ = κ_DML · (Sₙ + Bₙ) + Rₙ
    Large κ_DML amplifies both variance (Sₙ) and nuisance bias (Bₙ).
    """
    n = len(Y)
    rs = random_state if random_state is not None else DEFAULT_SEED
    
    # Initialize arrays for out-of-fold predictions
    m_hat = np.zeros(n)  # Predictions for E[D|X]
    g_hat = np.zeros(n)  # Predictions for E[Y|X]
    
    # K-fold cross-fitting with shuffle
    kf = KFold(n_splits=K, shuffle=True, random_state=rs)
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        D_train = D[train_idx]
        Y_train = Y[train_idx]
        
        # Create fresh model instances for each fold
        model_m = get_nuisance_model(learner_label, random_state=rs)
        model_g = get_nuisance_model(learner_label, random_state=rs)
        
        # Fit m̂(X) = E[D|X] and predict out-of-fold
        model_m.fit(X_train, D_train)
        m_hat[test_idx] = model_m.predict(X_test)
        
        # Fit ĝ(X) = E[Y|X] and predict out-of-fold
        model_g.fit(X_train, Y_train)
        g_hat[test_idx] = model_g.predict(X_test)
    
    # Compute residuals
    U_hat = D - m_hat  # Residualized treatment: Û = D − m̂(X)
    V_hat = Y - g_hat  # Residualized outcome: V̂ = Y − ĝ(X)
    
    # DML estimate: θ̂ = Σ(Û·V̂) / Σ(Û²)
    sum_U_sq = np.sum(U_hat ** 2)
    sum_UV = np.sum(U_hat * V_hat)
    
    # Handle near-zero denominator (extreme ill-conditioning)
    if sum_U_sq < 1e-12:
        return DMLResult(
            theta_hat=np.nan,
            kappa_dml=np.inf,
            se_dml=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            J_hat=0.0,
            U_hat=U_hat,
            V_hat=V_hat,
        )
    
    theta_hat = sum_UV / sum_U_sq
    
    # Empirical Jacobian: Ĵ_θ = −(1/n) Σ Û²
    J_hat = -sum_U_sq / n
    
    # Condition number: κ_DML = 1/|Ĵ_θ| = n / Σ Û²
    kappa_dml = n / sum_U_sq
    
    # Residuals for variance estimation
    eps_hat = Y - g_hat - theta_hat * U_hat
    
    # Standard error (heteroskedasticity-robust)
    # SE = (κ_DML/√n) · √[(1/n) Σᵢ Ûᵢ²·ε̂ᵢ²]
    var_component = np.mean(U_hat ** 2 * eps_hat ** 2)
    se_dml = kappa_dml / np.sqrt(n) * np.sqrt(var_component)
    
    # 95% confidence interval
    z_alpha = 1.96
    ci_lower = theta_hat - z_alpha * se_dml
    ci_upper = theta_hat + z_alpha * se_dml
    
    return DMLResult(
        theta_hat=theta_hat,
        kappa_dml=kappa_dml,
        se_dml=se_dml,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        J_hat=J_hat,
        U_hat=U_hat,
        V_hat=V_hat,
    )


# =============================================================================
# MONTE CARLO SIMULATION ENGINE
# =============================================================================

@dataclass
class ReplicationResult:
    """Container for a single Monte Carlo replication result."""
    # Design identifiers
    n: int
    R2_target: float
    rho: float
    learner: str
    replication: int
    
    # DML estimates
    theta_hat: float
    kappa_dml: float
    se_dml: float
    ci_lower: float
    ci_upper: float
    ci_length: float
    
    # Performance metrics
    coverage: int  # 1 if CI covers theta0, 0 otherwise
    bias: float
    squared_error: float
    
    # DGP diagnostics
    sample_R2: float


def run_single_replication(
    n: int,
    R2_target: float,
    learner_label: LearnerType,
    replication: int,
    rho: float = RHO_DEFAULT,
    p: int = P_DEFAULT,
    theta0: float = THETA0,
    K: int = K_FOLDS,
    base_seed: int = DEFAULT_SEED,
) -> ReplicationResult:
    """
    Run a single Monte Carlo replication.
    
    Parameters
    ----------
    n : int
        Sample size.
    R2_target : float
        Target R²(D|X).
    learner_label : {"LIN", "LAS", "RF"}
        Nuisance learner.
    replication : int
        Replication index.
    rho : float, default=0.5
        Toeplitz correlation.
    p : int, default=10
        Covariate dimension.
    theta0 : float, default=1.0
        True treatment effect.
    K : int, default=5
        Number of cross-fitting folds.
    base_seed : int
        Base seed for reproducibility.
    
    Returns
    -------
    result : ReplicationResult
        Complete replication results.
    """
    # Create unique seed for this replication
    # Encode: n, R2, learner, replication into seed
    learner_code = {"LIN": 0, "LAS": 1, "RF": 2}[learner_label]
    R2_code = int(R2_target * 100)  # e.g., 75, 90, 97
    seed = base_seed + n * 1000000 + R2_code * 10000 + learner_code * 1000 + replication
    
    # Generate data
    Y, D, X, dgp_info = generate_plr_data(
        n=n,
        R2_target=R2_target,
        rho=rho,
        p=p,
        theta0=theta0,
        random_state=seed,
    )
    
    # Run DML estimator
    dml_result = run_dml_plr(
        Y=Y,
        D=D,
        X=X,
        learner_label=learner_label,
        K=K,
        random_state=seed,
    )
    
    # Compute performance metrics
    theta_hat = dml_result.theta_hat
    
    if np.isnan(theta_hat):
        coverage = 0
        bias = np.nan
        squared_error = np.nan
    else:
        coverage = 1 if dml_result.covers(theta0) else 0
        bias = theta_hat - theta0
        squared_error = bias ** 2
    
    return ReplicationResult(
        n=n,
        R2_target=R2_target,
        rho=rho,
        learner=learner_label,
        replication=replication,
        theta_hat=theta_hat,
        kappa_dml=dml_result.kappa_dml,
        se_dml=dml_result.se_dml,
        ci_lower=dml_result.ci_lower,
        ci_upper=dml_result.ci_upper,
        ci_length=dml_result.ci_length,
        coverage=coverage,
        bias=bias,
        squared_error=squared_error,
        sample_R2=dgp_info.sample_R2,
    )


def run_simulation(
    n_list: List[int] = [500, 2000],
    R2_list: List[float] = [0.75, 0.90, 0.97],
    learners: List[LearnerType] = ["LIN", "LAS", "RF"],
    B: int = B_DEFAULT,
    rho: float = RHO_DEFAULT,
    p: int = P_DEFAULT,
    theta0: float = THETA0,
    K: int = K_FOLDS,
    base_seed: int = DEFAULT_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the full Monte Carlo simulation across the design grid.
    
    Design grid:
        - n ∈ n_list (sample sizes)
        - R²(D|X) ∈ R2_list (overlap levels)
        - learner ∈ learners (nuisance estimators)
        - B replications per design cell
    
    Total replications: len(n_list) × len(R2_list) × len(learners) × B
    
    Parameters
    ----------
    n_list : list of int, default=[500, 2000]
        Sample sizes.
    R2_list : list of float, default=[0.75, 0.90, 0.97]
        Target R²(D|X) levels.
    learners : list of str, default=["LIN", "LAS", "RF"]
        Nuisance learner labels.
    B : int, default=500
        Number of Monte Carlo replications per design cell.
    rho : float, default=0.5
        Toeplitz correlation (fixed).
    p : int, default=10
        Covariate dimension.
    theta0 : float, default=1.0
        True treatment effect.
    K : int, default=5
        Number of cross-fitting folds.
    base_seed : int
        Base seed for reproducibility.
    verbose : bool, default=True
        Print progress updates.
    
    Returns
    -------
    results_df : pd.DataFrame
        Long-format DataFrame with one row per replication.
        Columns: n, R2_target, rho, learner, replication, theta_hat,
                 kappa_dml, se_dml, ci_lower, ci_upper, ci_length,
                 coverage, bias, squared_error, sample_R2.
    """
    results = []
    
    # Calculate total number of design cells
    n_cells = len(n_list) * len(R2_list) * len(learners)
    total_reps = n_cells * B
    
    if verbose:
        print(f"DML Condition Number Monte Carlo Study")
        print(f"=" * 50)
        print(f"Design: {len(n_list)} sample sizes × {len(R2_list)} R² levels × {len(learners)} learners")
        print(f"Replications per cell: B = {B}")
        print(f"Total replications: {total_reps:,}")
        print(f"Fixed parameters: ρ = {rho}, p = {p}, K = {K}, θ₀ = {theta0}")
        print(f"=" * 50)
    
    cell_count = 0
    for n in n_list:
        for R2_target in R2_list:
            for learner_label in learners:
                cell_count += 1
                
                if verbose:
                    R2_label = {0.75: "high", 0.90: "moderate", 0.97: "low"}[R2_target]
                    print(f"\n[{cell_count}/{n_cells}] n={n}, R²={R2_target} ({R2_label}), learner={learner_label}")
                
                for rep in range(B):
                    result = run_single_replication(
                        n=n,
                        R2_target=R2_target,
                        learner_label=learner_label,
                        replication=rep,
                        rho=rho,
                        p=p,
                        theta0=theta0,
                        K=K,
                        base_seed=base_seed,
                    )
                    
                    # Convert dataclass to dict for DataFrame
                    results.append({
                        "n": result.n,
                        "R2_target": result.R2_target,
                        "rho": result.rho,
                        "learner": result.learner,
                        "replication": result.replication,
                        "theta_hat": result.theta_hat,
                        "kappa_dml": result.kappa_dml,
                        "se_dml": result.se_dml,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                        "ci_length": result.ci_length,
                        "coverage": result.coverage,
                        "bias": result.bias,
                        "squared_error": result.squared_error,
                        "sample_R2": result.sample_R2,
                    })
                    
                    if verbose and (rep + 1) % 100 == 0:
                        print(f"    Completed {rep + 1}/{B} replications")
    
    results_df = pd.DataFrame(results)
    
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Simulation complete. {len(results_df)} total replications.")
    
    return results_df


# =============================================================================
# SUMMARY STATISTICS AND TABLES
# =============================================================================

def compute_cell_summary(
    results_df: pd.DataFrame,
    theta0: float = THETA0,
) -> pd.DataFrame:
    """
    Compute cell-level summary statistics from raw simulation results.
    
    Aggregates by (n, R2_target, learner) and computes:
        - median_kappa: median condition number
        - coverage: Monte Carlo coverage rate
        - avg_ci_length: average CI length
        - bias: mean bias
        - rmse: root mean squared error
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Raw simulation results from run_simulation().
    theta0 : float, default=1.0
        True treatment effect.
    
    Returns
    -------
    cell_summary : pd.DataFrame
        One row per design cell with summary statistics.
    """
    grouped = results_df.groupby(["n", "R2_target", "learner"])
    
    summary = grouped.agg(
        n_reps=("replication", "count"),
        mean_sample_R2=("sample_R2", "mean"),
        median_kappa=("kappa_dml", "median"),
        mean_kappa=("kappa_dml", "mean"),
        sd_kappa=("kappa_dml", "std"),
        coverage=("coverage", "mean"),
        avg_ci_length=("ci_length", "mean"),
        mean_bias=("bias", "mean"),
        mean_se=("se_dml", "mean"),
    ).reset_index()
    
    # Compute RMSE
    rmse = grouped["squared_error"].mean().apply(np.sqrt).reset_index(name="rmse")
    summary = summary.merge(rmse, on=["n", "R2_target", "learner"])
    
    # Add overlap label
    overlap_labels = {0.75: "High", 0.90: "Moderate", 0.97: "Low"}
    summary["overlap"] = summary["R2_target"].map(overlap_labels)
    
    return summary


def assign_kappa_regime(median_kappa: float) -> str:
    """
    Assign κ-regime based on median κ_DML.
    
    Regimes:
        - "< 1": Well-conditioned
        - "1-2": Moderately ill-conditioned
        - "> 2": Severely ill-conditioned
    
    Parameters
    ----------
    median_kappa : float
        Median condition number for a design cell.
    
    Returns
    -------
    regime : str
        Regime label.
    """
    if median_kappa < 1:
        return "< 1"
    elif median_kappa <= 2:
        return "1-2"
    else:
        return "> 2"


def compute_regime_summary(cell_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Group cell-level summary by κ-regime and learner for Table 2.
    
    Parameters
    ----------
    cell_summary : pd.DataFrame
        Cell-level summary from compute_cell_summary().
    
    Returns
    -------
    regime_summary : pd.DataFrame
        Summary by (kappa_regime, learner).
    """
    # Assign regime based on median κ
    cell_summary = cell_summary.copy()
    cell_summary["kappa_regime"] = cell_summary["median_kappa"].apply(assign_kappa_regime)
    
    # Aggregate by regime and learner
    grouped = cell_summary.groupby(["kappa_regime", "learner"])
    
    regime_summary = grouped.agg(
        n_cells=("n", "count"),
        avg_coverage=("coverage", "mean"),
        avg_ci_length=("avg_ci_length", "mean"),
        avg_bias=("mean_bias", "mean"),
        avg_rmse=("rmse", "mean"),
        avg_median_kappa=("median_kappa", "mean"),
    ).reset_index()
    
    # Sort by regime order
    regime_order = {"< 1": 0, "1-2": 1, "> 2": 2}
    regime_summary["regime_order"] = regime_summary["kappa_regime"].map(regime_order)
    regime_summary = regime_summary.sort_values(["regime_order", "learner"]).drop(columns="regime_order")
    
    return regime_summary


def make_table1(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct Table 1: Design summary with median κ_DML by overlap level.
    
    Aggregates across n and learners to summarize by R²(D|X) level.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Raw simulation results.
    
    Returns
    -------
    table1 : pd.DataFrame
        Design summary table.
    """
    # Aggregate by R2_target only (across n, learners, replications)
    grouped = results_df.groupby("R2_target")
    
    summary = grouped.agg(
        median_kappa=("kappa_dml", "median"),
        mean_kappa=("kappa_dml", "mean"),
        sd_kappa=("kappa_dml", "std"),
        n_reps=("replication", "count"),
    ).reset_index()
    
    # Add labels
    overlap_labels = {0.75: "High (R²=0.75)", 0.90: "Moderate (R²=0.90)", 0.97: "Low (R²=0.97)"}
    summary["Overlap"] = summary["R2_target"].map(overlap_labels)
    
    # Format for display
    table1 = pd.DataFrame({
        "Overlap": summary["Overlap"],
        "R²(D|X)": summary["R2_target"],
        "Median κ_DML": summary["median_kappa"].round(2),
        "Mean κ_DML": summary["mean_kappa"].round(2),
        "SD κ_DML": summary["sd_kappa"].round(2),
        "n values": "500, 2000",
        "Learners": "LIN, LAS, RF",
    })
    
    return table1


def make_table2(cell_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Construct Table 2: Coverage by κ-regime and learner.
    
    Parameters
    ----------
    cell_summary : pd.DataFrame
        Cell-level summary from compute_cell_summary().
    
    Returns
    -------
    table2 : pd.DataFrame
        Coverage table by regime and learner.
    """
    regime_summary = compute_regime_summary(cell_summary)
    
    table2 = pd.DataFrame({
        "κ-Regime": regime_summary["kappa_regime"],
        "Learner": regime_summary["learner"],
        "Coverage": (regime_summary["avg_coverage"] * 100).round(1),
        "Avg CI Length": regime_summary["avg_ci_length"].round(3),
        "Bias": regime_summary["avg_bias"].round(4),
        "RMSE": regime_summary["avg_rmse"].round(4),
    })
    
    return table2


def table_to_latex(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
    """
    Convert DataFrame to LaTeX table string.
    
    Parameters
    ----------
    df : pd.DataFrame
        Table to convert.
    caption : str
        LaTeX caption.
    label : str
        LaTeX label.
    
    Returns
    -------
    latex_str : str
        LaTeX table code.
    """
    latex = df.to_latex(index=False, escape=False, float_format="%.3f")
    
    if caption or label:
        # Add caption and label
        lines = latex.split('\n')
        begin_idx = next(i for i, line in enumerate(lines) if 'begin{tabular}' in line)
        if caption:
            lines.insert(begin_idx, f"\\caption{{{caption}}}")
        if label:
            lines.insert(begin_idx + (1 if caption else 0), f"\\label{{{label}}}")
        latex = '\n'.join(lines)
    
    return latex


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_coverage_vs_kappa(
    cell_summary: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
):
    """
    Figure 1: Coverage vs κ_DML (killer plot).
    
    Scatter plot showing Monte Carlo coverage vs median κ_DML for each
    design cell, with different markers/colors for learners.
    
    Parameters
    ----------
    cell_summary : pd.DataFrame
        Cell-level summary from compute_cell_summary().
    save_path : str or None
        If provided, save figure to this path.
    figsize : tuple
        Figure size in inches.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    
    # Configure matplotlib
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Markers and colors for learners
    learner_styles = {
        "LIN": {"marker": "o", "color": "#1f77b4", "label": "LIN (Ridge)"},
        "LAS": {"marker": "s", "color": "#ff7f0e", "label": "LAS (Lasso)"},
        "RF": {"marker": "^", "color": "#2ca02c", "label": "RF (Random Forest)"},
    }
    
    # Size encoding for n
    size_map = {500: 80, 2000: 160}
    
    for learner in ["LIN", "LAS", "RF"]:
        df_learner = cell_summary[cell_summary["learner"] == learner]
        
        for n_val in [500, 2000]:
            df_sub = df_learner[df_learner["n"] == n_val]
            
            ax.scatter(
                df_sub["median_kappa"],
                df_sub["coverage"] * 100,
                marker=learner_styles[learner]["marker"],
                color=learner_styles[learner]["color"],
                s=size_map[n_val],
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
                label=f"{learner_styles[learner]['label']}, n={n_val}" if n_val == 500 else None,
            )
    
    # Add reference line at 95%
    ax.axhline(y=95, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, label="Nominal 95%")
    
    # Add vertical lines for regime boundaries
    ax.axvline(x=1, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.axvline(x=2, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    
    # Labels and formatting
    ax.set_xlabel(r"Median $\kappa_{\mathrm{DML}}$", fontsize=13)
    ax.set_ylabel("Monte Carlo Coverage of 95% CI (%)", fontsize=13)
    ax.set_title(r"Coverage vs Condition Number $\kappa_{\mathrm{DML}}$", fontsize=14, fontweight="bold")
    
    # Set axis limits
    ax.set_xlim(0, max(cell_summary["median_kappa"]) * 1.1)
    ax.set_ylim(0, 105)
    
    # Add regime annotations
    xlim = ax.get_xlim()
    ax.text(0.5, 102, "Well-cond.", ha="center", fontsize=10, style="italic")
    ax.text(1.5, 102, "Moderate", ha="center", fontsize=10, style="italic")
    ax.text(min(3, xlim[1] * 0.8), 102, "Severe", ha="center", fontsize=10, style="italic")
    
    # Custom legend
    handles, labels = ax.get_legend_handles_labels()
    # Add size legend manually
    from matplotlib.lines import Line2D
    size_legend = [
        Line2D([0], [0], marker='o', color='gray', markersize=8, linestyle='None', label='n=500'),
        Line2D([0], [0], marker='o', color='gray', markersize=12, linestyle='None', label='n=2000'),
    ]
    
    ax.legend(loc="lower left", framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    return fig, ax


def plot_ci_length_vs_kappa(
    cell_summary: pd.DataFrame,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 7),
):
    """
    Figure 2: CI length vs κ_DML.
    
    Scatter plot showing average CI length vs median κ_DML for each
    design cell.
    
    Parameters
    ----------
    cell_summary : pd.DataFrame
        Cell-level summary from compute_cell_summary().
    save_path : str or None
        If provided, save figure to this path.
    figsize : tuple
        Figure size in inches.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    learner_styles = {
        "LIN": {"marker": "o", "color": "#1f77b4", "label": "LIN (Ridge)"},
        "LAS": {"marker": "s", "color": "#ff7f0e", "label": "LAS (Lasso)"},
        "RF": {"marker": "^", "color": "#2ca02c", "label": "RF (Random Forest)"},
    }
    
    size_map = {500: 80, 2000: 160}
    
    for learner in ["LIN", "LAS", "RF"]:
        df_learner = cell_summary[cell_summary["learner"] == learner]
        
        for n_val in [500, 2000]:
            df_sub = df_learner[df_learner["n"] == n_val]
            
            ax.scatter(
                df_sub["median_kappa"],
                df_sub["avg_ci_length"],
                marker=learner_styles[learner]["marker"],
                color=learner_styles[learner]["color"],
                s=size_map[n_val],
                alpha=0.8,
                edgecolors="white",
                linewidths=0.5,
                label=f"{learner_styles[learner]['label']}, n={n_val}" if n_val == 500 else None,
            )
    
    # Add theoretical reference: CI length ∝ κ/√n
    # Plot reference lines
    kappa_range = np.linspace(0.5, cell_summary["median_kappa"].max() * 1.1, 100)
    
    # Labels and formatting
    ax.set_xlabel(r"Median $\kappa_{\mathrm{DML}}$", fontsize=13)
    ax.set_ylabel("Average CI Length", fontsize=13)
    ax.set_title(r"CI Length vs Condition Number $\kappa_{\mathrm{DML}}$", fontsize=14, fontweight="bold")
    
    ax.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    
    return fig, ax


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def run_full_study(
    n_list: List[int] = [500, 2000],
    R2_list: List[float] = [0.75, 0.90, 0.97],
    learners: List[LearnerType] = ["LIN", "LAS", "RF"],
    B: int = B_DEFAULT,
    save_results: bool = True,
    results_dir: str = "results",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the complete Monte Carlo study and produce all outputs.
    
    This is the main entry point for the simulation study. It:
    1. Runs the full Monte Carlo simulation
    2. Computes cell-level and regime-level summaries
    3. Produces Tables 1 and 2
    4. Creates Figures 1 and 2
    5. Optionally saves all results to disk
    
    Parameters
    ----------
    n_list : list of int, default=[500, 2000]
        Sample sizes.
    R2_list : list of float, default=[0.75, 0.90, 0.97]
        Target R²(D|X) levels.
    learners : list of str, default=["LIN", "LAS", "RF"]
        Nuisance learners.
    B : int, default=500
        Monte Carlo replications per design cell.
    save_results : bool, default=True
        Whether to save results to disk.
    results_dir : str, default="results"
        Directory for saving results.
    verbose : bool, default=True
        Print progress and summaries.
    
    Returns
    -------
    results_df : pd.DataFrame
        Raw simulation results (long format).
    cell_summary : pd.DataFrame
        Cell-level summary statistics.
    table1 : pd.DataFrame
        Table 1: Design summary.
    table2 : pd.DataFrame
        Table 2: Coverage by κ-regime.
    """
    import os
    
    # Run simulation
    results_df = run_simulation(
        n_list=n_list,
        R2_list=R2_list,
        learners=learners,
        B=B,
        verbose=verbose,
    )
    
    # Compute summaries
    cell_summary = compute_cell_summary(results_df)
    table1 = make_table1(results_df)
    table2 = make_table2(cell_summary)
    
    if verbose:
        print("\n" + "=" * 60)
        print("TABLE 1: Design Summary")
        print("=" * 60)
        print(table1.to_string(index=False))
        
        print("\n" + "=" * 60)
        print("TABLE 2: Coverage by κ-Regime and Learner")
        print("=" * 60)
        print(table2.to_string(index=False))
    
    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        
        results_df.to_csv(f"{results_dir}/raw_results.csv", index=False)
        cell_summary.to_csv(f"{results_dir}/cell_summary.csv", index=False)
        table1.to_csv(f"{results_dir}/table1.csv", index=False)
        table2.to_csv(f"{results_dir}/table2.csv", index=False)
        
        # Save LaTeX versions
        with open(f"{results_dir}/table1.tex", "w") as f:
            f.write(table_to_latex(table1, caption="Design Summary", label="tab:design"))
        with open(f"{results_dir}/table2.tex", "w") as f:
            f.write(table_to_latex(table2, caption="Coverage by Conditioning Regime", label="tab:coverage"))
        
        if verbose:
            print(f"\nResults saved to {results_dir}/")
    
    # Create figures
    try:
        plot_coverage_vs_kappa(
            cell_summary,
            save_path=f"{results_dir}/figure1_coverage_vs_kappa.pdf" if save_results else None,
        )
        plot_ci_length_vs_kappa(
            cell_summary,
            save_path=f"{results_dir}/figure2_ci_length_vs_kappa.pdf" if save_results else None,
        )
    except ImportError:
        if verbose:
            print("matplotlib not available; skipping figures.")
    
    return results_df, cell_summary, table1, table2


# =============================================================================
# MODULE-LEVEL EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "THETA0",
    "R2_TARGETS",
    "DEFAULT_SEED",
    "B_DEFAULT",
    # DGP functions
    "make_toeplitz_cov",
    "get_gamma_coeffs",
    "compute_V_gamma",
    "calibrate_sigma_xi_sq",
    "g0_function",
    "generate_plr_data",
    "DGPInfo",
    # DML estimation
    "get_nuisance_model",
    "run_dml_plr",
    "DMLResult",
    # Simulation
    "run_single_replication",
    "run_simulation",
    "ReplicationResult",
    # Summary and tables
    "compute_cell_summary",
    "compute_regime_summary",
    "assign_kappa_regime",
    "make_table1",
    "make_table2",
    "table_to_latex",
    # Visualization
    "plot_coverage_vs_kappa",
    "plot_ci_length_vs_kappa",
    # Main entry point
    "run_full_study",
]


if __name__ == "__main__":
    # Quick test with reduced replications
    print("Running quick test with B=10...")
    results_df, cell_summary, table1, table2 = run_full_study(
        n_list=[500],
        R2_list=[0.75, 0.97],
        learners=["LIN", "RF"],
        B=10,
        save_results=False,
        verbose=True,
    )
