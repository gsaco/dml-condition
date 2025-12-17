"""
Non-Linear Data Generating Process for DML Monte Carlo Study.

This module implements a Non-Linear DGP designed to break OLS (via misspecification)
while remaining learnable by flexible ML methods. The key feature is the ability
to calibrate the noise variance to achieve a target R²(D|X), controlling overlap.

Theoretical Foundation:
    The treatment equation uses a non-linear propensity score:
        D = m₀(X) + V, where V ~ N(0, σ_V²)
        m₀(X) = sigmoid(C · (X₁X₂ + X₃² - X₄ + 0.5X₅))
    
    The outcome equation includes non-linear confounding:
        Y = θ₀D + g₀(X) + ε, where ε ~ N(0, σ_ε²)
        g₀(X) = sin(X₁) + cos(X₂) + (X₃ · X₄)

Author: DML Monte Carlo Study
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import optimize


# =============================================================================
# CONSTANTS
# =============================================================================

THETA0: float = 1.0          # True treatment effect
P_DEFAULT: int = 10          # Covariate dimension
RHO_DEFAULT: float = 0.5     # AR(1) correlation
SIGMA_EPS: float = 1.0       # Outcome noise std


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sigmoid(x: NDArray) -> NDArray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def make_toeplitz_cov(p: int, rho: float) -> NDArray:
    """
    Construct Toeplitz covariance matrix Σ(ρ) with Σ_{jk} = ρ^{|j-k|}.
    
    This is an AR(1) correlation structure.
    
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
# NON-LINEAR DGP CLASS
# =============================================================================

@dataclass
class NonLinearDGP:
    """
    Non-Linear Data Generating Process for DML Monte Carlo Study.
    
    Generates data (Y, D, X) where:
        - X ~ N(0, Σ) with AR(1) correlation structure
        - D = m₀(X) + V with non-linear propensity score
        - Y = θ₀D + g₀(X) + ε with non-linear outcome function
    
    Parameters
    ----------
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    sigma_eps : float, default 1.0
        Standard deviation of outcome noise.
    target_r2 : float, default 0.90
        Target R²(D|X) for overlap calibration.
    random_state : int or None, default None
        Random seed for reproducibility.
    
    Attributes
    ----------
    C : float
        Scaling factor for propensity score (set after calibration).
    sigma_V : float
        Treatment noise std (set after calibration).
    is_calibrated : bool
        Whether calibrate_noise() has been called.
    """
    p: int = P_DEFAULT
    rho: float = RHO_DEFAULT
    theta0: float = THETA0
    sigma_eps: float = SIGMA_EPS
    target_r2: float = 0.90
    random_state: Optional[int] = None
    
    # Set after calibration
    C: float = field(default=1.0, init=False)
    sigma_V: float = field(default=0.5, init=False)
    is_calibrated: bool = field(default=False, init=False)
    _Sigma: NDArray = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize covariance matrix and calibrate noise."""
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Non-linear propensity score function m₀(X) = E[D|X].
        
        m₀(X) = sigmoid(C · (X₁X₂ + X₃² - X₄ + 0.5X₅))
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        m0_X : ndarray of shape (n,)
            Propensity score values.
        """
        linear_index = (
            X[:, 0] * X[:, 1] +          # X₁ · X₂
            X[:, 2] ** 2 -               # X₃²
            X[:, 3] +                    # -X₄
            0.5 * X[:, 4]                # 0.5 · X₅
        )
        return sigmoid(self.C * linear_index)
    
    def g0(self, X: NDArray) -> NDArray:
        """
        Non-linear outcome function g₀(X) = E[Y|D=0, X].
        
        g₀(X) = sin(X₁) + cos(X₂) + (X₃ · X₄)
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        g0_X : ndarray of shape (n,)
            Conditional outcome values.
        """
        return (
            np.sin(X[:, 0]) +            # sin(X₁)
            np.cos(X[:, 1]) +            # cos(X₂)
            X[:, 2] * X[:, 3]            # X₃ · X₄
        )
    
    def ell0(self, X: NDArray) -> NDArray:
        """
        Reduced-form outcome function ℓ₀(X) = E[Y|X] = θ₀·m₀(X) + g₀(X).
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        ell0_X : ndarray of shape (n,)
            Reduced-form outcome values.
        """
        return self.theta0 * self.m0(X) + self.g0(X)
    
    def calibrate_noise(
        self,
        target_r2: float,
        n_samples: int = 100000,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Numerically calibrate C and σ_V to achieve target population R²(D|X).
        
        The calibration works by:
        1. Generating a large sample of X
        2. For various values of C, compute Var(m₀(X))
        3. Solve for σ_V² such that Var(m₀(X)) / (Var(m₀(X)) + σ_V²) = target_r2
        
        Parameters
        ----------
        target_r2 : float
            Target R²(D|X), must be in (0, 1).
        n_samples : int, default 100000
            Number of samples for Monte Carlo estimation.
        seed : int, default 42
            Random seed for calibration.
        
        Returns
        -------
        C : float
            Calibrated scaling factor.
        sigma_V : float
            Calibrated treatment noise std.
        """
        if not 0 < target_r2 < 1:
            raise ValueError(f"target_r2 must be in (0, 1), got {target_r2}")
        
        rng = np.random.default_rng(seed)
        X_calib = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n_samples)
        
        def objective(C_val: float) -> float:
            """Find C such that we can achieve target R²."""
            # Temporarily set C
            old_C = self.C
            self.C = C_val
            
            m0_X = self.m0(X_calib)
            var_m0 = np.var(m0_X)
            
            # Restore C
            self.C = old_C
            
            # For target R², we need: R² = Var(m₀) / Var(D) = Var(m₀) / (Var(m₀) + σ_V²)
            # Solving: σ_V² = Var(m₀) * (1 - R²) / R²
            # We want Var(m₀) to be reasonable (not too small), so we minimize
            # the difference from a target variance
            target_var_m0 = 0.1  # Reasonable variance for propensity score
            return (var_m0 - target_var_m0) ** 2
        
        # First, find a good C value
        result = optimize.minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.C = result.x
        
        # Now compute the actual variance of m₀(X)
        m0_X = self.m0(X_calib)
        var_m0 = np.var(m0_X)
        
        # Calculate σ_V² to achieve target R²
        # R² = Var(m₀) / (Var(m₀) + σ_V²)
        # => σ_V² = Var(m₀) * (1 - R²) / R²
        sigma_V_sq = var_m0 * (1 - target_r2) / target_r2
        self.sigma_V = np.sqrt(sigma_V_sq)
        
        self.target_r2 = target_r2
        self.is_calibrated = True
        
        return self.C, self.sigma_V
    
    def generate(
        self,
        n: int,
        random_state: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Generate data from the Non-Linear DGP.
        
        Parameters
        ----------
        n : int
            Sample size.
        random_state : int or None, default None
            Random seed for reproducibility.
        
        Returns
        -------
        Y : ndarray of shape (n,)
            Outcome variable.
        D : ndarray of shape (n,)
            Treatment variable.
        X : ndarray of shape (n, p)
            Covariate matrix.
        info : dict
            Dictionary with calibration info:
            - target_r2: target R²(D|X)
            - sample_r2: realized R²(D|X)
            - C: scaling factor
            - sigma_V: treatment noise std
            - m0_X: true propensity scores
            - g0_X: true outcome function values
            - ell0_X: true reduced-form outcome values
        """
        if not self.is_calibrated:
            raise RuntimeError("DGP not calibrated. Call calibrate_noise() first.")
        
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)
        
        # Generate covariates X ~ N(0, Σ)
        X = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n)
        
        # Treatment equation: D = m₀(X) + V
        m0_X = self.m0(X)
        V = rng.normal(0, self.sigma_V, size=n)
        D = m0_X + V
        
        # Outcome equation: Y = θ₀D + g₀(X) + ε
        g0_X = self.g0(X)
        eps = rng.normal(0, self.sigma_eps, size=n)
        Y = self.theta0 * D + g0_X + eps
        
        # Compute sample R²(D|X)
        var_D = np.var(D)
        var_m0 = np.var(m0_X)
        sample_r2 = var_m0 / var_D if var_D > 0 else 0.0
        
        # Compute ℓ₀(X) = E[Y|X]
        ell0_X = self.ell0(X)
        
        info = {
            'target_r2': self.target_r2,
            'sample_r2': sample_r2,
            'C': self.C,
            'sigma_V': self.sigma_V,
            'm0_X': m0_X,
            'g0_X': g0_X,
            'ell0_X': ell0_X,
        }
        
        return Y, D, X, info


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def generate_nonlinear_data(
    n: int,
    target_r2: float,
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, NonLinearDGP]:
    """
    Convenience function to generate data from the Non-Linear DGP.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X) in (0, 1).
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    Y : ndarray of shape (n,)
        Outcome.
    D : ndarray of shape (n,)
        Treatment.
    X : ndarray of shape (n, p)
        Covariates.
    info : dict
        Calibration and diagnostic info.
    dgp : NonLinearDGP
        The DGP object (useful for Oracle learner).
    """
    dgp = NonLinearDGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    
    return Y, D, X, info, dgp


# =============================================================================
# BINARY TREATMENT DGP (JCI Enhancement #1)
# =============================================================================

@dataclass
class BinaryTreatmentDGP:
    """
    Binary Treatment DGP for canonical causal inference setting.
    
    This DGP generates binary treatment D ∈ {0, 1} with propensity-score-based
    overlap control, directly connecting to the propensity score literature.
    
    Treatment Assignment:
        e(X) = sigmoid(C · linear_index(X))  # Propensity score
        D | X ~ Bernoulli(e(X))
    
    Overlap Control:
        κ* is controlled by propensity score variance:
        - High C → extreme propensities (near 0 or 1) → poor overlap → high κ*
        - Low C → moderate propensities (near 0.5) → good overlap → low κ*
    
    Outcome Equation:
        Y = θ₀D + g₀(X) + ε, where ε ~ N(0, σ_ε²)
        g₀(X) = sin(X₁) + cos(X₂) + (X₃ · X₄)
    
    Parameters
    ----------
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    sigma_eps : float, default 1.0
        Standard deviation of outcome noise.
    target_overlap : float, default 0.90
        Target overlap parameter. Controls the minimum propensity score bound.
        Higher values → better overlap → lower κ*.
        Specifically: min_propensity ≈ (1 - target_overlap) / 2
    random_state : int or None, default None
        Random seed for reproducibility.
    """
    p: int = P_DEFAULT
    rho: float = RHO_DEFAULT
    theta0: float = THETA0
    sigma_eps: float = SIGMA_EPS
    target_overlap: float = 0.90
    random_state: Optional[int] = None
    
    # Set after calibration
    C: float = field(default=1.0, init=False)
    is_calibrated: bool = field(default=False, init=False)
    _Sigma: NDArray = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize covariance matrix and calibrate propensity scaling."""
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        self.calibrate_propensity(self.target_overlap)
    
    def propensity_score(self, X: NDArray) -> NDArray:
        """
        Propensity score function e(X) = P(D=1|X).
        
        Uses same linear index as NonLinearDGP's m₀(X) for consistency.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        e_X : ndarray of shape (n,)
            Propensity scores in (0, 1).
        """
        linear_index = (
            X[:, 0] * X[:, 1] +          # X₁ · X₂
            X[:, 2] ** 2 -               # X₃²
            X[:, 3] +                    # -X₄
            0.5 * X[:, 4]                # 0.5 · X₅
        )
        return sigmoid(self.C * linear_index)
    
    def m0(self, X: NDArray) -> NDArray:
        """E[D|X] for binary treatment equals propensity score."""
        return self.propensity_score(X)
    
    def g0(self, X: NDArray) -> NDArray:
        """
        Outcome function g₀(X) = E[Y|D=0, X].
        
        Same form as NonLinearDGP for comparability.
        """
        return (
            np.sin(X[:, 0]) +            # sin(X₁)
            np.cos(X[:, 1]) +            # cos(X₂)
            X[:, 2] * X[:, 3]            # X₃ · X₄
        )
    
    def ell0(self, X: NDArray) -> NDArray:
        """Reduced-form ℓ₀(X) = E[Y|X] = θ₀·e(X) + g₀(X)."""
        return self.theta0 * self.propensity_score(X) + self.g0(X)
    
    def calibrate_propensity(
        self,
        target_overlap: float,
        n_samples: int = 100000,
        seed: int = 42
    ) -> float:
        """
        Calibrate C to achieve target propensity score overlap.
        
        Parameters
        ----------
        target_overlap : float
            Target overlap in (0.5, 1.0). Higher = better overlap.
            - 0.99: Excellent overlap (κ* ~ 1-2)
            - 0.90: Good overlap (κ* ~ 4-10)
            - 0.75: Moderate overlap (κ* ~ 10-20)
            - 0.50: Poor overlap (κ* ~ 20-100)
        
        Returns
        -------
        C : float
            Calibrated propensity scaling factor.
        """
        if not 0.5 < target_overlap < 1.0:
            raise ValueError(f"target_overlap must be in (0.5, 1.0), got {target_overlap}")
        
        rng = np.random.default_rng(seed)
        X_calib = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n_samples)
        
        # Target: propensity scores between (1-target)/2 and 1-(1-target)/2
        # E.g., target_overlap=0.90 → propensities in [0.05, 0.95]
        min_prop = (1 - target_overlap) / 2
        max_prop = 1 - min_prop
        
        def objective(C_val: float) -> float:
            """Find C such that propensity scores stay within [min_prop, max_prop]."""
            self_C_old = self.C
            self.C = C_val
            e_X = self.propensity_score(X_calib)
            self.C = self_C_old
            
            # Fraction of propensities in valid range
            in_range = np.mean((e_X >= min_prop) & (e_X <= max_prop))
            # We want about target_overlap fraction to be in range
            return (in_range - target_overlap) ** 2
        
        result = optimize.minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
        self.C = result.x
        self.target_overlap = target_overlap
        self.is_calibrated = True
        
        return self.C
    
    def generate(
        self,
        n: int,
        random_state: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Generate data from the Binary Treatment DGP.
        
        Returns
        -------
        Y : ndarray of shape (n,)
            Outcome variable.
        D : ndarray of shape (n,)
            Binary treatment (0 or 1).
        X : ndarray of shape (n, p)
            Covariate matrix.
        info : dict
            Dictionary with calibration info.
        """
        if not self.is_calibrated:
            raise RuntimeError("DGP not calibrated.")
        
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)
        
        # Generate covariates X ~ N(0, Σ)
        X = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n)
        
        # Propensity score e(X) = P(D=1|X)
        e_X = self.propensity_score(X)
        
        # Treatment: D ~ Bernoulli(e(X))
        D = rng.binomial(1, e_X).astype(float)
        
        # Outcome: Y = θ₀D + g₀(X) + ε
        g0_X = self.g0(X)
        eps = rng.normal(0, self.sigma_eps, size=n)
        Y = self.theta0 * D + g0_X + eps
        
        # Compute κ* = 1 / (1 - R²(D|X))
        # For binary D, Var(D) = p(1-p) where p = mean(D)
        # Var(e(X)) is the signal variance
        var_e = np.var(e_X)
        # R²(D|X) ≈ Var(e(X)) / Var(D)
        p_hat = D.mean()
        var_D = p_hat * (1 - p_hat) if 0 < p_hat < 1 else 0.001
        sample_r2 = var_e / var_D if var_D > 0 else 0.0
        sample_r2 = np.clip(sample_r2, 0, 0.999)  # Bound to avoid inf
        
        # Reduced-form ℓ₀(X)
        ell0_X = self.ell0(X)
        
        info = {
            'target_overlap': self.target_overlap,
            'sample_r2': sample_r2,
            'C': self.C,
            'e_X': e_X,  # Propensity scores
            'm0_X': e_X,  # For Oracle learner compatibility
            'g0_X': g0_X,
            'ell0_X': ell0_X,
            'mean_propensity': e_X.mean(),
            'min_propensity': e_X.min(),
            'max_propensity': e_X.max(),
            'treatment_frac': D.mean(),
        }
        
        return Y, D, X, info


def generate_binary_treatment_data(
    n: int,
    target_overlap: float,
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, BinaryTreatmentDGP]:
    """
    Convenience function to generate data from the Binary Treatment DGP.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_overlap : float
        Target propensity score overlap in (0.5, 1.0).
    p : int, default 10
        Covariate dimension.
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    Y, D, X, info, dgp
    """
    dgp = BinaryTreatmentDGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_overlap=target_overlap,
        random_state=random_state,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    
    return Y, D, X, info, dgp


# =============================================================================
# HIGH-DIMENSIONAL DGP (JCI Enhancement #2)
# =============================================================================

@dataclass
class HighDimensionalDGP:
    """
    High-Dimensional DGP (p=100) with Sparse Signal.
    
    This DGP tests how κ* interacts with dimensionality and regularization.
    Only s=5 covariates are active in the propensity/outcome functions,
    simulating realistic high-dimensional settings.
    
    Treatment Equation:
        m₀(X) = sigmoid(C · Σ_{j∈S} β_j X_j)  # Sparse linear index
        D = m₀(X) + V, where V ~ N(0, σ_V²)
    
    Outcome Equation:
        g₀(X) = Σ_{j∈S} γ_j X_j + X_{s+1}²  # Sparse + one nonlinear term
        Y = θ₀D + g₀(X) + ε
    
    Parameters
    ----------
    p : int, default 100
        Covariate dimension.
    s : int, default 5
        Sparsity (number of active covariates).
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    sigma_eps : float, default 1.0
        Standard deviation of outcome noise.
    target_r2 : float, default 0.90
        Target R²(D|X) for overlap calibration.
    random_state : int or None, default None
        Random seed for reproducibility.
    """
    p: int = 100
    s: int = 5
    rho: float = RHO_DEFAULT
    theta0: float = THETA0
    sigma_eps: float = SIGMA_EPS
    target_r2: float = 0.90
    random_state: Optional[int] = None
    
    # Set after initialization
    beta: NDArray = field(default=None, init=False, repr=False)  # Sparse propensity coefficients
    gamma: NDArray = field(default=None, init=False, repr=False)  # Sparse outcome coefficients
    C: float = field(default=1.0, init=False)
    sigma_V: float = field(default=0.5, init=False)
    is_calibrated: bool = field(default=False, init=False)
    _Sigma: NDArray = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize sparse coefficients and covariance matrix."""
        if self.p < self.s:
            raise ValueError(f"p ({self.p}) must be >= s ({self.s})")
        
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        
        # Sparse propensity coefficients: first s covariates active
        # β = [1, -0.5, 0.8, -0.3, 0.6, 0, 0, ..., 0]
        self.beta = np.zeros(self.p)
        self.beta[:self.s] = np.array([1.0, -0.5, 0.8, -0.3, 0.6])[:self.s]
        
        # Sparse outcome coefficients: similar sparsity pattern
        self.gamma = np.zeros(self.p)
        self.gamma[:self.s] = np.array([0.8, 0.6, -0.4, 0.5, -0.3])[:self.s]
        
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Propensity score m₀(X) = E[D|X] = sigmoid(C · X @ β).
        """
        linear_index = X @ self.beta
        return sigmoid(self.C * linear_index)
    
    def g0(self, X: NDArray) -> NDArray:
        """
        Outcome function g₀(X) = X @ γ + X_{s+1}² (one nonlinear term).
        """
        linear_part = X @ self.gamma
        # Add one nonlinear term using covariate just after the sparse set
        nonlinear_idx = min(self.s, self.p - 1)
        nonlinear_part = X[:, nonlinear_idx] ** 2
        return linear_part + nonlinear_part
    
    def ell0(self, X: NDArray) -> NDArray:
        """Reduced-form ℓ₀(X) = θ₀·m₀(X) + g₀(X)."""
        return self.theta0 * self.m0(X) + self.g0(X)
    
    def calibrate_noise(
        self,
        target_r2: float,
        n_samples: int = 100000,
        seed: int = 42
    ) -> Tuple[float, float]:
        """
        Calibrate C and σ_V to achieve target R²(D|X).
        
        Same approach as NonLinearDGP.
        """
        if not 0 < target_r2 < 1:
            raise ValueError(f"target_r2 must be in (0, 1), got {target_r2}")
        
        rng = np.random.default_rng(seed)
        X_calib = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n_samples)
        
        def objective(C_val: float) -> float:
            old_C = self.C
            self.C = C_val
            m0_X = self.m0(X_calib)
            var_m0 = np.var(m0_X)
            self.C = old_C
            target_var_m0 = 0.1
            return (var_m0 - target_var_m0) ** 2
        
        result = optimize.minimize_scalar(objective, bounds=(0.1, 10.0), method='bounded')
        self.C = result.x
        
        m0_X = self.m0(X_calib)
        var_m0 = np.var(m0_X)
        sigma_V_sq = var_m0 * (1 - target_r2) / target_r2
        self.sigma_V = np.sqrt(sigma_V_sq)
        
        self.target_r2 = target_r2
        self.is_calibrated = True
        
        return self.C, self.sigma_V
    
    def generate(
        self,
        n: int,
        random_state: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Generate data from the High-Dimensional DGP.
        """
        if not self.is_calibrated:
            raise RuntimeError("DGP not calibrated.")
        
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)
        
        # Generate covariates X ~ N(0, Σ)
        X = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n)
        
        # Treatment equation: D = m₀(X) + V
        m0_X = self.m0(X)
        V = rng.normal(0, self.sigma_V, size=n)
        D = m0_X + V
        
        # Outcome equation: Y = θ₀D + g₀(X) + ε
        g0_X = self.g0(X)
        eps = rng.normal(0, self.sigma_eps, size=n)
        Y = self.theta0 * D + g0_X + eps
        
        # Compute sample R²(D|X)
        var_D = np.var(D)
        var_m0 = np.var(m0_X)
        sample_r2 = var_m0 / var_D if var_D > 0 else 0.0
        
        # Reduced-form ℓ₀(X)
        ell0_X = self.ell0(X)
        
        info = {
            'target_r2': self.target_r2,
            'sample_r2': sample_r2,
            'C': self.C,
            'sigma_V': self.sigma_V,
            'm0_X': m0_X,
            'g0_X': g0_X,
            'ell0_X': ell0_X,
            'p': self.p,
            's': self.s,
            'beta': self.beta.copy(),
            'gamma': self.gamma.copy(),
        }
        
        return Y, D, X, info


def generate_highdim_data(
    n: int,
    target_r2: float,
    p: int = 100,
    s: int = 5,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, HighDimensionalDGP]:
    """
    Convenience function to generate data from the High-Dimensional DGP.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X) in (0, 1).
    p : int, default 100
        Covariate dimension.
    s : int, default 5
        Sparsity (number of active covariates).
    rho : float, default 0.5
        AR(1) correlation parameter.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int or None
        Random seed.
    
    Returns
    -------
    Y, D, X, info, dgp
    """
    dgp = HighDimensionalDGP(
        p=p,
        s=s,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    
    return Y, D, X, info, dgp
