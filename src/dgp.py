"""
Data Generating Process for DML Monte Carlo Study.

This module implements the PLR (Partially Linear Regression) DGP
matching main.tex Section 5.1 and math.tex theoretical framework.

Theoretical Foundation (from math.tex):
    Y = θ₀D + g₀(X) + ε,  E[ε|D,X] = 0
    D = m₀(X) + V,        E[V|X] = 0
    
    The condition number κ = σ_D²/σ_V² = 1/(1 - R²(D|X))
    governs both variance inflation and bias amplification.

DGP Specification (from main.tex lines 700-719):
    X ~ N(0, Σ) with AR(1) correlation, ρ = 0.5
    D = β'X + σ_U·U,  β = (0.7, 0.7², ..., 0.7^p)  [LINEAR]
    Y = θ₀D + g₀(X) + ε,  g₀(X) = X₁ + X₂² + X₃X₄  [POLYNOMIAL]
    
Key Design Choice:
    - Linear D: OLS is unbiased for m₀(X) → shows VARIANCE INFLATION only
    - Polynomial g₀: RF introduces regularization bias → shows BIAS AMPLIFICATION

Author: Gabriel Saco
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


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

def make_toeplitz_cov(p: int, rho: float) -> NDArray:
    """
    Construct Toeplitz covariance matrix Σ(ρ) with Σ_{jk} = ρ^{|j-k|}.
    
    This is an AR(1) correlation structure as specified in main.tex.
    
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
# PLR DGP (Matches main.tex Section 5.1)
# =============================================================================

@dataclass
class DGP:
    """
    Partially Linear Regression DGP matching main.tex Section 5.1.
    
    This is the CANONICAL DGP for the paper's simulations.
    
    Covariate Generation (AR(1) process, Eq. 6.3):
        X_j = ρ X_{j-1} + √(1-ρ²) η_j,  η_j ~ N(0,1),  X_0 = 0, ρ = 0.5
    
    Treatment Equation (LINEAR, Eq. 6.4):
        D = β'X + σ_U · U,  U ~ N(0,1)
        β = (0.7, 0.7², ..., 0.7^p)  # Decaying coefficients
    
    Outcome Equation (Eq. 6.5):
        Y = θ₀ D + g₀(X) + ε,  ε ~ N(0,1)
        g₀(X) = X₁ + X₂² + X₃ X₄
    
    Key Design:
        - Linear m₀(X) = β'X: OLS is unbiased, shows pure VARIANCE INFLATION
        - Polynomial g₀(X): RF introduces bias that gets AMPLIFIED by κ*
    
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
    beta : ndarray of shape (p,)
        Treatment coefficients β = (0.7^1, 0.7^2, ..., 0.7^p).
    sigma_U : float
        Treatment noise std (calibrated to achieve target_r2).
    
    References
    ----------
    main.tex, Section 5.1, Equations (6.3)-(6.5), lines 700-719
    math.tex, Section 2 (PLR Model)
    """
    p: int = P_DEFAULT
    rho: float = RHO_DEFAULT
    theta0: float = THETA0
    sigma_eps: float = SIGMA_EPS
    target_r2: float = 0.90
    random_state: Optional[int] = None
    
    # Set after initialization
    beta: NDArray = field(default=None, init=False, repr=False)
    sigma_U: float = field(default=0.5, init=False)
    is_calibrated: bool = field(default=False, init=False)
    _Sigma: NDArray = field(default=None, init=False, repr=False)
    
    def __post_init__(self) -> None:
        """Initialize β coefficients, covariance matrix, and calibrate."""
        # β = (0.7, 0.7², 0.7³, ..., 0.7^p) per main.tex Eq. 6.4
        self.beta = np.array([0.7 ** j for j in range(1, self.p + 1)])
        
        # AR(1) Toeplitz covariance matrix per main.tex Eq. 6.3
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        
        # Calibrate σ_U to achieve target R²(D|X)
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Linear propensity score: m₀(X) = E[D|X] = β'X.
        
        This is LINEAR, so OLS perfectly estimates m₀(X) (zero bias).
        """
        return X @ self.beta
    
    def g0(self, X: NDArray) -> NDArray:
        """
        Polynomial outcome function: g₀(X) = X₁ + X₂² + X₃X₄.
        
        This is NON-LINEAR (polynomial), so:
        - OLS is misspecified for g₀(X)
        - RF can learn g₀(X) but introduces regularization bias
        - Regularization bias gets AMPLIFIED by κ*
        """
        return X[:, 0] + X[:, 1]**2 + X[:, 2] * X[:, 3]
    
    def ell0(self, X: NDArray) -> NDArray:
        """
        Reduced-form outcome: ℓ₀(X) = E[Y|X] = θ₀·m₀(X) + g₀(X).
        
        From math.tex Lemma 1: ℓ₀(X) = θ₀ m₀(X) + g₀(X).
        """
        return self.theta0 * self.m0(X) + self.g0(X)
    
    def calibrate_noise(
        self,
        target_r2: float,
        n_samples: int = 100000,
        seed: int = 42
    ) -> float:
        """
        Calibrate σ_U to achieve target R²(D|X).
        
        For linear D = β'X + σ_U·U:
            Var(D) = Var(β'X) + σ_U²
            R²(D|X) = Var(β'X) / Var(D) = Var(m₀) / (Var(m₀) + σ_U²)
        
        Solving for σ_U²:
            σ_U² = Var(m₀) × (1 - R²) / R²
        
        Parameters
        ----------
        target_r2 : float
            Target R²(D|X), must be in (0, 1).
            - R² = 0.75 → κ* ≈ 4 (well-conditioned)
            - R² = 0.90 → κ* ≈ 10 (moderate)
            - R² = 0.97 → κ* ≈ 33 (severely ill-conditioned)
        n_samples : int, default 100000
            Number of samples for Monte Carlo variance estimation.
        seed : int, default 42
            Random seed for calibration.
        
        Returns
        -------
        sigma_U : float
            Calibrated treatment noise std.
        """
        if not 0 < target_r2 < 1:
            raise ValueError(f"target_r2 must be in (0, 1), got {target_r2}")
        
        rng = np.random.default_rng(seed)
        X_calib = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n_samples)
        
        # Compute Var(β'X) = Var(m₀(X))
        m0_X = self.m0(X_calib)
        var_m0 = np.var(m0_X)
        
        # Solve for σ_U²: R² = Var(m₀) / (Var(m₀) + σ_U²)
        # => σ_U² = Var(m₀) × (1 - R²) / R²
        sigma_U_sq = var_m0 * (1 - target_r2) / target_r2
        self.sigma_U = np.sqrt(sigma_U_sq)
        
        self.target_r2 = target_r2
        self.is_calibrated = True
        
        return self.sigma_U
    
    def generate(
        self,
        n: int,
        random_state: Optional[int] = None
    ) -> Tuple[NDArray, NDArray, NDArray, Dict]:
        """
        Generate data from the PLR DGP.
        
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
            Dictionary with calibration info and true nuisance values:
            - target_r2: target R²(D|X)
            - sample_r2: realized sample R²(D|X)
            - sigma_U: treatment noise std
            - m0_X: true propensity scores m₀(X)
            - g0_X: true outcome function values g₀(X)
            - ell0_X: true reduced-form outcome values ℓ₀(X)
            - beta: coefficient vector
        """
        if not self.is_calibrated:
            raise RuntimeError("DGP not calibrated. Call calibrate_noise() first.")
        
        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)
        
        # Generate covariates X ~ N(0, Σ) with AR(1) correlation
        X = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=n)
        
        # Treatment equation: D = β'X + σ_U · U (LINEAR)
        m0_X = self.m0(X)
        U = rng.normal(0, 1, size=n)
        D = m0_X + self.sigma_U * U
        
        # Outcome equation: Y = θ₀D + g₀(X) + ε
        g0_X = self.g0(X)
        eps = rng.normal(0, self.sigma_eps, size=n)
        Y = self.theta0 * D + g0_X + eps
        
        # Compute sample R²(D|X) = Var(m₀) / Var(D)
        var_D = np.var(D)
        var_m0 = np.var(m0_X)
        sample_r2 = var_m0 / var_D if var_D > 0 else 0.0
        
        # Reduced-form ℓ₀(X) = E[Y|X]
        ell0_X = self.ell0(X)
        
        info = {
            'target_r2': self.target_r2,
            'sample_r2': sample_r2,
            'sigma_U': self.sigma_U,
            'm0_X': m0_X,
            'g0_X': g0_X,
            'ell0_X': ell0_X,
            'beta': self.beta.copy(),
        }
        
        return Y, D, X, info


def generate_data(
    n: int,
    target_r2: float,
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, DGP]:
    """
    Convenience function to generate data from the PLR DGP.
    
    This is the CANONICAL function for the paper's simulations.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X) in (0, 1). Higher values → higher κ* → weaker overlap.
        - 0.75 → κ* ≈ 4 (well-conditioned, main.tex "High overlap")
        - 0.90 → κ* ≈ 10 (moderate, main.tex "Moderate overlap")
        - 0.97 → κ* ≈ 33 (severe, main.tex "Low overlap")
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
    dgp : DGP
        The DGP object (useful for Oracle learner).
    
    Examples
    --------
    >>> Y, D, X, info, dgp = generate_data(n=1000, target_r2=0.90)
    >>> print(f"κ* ≈ {1/(1 - info['sample_r2']):.1f}")
    κ* ≈ 10.0
    """
    dgp = DGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    
    return Y, D, X, info, dgp


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================

# For backwards compatibility with existing code
LinearPolynomialDGP = DGP
generate_linear_polynomial_data = generate_data


# =============================================================================
# NON-LINEAR DGP (For Robustness Testing)
# =============================================================================

@dataclass
class NonLinearDGP(DGP):
    """
    Non-Linear DGP variant for robustness testing.
    
    This DGP uses a non-linear treatment equation to test whether the
    κ* mechanism holds beyond the linear propensity score case.
    
    Treatment Equation (NON-LINEAR):
        D = tanh(β'X) × scale + σ_U · U
        
    The tanh function creates non-linearities that:
    1. Test whether RF can learn m₀(X) better than OLS
    2. Test whether κ* still predicts bias amplification
    
    Everything else remains the same as the linear DGP.
    
    Parameters (same as DGP plus):
    ----------
    nonlinearity : str, default 'tanh'
        Type of nonlinearity: 'tanh', 'sin', or 'interaction'
    """
    nonlinearity: str = 'tanh'
    _scale: float = field(default=1.0, init=False)
    
    def __post_init__(self) -> None:
        """Initialize with non-linear calibration."""
        # β = (0.7, 0.7², 0.7³, ..., 0.7^p) per main.tex Eq. 6.4
        self.beta = np.array([0.7 ** j for j in range(1, self.p + 1)])
        
        # AR(1) Toeplitz covariance matrix
        self._Sigma = make_toeplitz_cov(self.p, self.rho)
        
        # Estimate scale for nonlinear transformation
        rng = np.random.default_rng(42)
        X_cal = rng.multivariate_normal(np.zeros(self.p), self._Sigma, size=10000)
        linear_part = X_cal @ self.beta
        
        if self.nonlinearity == 'tanh':
            self._scale = np.std(linear_part) / np.std(np.tanh(linear_part))
        elif self.nonlinearity == 'sin':
            self._scale = np.std(linear_part) / np.std(np.sin(linear_part))
        else:
            self._scale = 1.0
        
        # Calibrate σ_U to achieve target R²(D|X)
        self.calibrate_noise(self.target_r2)
    
    def m0(self, X: NDArray) -> NDArray:
        """
        Non-linear propensity score: m₀(X) = f(β'X).
        
        Uses tanh, sin, or interaction non-linearity.
        """
        linear_part = X @ self.beta
        
        if self.nonlinearity == 'tanh':
            return np.tanh(linear_part) * self._scale
        elif self.nonlinearity == 'sin':
            return np.sin(linear_part) * self._scale
        elif self.nonlinearity == 'interaction':
            # Add interaction terms: X₁X₂ + X₃X₄
            interactions = X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3]
            return linear_part + 0.5 * interactions
        else:
            return linear_part


def generate_nonlinear_data(
    n: int,
    target_r2: float,
    nonlinearity: str = 'tanh',
    p: int = P_DEFAULT,
    rho: float = RHO_DEFAULT,
    theta0: float = THETA0,
    random_state: Optional[int] = None,
) -> Tuple[NDArray, NDArray, NDArray, Dict, NonLinearDGP]:
    """
    Generate data from the Non-Linear DGP.
    
    Used for robustness testing to verify κ* mechanism works
    beyond the linear propensity score case.
    
    Parameters
    ----------
    n : int
        Sample size.
    target_r2 : float
        Target R²(D|X).
    nonlinearity : str, default 'tanh'
        Type of nonlinearity: 'tanh', 'sin', or 'interaction'
    p, rho, theta0, random_state : same as generate_data
    
    Returns
    -------
    Y, D, X, info, dgp : same as generate_data
    """
    dgp = NonLinearDGP(
        p=p,
        rho=rho,
        theta0=theta0,
        target_r2=target_r2,
        random_state=random_state,
        nonlinearity=nonlinearity,
    )
    
    Y, D, X, info = dgp.generate(n, random_state=random_state)
    info['nonlinearity'] = nonlinearity
    
    return Y, D, X, info, dgp


__all__ = [
    'DGP',
    'NonLinearDGP',
    'generate_data',
    'generate_nonlinear_data',
    'make_toeplitz_cov',
    'THETA0',
    'P_DEFAULT',
    'RHO_DEFAULT',
    'SIGMA_EPS',
    # Backwards compatibility
    'LinearPolynomialDGP',
    'generate_linear_polynomial_data',
]
