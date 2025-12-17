"""
DML Estimator for Monte Carlo Study.

This module implements the Double Machine Learning estimator with:
    - 2-Stage Partialling Out with K=5 fold cross-fitting
    - Repeated Cross-Fitting with median aggregation
    - Comprehensive diagnostic output including κ* and nuisance MSEs

Theoretical Foundation:
    The DML estimator uses the orthogonal score for PLR:
        ψ(W; θ, η) = (Y - ℓ(X) - θ(D - m(X))) · (D - m(X))
    
    The standardized condition number is:
        κ* = 1 / (1 - R²(D|X)) = Σ(Dᵢ - D̄)² / Σ V̂ᵢ²
    
    This captures identification strength and amplifies both variance and bias.

Author: DML Monte Carlo Study
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from src.dgp import NonLinearDGP
    from src.learners import OracleLearner


# =============================================================================
# CONSTANTS
# =============================================================================

THETA0: float = 1.0          # True treatment effect
K_FOLDS: int = 5             # Number of cross-fitting folds
N_REPEATS: int = 5           # Number of repeated cross-fitting iterations
Z_ALPHA: float = 1.96        # Critical value for 95% CI


# =============================================================================
# DML RESULT CONTAINER
# =============================================================================

@dataclass
class DMLResult:
    """
    Container for DML estimation results with forensic diagnostics.
    
    Attributes
    ----------
    theta_hat : float
        Estimated treatment effect.
    se : float
        Standard error.
    ci_lower : float
        Lower bound of 95% CI.
    ci_upper : float
        Upper bound of 95% CI.
    kappa_star : float
        Standardized condition number κ* = 1 / (1 - R²(D|X)).
    bias : float
        Estimated bias = θ̂ - θ₀.
    nuisance_mse_m : float
        MSE of m̂(X) relative to true m₀(X).
    nuisance_mse_l : float
        MSE of ℓ̂(X) relative to true ℓ₀(X).
    sample_r2 : float
        Sample R²(D|X).
    n : int
        Sample size.
    """
    theta_hat: float
    se: float
    ci_lower: float
    ci_upper: float
    kappa_star: float
    bias: float
    nuisance_mse_m: float
    nuisance_mse_l: float
    sample_r2: float
    n: int
    
    @property
    def rmse_m(self) -> float:
        """RMSE of m̂(X) relative to true m₀(X)."""
        import numpy as np
        return np.sqrt(self.nuisance_mse_m) if not np.isnan(self.nuisance_mse_m) else np.nan
    
    @property
    def rmse_l(self) -> float:
        """RMSE of ℓ̂(X) relative to true ℓ₀(X)."""
        import numpy as np
        return np.sqrt(self.nuisance_mse_l) if not np.isnan(self.nuisance_mse_l) else np.nan
    
    @property
    def ci_length(self) -> float:
        """Length of 95% confidence interval."""
        return self.ci_upper - self.ci_lower
    
    def covers(self, theta0: float = THETA0) -> bool:
        """Check if CI covers true parameter."""
        return self.ci_lower <= theta0 <= self.ci_upper
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'theta_hat': self.theta_hat,
            'se': self.se,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_length': self.ci_length,
            'kappa_star': self.kappa_star,
            'bias': self.bias,
            'nuisance_mse_m': self.nuisance_mse_m,
            'nuisance_mse_l': self.nuisance_mse_l,
            'rmse_m': self.rmse_m,
            'rmse_l': self.rmse_l,
            'sample_r2': self.sample_r2,
            'n': self.n,
        }


# =============================================================================
# DML ESTIMATOR CLASS
# =============================================================================

class DMLEstimator:
    """
    Double Machine Learning Estimator with Forensic Diagnostics.
    
    Implements 2-stage partialling out with K-fold cross-fitting and
    optional repeated cross-fitting for variance reduction.
    
    Parameters
    ----------
    learner_m : BaseEstimator
        Learner for treatment model m(X) = E[D|X].
    learner_l : BaseEstimator
        Learner for outcome model ℓ(X) = E[Y|X].
    K : int, default 5
        Number of cross-fitting folds.
    n_repeats : int, default 1
        Number of repeated cross-fitting iterations.
        If > 1, uses median aggregation.
    theta0 : float, default 1.0
        True treatment effect (for bias calculation).
    random_state : int, default 42
        Random state for reproducibility.
    
    Attributes
    ----------
    result_ : DMLResult or None
        Estimation results after calling fit().
    """
    
    def __init__(
        self,
        learner_m: BaseEstimator,
        learner_l: BaseEstimator,
        K: int = K_FOLDS,
        n_repeats: int = 1,
        theta0: float = THETA0,
        random_state: int = 42,
    ) -> None:
        self.learner_m = learner_m
        self.learner_l = learner_l
        self.K = K
        self.n_repeats = n_repeats
        self.theta0 = theta0
        self.random_state = random_state
        self.result_: Optional[DMLResult] = None
    
    def _single_crossfit(
        self,
        Y: NDArray,
        D: NDArray,
        X: NDArray,
        m0_X: Optional[NDArray],
        ell0_X: Optional[NDArray],
        split_seed: int,
    ) -> Tuple[float, float, float, float, float]:
        """
        Perform a single cross-fitting iteration.
        
        Returns
        -------
        theta_hat : float
        var_hat : float
        kappa_star : float
        mse_m : float
        mse_l : float
        """
        n = len(Y)
        m_hat = np.zeros(n)
        l_hat = np.zeros(n)
        
        kf = KFold(n_splits=self.K, shuffle=True, random_state=split_seed)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            D_train, Y_train = D[train_idx], Y[train_idx]
            
            # Clone learners for each fold
            model_m = clone(self.learner_m)
            model_l = clone(self.learner_l)
            
            # Fit m̂(X) = E[D|X]
            model_m.fit(X_train, D_train)
            m_hat[test_idx] = model_m.predict(X_test)
            
            # Fit ℓ̂(X) = E[Y|X]
            model_l.fit(X_train, Y_train)
            l_hat[test_idx] = model_l.predict(X_test)
        
        # Compute residuals
        V_hat = D - m_hat  # Residualized treatment
        U_hat = Y - l_hat  # Residualized outcome
        
        # DML estimate: θ̂ = Σ(V̂·Ũ) / Σ(V̂²) where Ũ = Y - ℓ̂(X)
        # But we use the formula: θ̂ = Σ(V̂ · (Y - ℓ̂(X))) / Σ(V̂²)
        sum_V_sq = np.sum(V_hat ** 2)
        
        if sum_V_sq < 1e-12:
            return np.nan, np.nan, np.inf, np.nan, np.nan
        
        theta_hat = np.sum(V_hat * U_hat) / sum_V_sq
        
        # Final residuals for variance estimation
        eps_hat = U_hat - theta_hat * V_hat
        
        # Variance estimation (heteroskedasticity-robust)
        var_component = np.mean(V_hat ** 2 * eps_hat ** 2)
        var_hat = (n / sum_V_sq) ** 2 * var_component / n
        
        # Standardized condition number: κ* = 1 / (1 - R²(D|X))
        # κ* = n·Var(D) / Σ V̂² = Σ(D - D̄)² / Σ V̂²
        var_D = np.var(D)
        kappa_star = (n * var_D) / sum_V_sq
        
        # Nuisance MSE (if true values provided)
        if m0_X is not None:
            mse_m = np.mean((m_hat - m0_X) ** 2)
        else:
            mse_m = np.nan
        
        if ell0_X is not None:
            mse_l = np.mean((l_hat - ell0_X) ** 2)
        else:
            mse_l = np.nan
        
        return theta_hat, var_hat, kappa_star, mse_m, mse_l
    
    def fit(
        self,
        Y: NDArray,
        D: NDArray,
        X: NDArray,
        m0_X: Optional[NDArray] = None,
        ell0_X: Optional[NDArray] = None,
    ) -> DMLResult:
        """
        Fit DML estimator with cross-fitting.
        
        Parameters
        ----------
        Y : ndarray of shape (n,)
            Outcome variable.
        D : ndarray of shape (n,)
            Treatment variable.
        X : ndarray of shape (n, p)
            Covariate matrix.
        m0_X : ndarray or None, default None
            True propensity scores (for nuisance MSE calculation).
        ell0_X : ndarray or None, default None
            True reduced-form outcomes (for nuisance MSE calculation).
        
        Returns
        -------
        result : DMLResult
            Estimation results.
        """
        n = len(Y)
        
        if self.n_repeats == 1:
            # Single cross-fitting
            theta_hat, var_hat, kappa_star, mse_m, mse_l = self._single_crossfit(
                Y, D, X, m0_X, ell0_X, self.random_state
            )
            se = np.sqrt(var_hat)
        else:
            # Repeated cross-fitting with median aggregation
            theta_hats = []
            var_hats = []
            kappa_stars = []
            mse_ms = []
            mse_ls = []
            
            for rep in range(self.n_repeats):
                split_seed = self.random_state + rep * 1000
                th, vh, ks, mm, ml = self._single_crossfit(
                    Y, D, X, m0_X, ell0_X, split_seed
                )
                theta_hats.append(th)
                var_hats.append(vh)
                kappa_stars.append(ks)
                mse_ms.append(mm)
                mse_ls.append(ml)
            
            # Median aggregation (Chernozhukov et al. 2018)
            theta_hat = np.nanmedian(theta_hats)
            
            # Variance pooling: median variance + adjustment for splitting noise
            # σ̄² = median(σ̂ₛ²) + median((θ̂ₛ - θ̄)²) / S
            median_var = np.nanmedian(var_hats)
            split_var = np.nanmedian((np.array(theta_hats) - theta_hat) ** 2) / self.n_repeats
            var_hat = median_var + split_var
            se = np.sqrt(var_hat)
            
            kappa_star = np.nanmedian(kappa_stars)
            mse_m = np.nanmedian(mse_ms)
            mse_l = np.nanmedian(mse_ls)
        
        # Confidence interval
        ci_lower = theta_hat - Z_ALPHA * se
        ci_upper = theta_hat + Z_ALPHA * se
        
        # Bias
        bias = theta_hat - self.theta0
        
        # Sample R²(D|X) - compute directly from definition
        # R² = 1 - Var(V̂)/Var(D) = 1 - (1/κ*)
        sample_r2 = 1.0 - 1.0 / kappa_star if kappa_star > 0 and kappa_star < np.inf else 0.0
        
        self.result_ = DMLResult(
            theta_hat=theta_hat,
            se=se,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            kappa_star=kappa_star,
            bias=bias,
            nuisance_mse_m=mse_m,
            nuisance_mse_l=mse_l,
            sample_r2=sample_r2,
            n=n,
        )
        
        return self.result_


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_dml(
    Y: NDArray,
    D: NDArray,
    X: NDArray,
    learner_m: BaseEstimator,
    learner_l: BaseEstimator,
    m0_X: Optional[NDArray] = None,
    ell0_X: Optional[NDArray] = None,
    K: int = K_FOLDS,
    n_repeats: int = 1,
    theta0: float = THETA0,
    random_state: int = 42,
) -> DMLResult:
    """
    Convenience function to run DML estimation.
    
    Parameters
    ----------
    Y : ndarray of shape (n,)
        Outcome.
    D : ndarray of shape (n,)
        Treatment.
    X : ndarray of shape (n, p)
        Covariates.
    learner_m : BaseEstimator
        Learner for m(X).
    learner_l : BaseEstimator
        Learner for ℓ(X).
    m0_X : ndarray or None
        True m₀(X) for MSE calculation.
    ell0_X : ndarray or None
        True ℓ₀(X) for MSE calculation.
    K : int, default 5
        Number of folds.
    n_repeats : int, default 1
        Number of repeated cross-fittings.
    theta0 : float, default 1.0
        True treatment effect.
    random_state : int, default 42
        Random seed.
    
    Returns
    -------
    result : DMLResult
    """
    estimator = DMLEstimator(
        learner_m=learner_m,
        learner_l=learner_l,
        K=K,
        n_repeats=n_repeats,
        theta0=theta0,
        random_state=random_state,
    )
    return estimator.fit(Y, D, X, m0_X, ell0_X)
