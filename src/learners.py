"""
Learner Factory for DML Monte Carlo Study.

This module provides a factory function for creating nuisance learners,
including OLS (for misspecification control), regularized methods (Lasso),
tree-based methods (RF, XGB), and neural networks (MLP).

Key Features:
    - Default and tuned variants of RF and XGB
    - Oracle learner for theoretical lower bound
    - All learners are scikit-learn compatible

Author: DML Monte Carlo Study
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, TYPE_CHECKING, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


if TYPE_CHECKING:
    from src.dgp import DGP


# =============================================================================
# CONSTANTS
# =============================================================================

LEARNER_NAMES = Literal[
    "OLS", "Lasso", "Ridge", "RF", "RF_Tuned", "GBM", "MLP", "Oracle", "CorruptedOracle"
]

RF_PARAM_GRID: Dict[str, Any] = {
    'max_depth': [None, 10, 20],
    'min_samples_leaf': [1, 5, 10],
    'max_features': ['sqrt', None],
}




# =============================================================================
# ORACLE LEARNER
# =============================================================================

class OracleLearner(BaseEstimator, RegressorMixin):
    """
    Oracle learner that returns true nuisance function values.
    
    This learner accepts a DGP object and, when fit() is called,
    simply stores the true m₀(X) or ℓ₀(X) values. When predict() is called,
    it returns the stored true values.
    
    This establishes the theoretical lower bound of error. Any error in
    ML models beyond this Oracle error is attributable to the "Bias
    Amplification" mechanism.
    
    Parameters
    ----------
    dgp : DGP
        The data generating process object.
    target : str, default 'm'
        Which nuisance function to return: 'm' for m₀(X), 'l' for ℓ₀(X).
    
    Attributes
    ----------
    true_values_ : ndarray
        The true nuisance function values from training.
    X_train_ : ndarray
        The training covariates (for matching during predict).
    """
    
    def __init__(
        self,
        dgp: Optional["DGP"] = None,
        target: Literal['m', 'l'] = 'm'
    ) -> None:
        self.dgp = dgp
        self.target = target
        self.true_values_: Optional[NDArray] = None
        self.X_train_: Optional[NDArray] = None
    
    def fit(
        self,
        X: NDArray,
        y: NDArray,
        **kwargs
    ) -> "OracleLearner":
        """
        Store the true nuisance function values.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        y : ndarray of shape (n,)
            Target values (ignored, we use true values from DGP).
        
        Returns
        -------
        self
        """
        if self.dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        
        self.X_train_ = X.copy()
        
        if self.target == 'm':
            self.true_values_ = self.dgp.m0(X)
        else:
            self.true_values_ = self.dgp.ell0(X)
        
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Return the true nuisance function values.
        
        For Oracle to work in cross-fitting context, we compute
        the true values for the new X directly.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        predictions : ndarray of shape (n,)
            True nuisance function values.
        """
        if self.dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        
        if self.target == 'm':
            return self.dgp.m0(X)
        else:
            return self.dgp.ell0(X)


# =============================================================================
# CORRUPTED ORACLE LEARNER (For Bias Amplification Analysis)
# =============================================================================

class CorruptedOracle(BaseEstimator, RegressorMixin):
    """
    Corrupted Oracle that returns true nuisance values with multiplicative bias.
    
    This learner isolates the pure effect of κ* on bias amplification by
    injecting a known, multiplicative bias (1 + δ) into perfect nuisance estimates.
    
    From our theory:
        θ̂ - θ₀ ≈ κ* × B_n + S_n + R_n
    
    Where B_n is the nuisance bias. By setting predictions = truth × (1 + δ),
    we can verify that |θ̂ - θ₀| ∝ κ* × δ, proving κ* is a pure multiplier.
    
    Note on Bias Injection and DML Centering
    -----------------------------------------
    This implementation uses **multiplicative bias** (1 + δ) instead of additive
    bias. Multiplicative bias survives DML's centering of residuals because it
    scales the entire function rather than shifting it by a constant that could
    be absorbed into the intercept.
    
    Parameters
    ----------
    true_function_callback : callable
        A function that takes X and returns the true nuisance values.
        E.g., dgp.m0 or dgp.ell0.
    bias : float, default 0.01
        Multiplicative bias factor. Predictions = truth × (1 + bias).
    
    Examples
    --------
    >>> corrupted_m = CorruptedOracle(dgp.m0, bias=0.05)
    >>> corrupted_m.fit(X, D)  # Does nothing
    >>> predictions = corrupted_m.predict(X)  # Returns m0(X) × 1.05
    """

    
    def __init__(
        self,
        true_function_callback: Optional[callable] = None,
        bias: float = 0.01,
    ) -> None:
        self.true_function_callback = true_function_callback
        self.bias = bias
    
    def fit(self, X: NDArray, y: NDArray, **kwargs) -> "CorruptedOracle":
        """
        Fit method (no-op for CorruptedOracle).
        
        The CorruptedOracle doesn't learn anything - it just returns
        the true values plus bias.
        """
        # No fitting needed - we use the true function callback
        return self
    
    def predict(self, X: NDArray) -> NDArray:
        """
        Return true nuisance values with multiplicative bias.
        
        Uses multiplicative bias (1 + bias) instead of additive bias
        to ensure the bias survives DML's centering of residuals.
        
        Parameters
        ----------
        X : ndarray of shape (n, p)
            Covariate matrix.
        
        Returns
        -------
        predictions : ndarray of shape (n,)
            True nuisance values × (1 + bias).
        """
        if self.true_function_callback is None:
            raise ValueError("true_function_callback must be provided")
        
        truth = self.true_function_callback(X)
        # Use multiplicative bias to survive DML centering
        return truth * (1.0 + self.bias)


def get_corrupted_oracle_pair(
    dgp: "DGP",
    bias_m: float = 0.01,
    bias_l: float = 0.01,
) -> tuple[CorruptedOracle, CorruptedOracle]:
    """
    Get a pair of CorruptedOracle learners for both nuisance functions.
    
    Parameters
    ----------
    dgp : DGP
        The data generating process.
    bias_m : float, default 0.01
        Bias to add to m₀(X).
    bias_l : float, default 0.01
        Bias to add to ℓ₀(X).
    
    Returns
    -------
    corrupted_m : CorruptedOracle
        Corrupted oracle for m₀(X) = E[D|X].
    corrupted_l : CorruptedOracle
        Corrupted oracle for ℓ₀(X) = E[Y|X].
    """
    return (
        CorruptedOracle(true_function_callback=dgp.m0, bias=bias_m),
        CorruptedOracle(true_function_callback=dgp.ell0, bias=bias_l),
    )


# =============================================================================
# LEARNER FACTORY
# =============================================================================

def get_learner(
    name: LEARNER_NAMES,
    tuned: bool = False,
    dgp: Optional["DGP"] = None,
    random_state: int = 42,
    n_jobs: int = -1,
    params: Optional[Dict[str, Any]] = None,
) -> BaseEstimator:
    """
    Factory function to create nuisance regression models.
    
    Parameters
    ----------
    name : str
        Learner name: 'OLS', 'Lasso', 'RF', 'RF_Tuned', 'MLP', 'Oracle'.
    tuned : bool, default False
        For backward compatibility. If True and name is 'RF',
        will return the tuned variant.
    dgp : DGP or None, default None
        Required for Oracle learner.
    random_state : int, default 42
        Random state for reproducibility.
    n_jobs : int, default -1
        Number of parallel jobs.
    params : dict or None, default None
        Pre-tuned hyperparameters for RF_Tuned. If provided, returns a
        RandomForestRegressor with these params. If None, returns
        RandomizedSearchCV for pre-tuning phase.
    
    Returns
    -------
    model : sklearn estimator
        Configured regression model.
    
    Raises
    ------
    ValueError
        If unknown learner name or if XGBoost is not installed.
    """
    name_upper = name.upper()
    
    # Handle backward compatibility with tuned flag
    if tuned:
        if 'RF' in name_upper and 'TUNED' not in name_upper:
            name_upper = 'RF_TUNED'
        elif 'XGB' in name_upper and 'TUNED' not in name_upper:
            name_upper = 'XGB_TUNED'
    
    if name_upper == 'OLS':
        return LinearRegression()
    
    elif name_upper == 'LASSO':
        return LassoCV(
            cv=5,
            fit_intercept=True,
            max_iter=10000,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    
    elif name_upper == 'RIDGE':
        # Ridge regression with built-in cross-validation
        return RidgeCV(
            cv=5,
            fit_intercept=True,
        )
    
    elif name_upper == 'RF':
        # Fast RF with sensible fixed hyperparameters (no tuning overhead)
        # Good defaults from empirical studies: max_depth=15, min_samples_leaf=5
        return RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=random_state,
            n_jobs=n_jobs,
        )
    
    elif name_upper == 'RF_TUNED':
        # If params provided, use fixed hyperparameters (post pre-tuning)
        if params is not None:
            return RandomForestRegressor(
                n_estimators=200,
                random_state=random_state,
                n_jobs=n_jobs,
                **params,
            )
        # Otherwise, return search object for pre-tuning phase
        base_rf = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,  # RandomizedSearchCV handles parallelism
        )
        return RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=RF_PARAM_GRID,
            n_iter=10,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=random_state,
            n_jobs=n_jobs,
        )
    

    elif name_upper == 'GBM':
        # Gradient Boosting with sklearn's HistGradientBoostingRegressor
        # Faster than traditional GradientBoostingRegressor, handles missing values
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(
            max_iter=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=random_state,
        )
    
    elif name_upper == 'MLP':
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=500,
                random_state=random_state,
            )
        )
    
    elif name_upper == 'ORACLE':
        if dgp is None:
            raise ValueError("DGP must be provided for Oracle learner")
        return OracleLearner(dgp=dgp, target='m')
    
    elif name_upper == 'CORRUPTEDORACLE' or name_upper == 'CORRUPTED_ORACLE':
        if dgp is None:
            raise ValueError("DGP must be provided for CorruptedOracle learner")
        # Default bias; use get_corrupted_oracle_pair for custom bias
        return CorruptedOracle(true_function_callback=dgp.m0, bias=0.01)
    
    else:
        raise ValueError(
            f"Unknown learner: '{name}'. "
            f"Choose from: OLS, Lasso, RF_Tuned, MLP, Oracle, CorruptedOracle"
        )


def get_oracle_pair(
    dgp: "DGP"
) -> tuple[OracleLearner, OracleLearner]:
    """
    Get a pair of Oracle learners for both nuisance functions.
    
    Parameters
    ----------
    dgp : DGP
        The data generating process.
    
    Returns
    -------
    oracle_m : OracleLearner
        Oracle for m₀(X) = E[D|X].
    oracle_l : OracleLearner
        Oracle for ℓ₀(X) = E[Y|X].
    """
    return OracleLearner(dgp=dgp, target='m'), OracleLearner(dgp=dgp, target='l')


# =============================================================================
# STRUCTURAL KAPPA COMPUTATION
# =============================================================================

def compute_structural_kappa(
    D: NDArray,
    m0_X: NDArray,
) -> float:
    """
    Compute the structural condition number κ* from true nuisance values.
    
    This is used in Corrupted Oracle analysis to get the "true" κ* before
    bias is injected. The structural κ* should be constant across all
    bias levels for a given R² regime.
    
    From math.tex:
        κ* = σ_D² / σ_V² = 1 / (1 - R²(D|X))
    
    Parameters
    ----------
    D : ndarray of shape (n,)
        Treatment variable.
    m0_X : ndarray of shape (n,)
        True propensity scores m₀(X) = E[D|X].
    
    Returns
    -------
    structural_kappa : float
        The structural condition number.
    
    Examples
    --------
    >>> from src.dgp import generate_data
    >>> Y, D, X, info, dgp = generate_data(n=1000, target_r2=0.90)
    >>> structural_kappa = compute_structural_kappa(D, info['m0_X'])
    >>> print(f"κ* ≈ {structural_kappa:.1f}")  # Should be ~10 for R²=0.90
    κ* ≈ 10.0
    """
    n = len(D)
    V_true = D - m0_X  # True treatment residual
    var_D = np.var(D)
    sum_V_sq = np.sum(V_true ** 2)
    
    if sum_V_sq < 1e-12:
        return np.inf
    
    return (n * var_D) / sum_V_sq


# =============================================================================
# AVAILABLE LEARNERS
# =============================================================================

AVAILABLE_LEARNERS = ['OLS', 'Lasso', 'Ridge', 'RF', 'RF_Tuned', 'GBM', 'MLP']

# For LaLonde application: comprehensive learner comparison
LALONDE_LEARNERS = ['OLS', 'Lasso', 'Ridge', 'RF_Tuned', 'GBM', 'MLP']

# For simulation study: RF (fast) instead of RF_Tuned for JCI-standard replications
SIMULATION_LEARNERS = ['OLS', 'Lasso', 'RF', 'MLP']

__all__ = [
    'OracleLearner',
    'CorruptedOracle',
    'get_learner',
    'get_oracle_pair',
    'get_corrupted_oracle_pair',
    'compute_structural_kappa',
    'AVAILABLE_LEARNERS',
    'SIMULATION_LEARNERS',
    'LEARNER_NAMES',
]
