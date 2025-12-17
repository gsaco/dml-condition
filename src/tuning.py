"""
Hyperparameter Tuning for DML Monte Carlo Study.

This module provides pre-tuning functionality for Random Forest learners.
The idea is to tune hyperparameters ONCE per (N, R²) regime, then use
these fixed parameters in all Monte Carlo replications.

This saves significant compute compared to running RandomizedSearchCV
inside every replication.

Author: DML Monte Carlo Study
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.dgp import generate_nonlinear_data
from src.learners import RF_PARAM_GRID


def tune_rf_hyperparameters(
    n_samples: int,
    target_r2: float,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Pre-tune Random Forest hyperparameters for a given (N, R²) regime.
    
    Generates a validation dataset with the specified parameters and runs
    RandomizedSearchCV to find optimal hyperparameters for both the
    propensity score (m) and outcome (ℓ) models.
    
    Parameters
    ----------
    n_samples : int
        Sample size for the tuning dataset.
    target_r2 : float
        Target R²(D|X) for the DGP.
    random_state : int, default 42
        Random seed for reproducibility.
    n_iter : int, default 10
        Number of parameter settings sampled in RandomizedSearchCV.
    cv : int, default 3
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs.
    
    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters found.
        Keys may include: 'max_depth', 'min_samples_leaf', 'max_features'.
    
    Notes
    -----
    The tuning is done on the propensity score target (D) since this is
    typically the harder prediction task in DML with weak overlap.
    """
    # Generate validation data
    Y, D, X, info, dgp = generate_nonlinear_data(
        n=n_samples,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    # Create base RF
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,  # RandomizedSearchCV handles parallelism
    )
    
    # Run RandomizedSearchCV on propensity score (m) target
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    # Fit on propensity score prediction task
    search.fit(X, D)
    
    return search.best_params_


def tune_rf_for_data(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Tune Random Forest hyperparameters for a specific dataset.
    
    This is used for the LaLonde application where we tune directly
    on the real data rather than simulated data.
    
    Parameters
    ----------
    X : ndarray of shape (n, p)
        Covariate matrix.
    y : ndarray of shape (n,)
        Target variable (treatment or outcome).
    random_state : int, default 42
        Random seed for reproducibility.
    n_iter : int, default 10
        Number of parameter settings sampled.
    cv : int, default 3
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs.
    
    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters.
    """
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
    )
    
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    search.fit(X, y)
    
    return search.best_params_


def tune_rf_hyperparameters_highdim(
    n_samples: int,
    target_r2: float,
    p: int = 100,
    s: int = 5,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Pre-tune Random Forest hyperparameters for high-dimensional setting.
    
    Parameters
    ----------
    n_samples : int
        Sample size for the tuning dataset.
    target_r2 : float
        Target R²(D|X) for the DGP.
    p : int, default 100
        Covariate dimension.
    s : int, default 5
        Sparsity (number of active covariates).
    random_state : int, default 42
        Random seed for reproducibility.
    n_iter : int, default 10
        Number of parameter settings sampled.
    cv : int, default 3
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs.
    
    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters found.
    """
    from src.dgp import generate_highdim_data
    
    # Generate validation data
    Y, D, X, info, dgp = generate_highdim_data(
        n=n_samples,
        target_r2=target_r2,
        p=p,
        s=s,
        random_state=random_state,
    )
    
    # Create base RF
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
    )
    
    # Run RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    search.fit(X, D)
    
    return search.best_params_


def tune_rf_for_binary_treatment(
    n_samples: int,
    target_overlap: float,
    random_state: int = 42,
    n_iter: int = 10,
    cv: int = 3,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Pre-tune Random Forest hyperparameters for binary treatment DGP.
    
    Parameters
    ----------
    n_samples : int
        Sample size for the tuning dataset.
    target_overlap : float
        Target propensity score overlap.
    random_state : int, default 42
        Random seed for reproducibility.
    n_iter : int, default 10
        Number of parameter settings sampled.
    cv : int, default 3
        Number of cross-validation folds.
    n_jobs : int, default -1
        Number of parallel jobs.
    
    Returns
    -------
    best_params : dict
        Dictionary containing the best hyperparameters found.
    """
    from src.dgp import generate_binary_treatment_data
    
    # Generate validation data
    Y, D, X, info, dgp = generate_binary_treatment_data(
        n=n_samples,
        target_overlap=target_overlap,
        random_state=random_state,
    )
    
    # Create base RF
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
    )
    
    # Run RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=base_rf,
        param_distributions=RF_PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_squared_error',
        random_state=random_state,
        n_jobs=n_jobs,
    )
    
    search.fit(X, D)
    
    return search.best_params_


__all__ = [
    'tune_rf_hyperparameters',
    'tune_rf_for_data',
    'tune_rf_hyperparameters_highdim',
    'tune_rf_for_binary_treatment',
]
