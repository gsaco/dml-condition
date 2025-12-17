"""
Hyperparameter Tuning for DML Monte Carlo Study.

This module provides pre-tuning functionality for Random Forest learners.
The idea is to tune hyperparameters ONCE per (N, R²) regime, then use
these fixed parameters in all Monte Carlo replications.

Author: Gabriel Saco
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from src.dgp import generate_data
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
    RandomizedSearchCV to find optimal hyperparameters for the propensity
    score model.
    
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
    """
    # Generate validation data
    Y, D, X, info, dgp = generate_data(
        n=n_samples,
        target_r2=target_r2,
        random_state=random_state,
    )
    
    # Create base RF
    base_rf = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=1,
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


__all__ = [
    'tune_rf_hyperparameters',
    'tune_rf_for_data',
]
