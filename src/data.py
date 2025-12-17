"""
LaLonde Data Loader for DML Empirical Application.

This module provides functions to load the LaLonde (1986) / Dehejia-Wahba (1999)
dataset for comparing experimental and observational causal inference.

The LaLonde dataset is a canonical benchmark in causal inference, where the
experimental sample (NSW) provides a "ground truth" against which observational
methods can be validated.

Key Features:
    - Two modes: 'experimental' (NSW only) and 'observational' (NSW treated + PSID control)
    - Standardization of continuous covariates for MLP convergence
    - Returns (y, d, X) tuple compatible with DML estimator

References:
    - LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training Programs.
    - Dehejia, R. & Wahba, S. (1999). Causal Effects in Nonexperimental Studies.

Author: DML Monte Carlo Study
"""

from __future__ import annotations

from typing import Literal, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONSTANTS
# =============================================================================

# URLs for Dehejia-Wahba data (hosted on Rajeev Dehejia's website)
# Using nswre74 files which include the re74 covariate (10 columns)
NSW_TREATED_URL = "https://users.nber.org/~rdehejia/data/nswre74_treated.txt"
NSW_CONTROL_URL = "https://users.nber.org/~rdehejia/data/nswre74_control.txt"
PSID_CONTROL_URL = "https://users.nber.org/~rdehejia/data/psid_controls.txt"

# Column names for the Dehejia-Wahba format
COLUMN_NAMES = [
    'treat', 'age', 'education', 'black', 'hispanic', 
    'married', 'nodegree', 're74', 're75', 're78'
]

# Covariates for the analysis
COVARIATE_COLS = ['age', 'education', 'black', 'hispanic', 'married', 'nodegree', 're74', 're75']
CONTINUOUS_COLS = ['age', 'education', 're74', 're75']
BINARY_COLS = ['black', 'hispanic', 'married', 'nodegree']

# Known experimental benchmark (for reference)
EXPERIMENTAL_BENCHMARK = 1794  # Approximate experimental estimate


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def _load_dehejia_wahba_txt(url: str) -> pd.DataFrame:
    """
    Load a Dehejia-Wahba formatted text file.
    
    Parameters
    ----------
    url : str
        URL to the data file.
    
    Returns
    -------
    df : pd.DataFrame
        Loaded dataframe.
    """
    try:
        df = pd.read_csv(
            url, 
            sep=r'\s+',  # whitespace separated
            header=None, 
            names=COLUMN_NAMES
        )
        return df
    except Exception as e:
        raise RuntimeError(
            f"Failed to load data from {url}. "
            f"Check your internet connection. Error: {e}"
        )


def load_lalonde(
    mode: Literal['experimental', 'observational'] = 'observational',
    standardize: bool = True,
    return_df: bool = False,
) -> Tuple[NDArray, NDArray, NDArray] | Tuple[NDArray, NDArray, NDArray, pd.DataFrame]:
    """
    Load the LaLonde / Dehejia-Wahba dataset.
    
    This function loads the canonical LaLonde dataset in two modes:
    - 'experimental': NSW treated vs. NSW control (randomized experiment)
    - 'observational': NSW treated vs. PSID-1 control (observational comparison)
    
    The experimental sample provides a benchmark treatment effect (~$1794),
    while the observational sample is known to produce biased estimates
    due to selection on unobservables and weak overlap.
    
    Parameters
    ----------
    mode : {'experimental', 'observational'}, default 'observational'
        Which sample to load:
        - 'experimental': NSW treated + NSW control (n ≈ 445)
        - 'observational': NSW treated + PSID-1 control (n ≈ 2675)
    standardize : bool, default True
        Whether to standardize continuous covariates (age, education, re74, re75).
        Recommended for MLP convergence.
    return_df : bool, default False
        If True, also return the full DataFrame.
    
    Returns
    -------
    y : ndarray of shape (n,)
        Outcome variable (re78 - earnings in 1978).
    d : ndarray of shape (n,)
        Treatment indicator (1 = NSW training, 0 = control).
    X : ndarray of shape (n, 8)
        Covariate matrix [age, education, black, hispanic, married, nodegree, re74, re75].
    df : pd.DataFrame (optional)
        Full dataframe if return_df=True.
    
    Examples
    --------
    >>> y, d, X = load_lalonde(mode='experimental')
    >>> print(f"N = {len(y)}, Treated = {d.sum()}")
    N = 445, Treated = 185
    
    >>> y, d, X = load_lalonde(mode='observational')
    >>> print(f"N = {len(y)}, Treated = {d.sum()}")
    N = 2675, Treated = 185
    
    Notes
    -----
    The experimental sample has good overlap (low κ*) because treatment was
    randomized. The observational sample has poor overlap (high κ*) because
    the PSID control group differs systematically from NSW participants.
    """
    mode = mode.lower()
    if mode not in ['experimental', 'observational']:
        raise ValueError(f"mode must be 'experimental' or 'observational', got '{mode}'")
    
    # Load NSW treated (always needed)
    print(f"Loading LaLonde data (mode='{mode}')...")
    df_treated = _load_dehejia_wahba_txt(NSW_TREATED_URL)
    
    # Load appropriate control group
    if mode == 'experimental':
        df_control = _load_dehejia_wahba_txt(NSW_CONTROL_URL)
        sample_name = "Experimental (NSW)"
    else:
        df_control = _load_dehejia_wahba_txt(PSID_CONTROL_URL)
        sample_name = "Observational (NSW-PSID)"
    
    # Combine treated and control
    df = pd.concat([df_treated, df_control], ignore_index=True)
    
    # Extract variables
    y = df['re78'].values.astype(np.float64)
    d = df['treat'].values.astype(np.float64)
    X_raw = df[COVARIATE_COLS].values.astype(np.float64)
    
    # Standardize continuous covariates if requested
    if standardize:
        X = X_raw.copy()
        # Get indices of continuous columns
        cont_indices = [COVARIATE_COLS.index(col) for col in CONTINUOUS_COLS]
        scaler = StandardScaler()
        X[:, cont_indices] = scaler.fit_transform(X[:, cont_indices])
    else:
        X = X_raw
    
    # Print summary
    n_treated = int(d.sum())
    n_control = len(d) - n_treated
    print(f"  Sample: {sample_name}")
    print(f"  N = {len(y):,} (Treated: {n_treated}, Control: {n_control})")
    print(f"  Covariates: {COVARIATE_COLS}")
    print(f"  Standardized: {standardize}")
    
    if return_df:
        return y, d, X, df
    return y, d, X


def load_lalonde_experimental(
    standardize: bool = True,
    return_df: bool = False,
) -> Tuple[NDArray, NDArray, NDArray] | Tuple[NDArray, NDArray, NDArray, pd.DataFrame]:
    """
    Convenience function to load the experimental (NSW) sample.
    
    See load_lalonde for full documentation.
    """
    return load_lalonde(mode='experimental', standardize=standardize, return_df=return_df)


def load_lalonde_observational(
    standardize: bool = True,
    return_df: bool = False,
) -> Tuple[NDArray, NDArray, NDArray] | Tuple[NDArray, NDArray, NDArray, pd.DataFrame]:
    """
    Convenience function to load the observational (NSW-PSID) sample.
    
    See load_lalonde for full documentation.
    """
    return load_lalonde(mode='observational', standardize=standardize, return_df=return_df)


def get_propensity_trimmed_sample(
    y: NDArray,
    d: NDArray,
    X: NDArray,
    alpha: float = 0.1,
    propensity_model: Optional[object] = None,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Trim sample based on propensity score overlap.
    
    Removes observations with extreme propensity scores to improve overlap.
    This is a standard robustness check in observational causal inference.
    
    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome variable.
    d : ndarray of shape (n,)
        Treatment indicator.
    X : ndarray of shape (n, p)
        Covariate matrix.
    alpha : float, default 0.1
        Trimming threshold. Keeps observations where α < e(X) < 1-α.
    propensity_model : sklearn estimator, optional
        Fitted propensity score model. If None, fits a LogisticRegression.
    
    Returns
    -------
    y_trimmed : ndarray
        Trimmed outcome.
    d_trimmed : ndarray
        Trimmed treatment.
    X_trimmed : ndarray
        Trimmed covariates.
    e_hat : ndarray
        Propensity scores for trimmed sample.
    
    Notes
    -----
    As α increases, the sample size decreases but overlap improves (lower κ*).
    This creates a bias-variance tradeoff: better overlap vs. less data.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    
    if alpha < 0 or alpha >= 0.5:
        raise ValueError(f"alpha must be in [0, 0.5), got {alpha}")
    
    # Fit propensity model if not provided
    if propensity_model is None:
        # Use calibrated random forest for better probability estimates
        base_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
        )
        propensity_model = CalibratedClassifierCV(base_clf, cv=5)
        propensity_model.fit(X, d)
    
    # Get propensity scores
    e_hat = propensity_model.predict_proba(X)[:, 1]
    
    # Trim based on threshold
    if alpha > 0:
        mask = (e_hat > alpha) & (e_hat < (1 - alpha))
    else:
        mask = np.ones(len(y), dtype=bool)
    
    return (
        y[mask],
        d[mask],
        X[mask],
        e_hat[mask],
    )


# =============================================================================
# SAMPLE INFO
# =============================================================================

def get_sample_summary(y: NDArray, d: NDArray, X: NDArray) -> dict:
    """
    Get summary statistics for a sample.
    
    Parameters
    ----------
    y : ndarray of shape (n,)
        Outcome variable.
    d : ndarray of shape (n,)
        Treatment indicator.
    X : ndarray of shape (n, p)
        Covariate matrix.
    
    Returns
    -------
    summary : dict
        Dictionary with sample summary statistics.
    """
    n = len(y)
    n_treated = int(d.sum())
    n_control = n - n_treated
    
    # Create boolean masks (handle float treatment indicator)
    treated_mask = d > 0.5
    control_mask = d < 0.5
    
    return {
        'n': n,
        'n_treated': n_treated,
        'n_control': n_control,
        'prop_treated': n_treated / n,
        'mean_outcome': float(y.mean()),
        'mean_outcome_treated': float(y[treated_mask].mean()),
        'mean_outcome_control': float(y[control_mask].mean()),
        'naive_ate': float(y[treated_mask].mean() - y[control_mask].mean()),
        'n_covariates': X.shape[1],
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'load_lalonde',
    'load_lalonde_experimental',
    'load_lalonde_observational',
    'get_propensity_trimmed_sample',
    'get_sample_summary',
    'EXPERIMENTAL_BENCHMARK',
    'COVARIATE_COLS',
]
