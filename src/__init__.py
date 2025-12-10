"""
DML Condition Number Study - Source Modules
============================================

This package contains research code for the DML condition number study.

Modules
-------
- simulation: Monte Carlo simulation DGP and runners
- empirical: LaLonde data loading and DML analysis
- visualization: Plotting and table formatting utilities

For the publishable package, see dml_diagnostic/
"""

from src.simulation import (
    # Constants
    THETA0,
    R2_TARGETS,
    DEFAULT_SEED,
    B_DEFAULT,
    RHO_DEFAULT,
    P_DEFAULT,
    K_FOLDS,
    # DGP functions
    generate_plr_data,
    calibrate_sigma_xi_sq,
    compute_V_gamma,
    # DML functions
    run_dml_plr,
    get_nuisance_model,
    run_single_replication,
    # Simulation engine
    run_simulation,
    # Data classes
    DMLResult,
    DGPInfo,
    ReplicationResult,
    # Summary functions
    compute_cell_summary,
    assign_kappa_regime,
    compute_regime_summary,
    # Table functions
    make_table1,
    make_table2,
    table_to_latex,
    # Visualization
    plot_coverage_vs_kappa,
    plot_ci_length_vs_kappa,
)

from src.empirical import (
    load_lalonde_data,
    get_experimental_sample,
    get_observational_sample,
    get_covariate_matrix,
    PLRDoubleMLKappa,
    run_dml_analysis,
)

from src.visualization import (
    results_to_dataframe,
    results_to_latex,
    plot_propensity_histogram,
    plot_kappa_by_design,
    plot_ci_length_vs_kappa,
)

__all__ = [
    # Constants
    "THETA0",
    "R2_TARGETS",
    "DEFAULT_SEED",
    "B_DEFAULT",
    "RHO_DEFAULT",
    "P_DEFAULT",
    "K_FOLDS",
    # DGP functions
    "generate_plr_data",
    "calibrate_sigma_xi_sq",
    "compute_V_gamma",
    # DML functions
    "run_dml_plr",
    "get_nuisance_model",
    "run_single_replication",
    # Simulation
    "run_simulation",
    # Data classes
    "DMLResult",
    "DGPInfo",
    "ReplicationResult",
    # Summary functions
    "compute_cell_summary",
    "assign_kappa_regime",
    "compute_regime_summary",
    # Table functions
    "make_table1",
    "make_table2",
    "table_to_latex",
    # Visualization (simulation)
    "plot_coverage_vs_kappa",
    "plot_ci_length_vs_kappa",
    # Empirical
    "load_lalonde_data",
    "get_experimental_sample",
    "get_observational_sample",
    "get_covariate_matrix",
    "PLRDoubleMLKappa",
    "run_dml_analysis",
    # Visualization (empirical)
    "results_to_dataframe",
    "results_to_latex",
    "plot_propensity_histogram",
    "plot_kappa_by_design",
]
