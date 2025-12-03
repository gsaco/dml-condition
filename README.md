# Finite-Sample Conditioning in Double Machine Learning: A Short Note

This repository contains the code and manuscript for the short communication:

> **"Finite-Sample Conditioning in Double Machine Learning: A Short Note"**

## Overview

Double Machine Learning (DML) provides asymptotically valid inference for low-dimensional parameters in high-dimensional settings. However, finite-sample performance can deteriorate when the empirical Jacobian is nearly singular—a situation arising from poor overlap or strong collinearity between treatment and covariates.

This short communication:
1. Introduces a simple, interpretable **condition number** $\kappa_{\mathrm{DML}} := 1/|\hat{J}_\theta|$ for the Partially Linear Regression (PLR) model
2. Establishes a **coverage error bound** showing that miscoverage scales as $\kappa_{\mathrm{DML}}/\sqrt{n} + \kappa_{\mathrm{DML}}\sqrt{n} \cdot r_n$
3. Characterizes three **conditioning regimes** (well-conditioned, moderately ill-conditioned, severely ill-conditioned)
4. Provides **Monte Carlo simulations** demonstrating how $\kappa_{\mathrm{DML}}$ predicts coverage failures
5. Offers **practical diagnostic recommendations** for applied researchers

## Repository Structure

```
dml_paper/
├── paper/
│   └── main.tex              # LaTeX source for the short communication
├── code/
│   └── simulations.ipynb     # Jupyter notebook with Monte Carlo simulations
├── results/
│   ├── results_summary.csv   # Simulation results summary
│   ├── coverage_vs_kappa.png # Coverage vs kappa plot
│   ├── rmse_vs_kappa.png     # RMSE vs kappa plot
│   └── theta_distributions.png
├── archive/                  # Old/legacy files from full paper version
│   ├── main_full_version.tex
│   ├── bias_aware_dml_simulations.ipynb
│   └── kappa_theory_validation.ipynb
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Reproducing the Results

### 1. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Simulations

Open and run the Jupyter notebook:

```bash
cd code
jupyter notebook simulations.ipynb
```

The notebook will:
- Generate PLR data with varying overlap levels and covariate correlation
- Fit DML estimators with 5-fold cross-fitting using Random Forests
- Compute the condition number $\kappa_{\mathrm{DML}}$ for each design
- Produce the summary table and figures used in the paper

Results are saved to `results/`.

### 4. Compile the Paper

```bash
cd paper
pdflatex main.tex
# Run twice for references:
pdflatex main.tex
```

## Key Results

The Monte Carlo simulations reveal clear stratification by conditioning:

| Regime | $\kappa_{\mathrm{DML}}$ | Coverage |
|--------|-------------------------|----------|
| Well-conditioned | 0.7–0.9 | 89%–93% |
| Moderately ill-conditioned | 1.7–1.8 | 68%–91% |
| Severely ill-conditioned | 4.8–5.6 | **8.8%**–38.6% |

The striking result: at $n=2000$ with low overlap and $\rho=0.9$, nominal 95% CIs cover the true parameter only **8.8%** of the time.

## Practical Recommendations

We recommend computing and reporting $\kappa_{\mathrm{DML}}$ alongside DML estimates:

- **$\kappa_{\mathrm{DML}} < 1$**: Standard inference is reliable
- **$1 \le \kappa_{\mathrm{DML}} < 3$**: Exercise caution; consider robustness checks
- **$\kappa_{\mathrm{DML}} \ge 3$**: Standard CIs are unreliable

## Citation

If you use this code or methodology, please cite:

```bibtex
@article{dml_conditioning_2024,
  title={Finite-Sample Conditioning in Double Machine Learning: A Short Note},
  author={Anonymous},
  journal={TBD},
  year={2024}
}
```

## License

MIT License
