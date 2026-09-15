# Validation Framework

Validation is split into four layers.

## Classification quality

- State occupancy and episode counts
- Average and median duration
- Self-transition persistence
- Walk-forward stability
- Short-episode rate
- A→B→A regime-reversion rate
- Bootstrap state stability
- Cross-model agreement

## Forecast quality

- 1M / 3M / 6M Brier score
- Multiclass log loss
- Reliability / probability calibration
- Transition-destination accuracy
- First-passage transition diagnostics
- NBER recession precision, recall and F1 where an observable target exists

## Robustness

- GMM/HMM parameter stability
- PCA loading and explained-variance stability
- State-count sensitivity
- Rolling vs. expanding estimation
- Leave-one-gauge-out sensitivity
- Structural-era breakdowns
- Vintage-data sensitivity
- Bootstrap uncertainty

## Economic reality check

The four-state taxonomy has no directly observable master label. Economic validation therefore tests whether held-out variables behave consistently with the interpretation of inferred states. Recession validation against NBER `USREC` is treated as a separate supervised binary/event problem and does not validate the full four-state taxonomy by itself.


# Changelog

## 0.1.0

- Two-axis economic and financial-stress regime architecture.
- Point-in-time FRED/ALFRED data layer with vintage-aware YoY feature reconstruction.
- Walk-forward PCA feature factors with Level / Direction / Acceleration diagnostics.
- GMM benchmark plus causally filtered HMM primary state estimator.
- Probability-based regime confirmation and transition-watch logic.
- Multi-horizon transition and recession forecasting.
- Confidence, entropy, OOD, bootstrap and stability diagnostics.
- Dedicated classifier validation and economic reality-check suite.
- Cross-asset regime mapping and demonstration tactical allocation.


# Methodology

The system is organized into ten research layers.

## 1. Regime Definition & Taxonomy

The model uses two independent axes rather than one combined state space.

**Economic axis — Growth × Inflation**
- Stable Growth
- Overheating
- Stagflation
- Contraction / Disinflation

**Financial-stress axis — Volatility × Liquidity**
- Calm
- Stress
- Crisis

"Recession" is deliberately separate from the four-state economic taxonomy and is modeled against the NBER `USREC` reference series.

## 2. Data Universe

Raw indicators are grouped into Growth, Labour, Inflation, Housing, Credit, Liquidity, Rates, Financial Conditions, and Market Stress blocks. Indicators are selected for economic relevance, coverage, point-in-time availability, reliability, incremental information, regime differentiation, and structural robustness.

## 3. Point-in-Time Data Engineering

The data layer distinguishes observation date, release/vintage date, and decision date. With a FRED API key, the model retains ALFRED observation × vintage history and reconstructs the latest vintage actually available at each historical decision date. YoY inputs are rebuilt from the contemporaneously available vintage curve before entering feature engineering.

## 4. Feature Engineering

Features are transformed according to economic meaning, directionally oriented, normalized with rolling standard or robust z-scores, and compressed into block-level walk-forward PCA gauges. The feature set also includes Level, Direction, and Acceleration descriptors plus market-stress diagnostics such as cross-asset correlation concentration and realized volatility.

## 5. Latent State Estimation

GMM provides the clustering benchmark while a Gaussian HMM provides the primary temporally persistent state estimate. Economic GMM components are mapped one-to-one to semantic prototypes with the Hungarian assignment algorithm. HMM inference is causal and retains the complete posterior probability vector.

## 6. Regime Classification & Confirmation

Raw HMM probabilities are converted into confirmed regimes using posterior thresholds, probability margins, persistence, transition plausibility, GMM/HMM agreement, asymmetric crisis confirmation, and high-confidence overrides. Majority-vote smoothing remains diagnostic rather than the primary confirmation rule.

## 7. Transition & Forecasting Engine

The engine forecasts future regime distributions using HMM transition matrices and a feature-conditioned multinomial logistic-regression challenger. Forecasts are generated on purged walk-forward training windows. Monte Carlo paths, transition risk, destination probabilities, recession risk, and expected duration diagnostics are also produced.

## 8. Confidence, Uncertainty & Regime Stability

Reported uncertainty includes posterior probabilities, probability margins, normalized entropy, transition entropy, GMM/HMM agreement, Mahalanobis cluster distance, out-of-distribution flags, state age, factor stability, and bootstrap sensitivity.

## 9. Validation & Economic Reality Check

Classifier validation is distinct from allocation validation. It includes walk-forward state stability, persistence, detection lag, short-episode and reversal diagnostics, Brier score, log loss, probability calibration, recession precision/recall/F1, bootstrap state stability, parameter/feature stability, state-count sensitivity, structural-break analysis, and held-out economic reality checks.

## 10. Cross-Asset Regime Mapping

Cross-asset mapping is a downstream application of the classifier. The model estimates regime-conditioned forward returns, volatility, tail risk, correlation, beta, and probability-weighted expected returns across equities, rates, credit, commodities, gold, and USD exposures. The relationships are estimated from data rather than hard-coded.

# GitHub Setup

Create a new empty GitHub repository named `macro-regime-classifier` under your account. Do **not** initialize it with a README, `.gitignore`, or license because those files already exist locally.

Then from the project folder:

```bash
git init -b main
git add .
git commit -m "Initial macro regime classifier release"
git remote add origin https://github.com/ChristianKinnell/macro-regime-classifier.git
git push -u origin main
```

A license has deliberately not been selected yet. Choose one only when you decide how permissively you want others to reuse the code.


# Examples

Run the model from the repository root:

```bash
python run.py
```

or, after installing the package:

```bash
macro-regime
```

The program writes research outputs to `outputs/` and opens an interactive Plotly dashboard locally.

For true ALFRED vintage reconstruction, set `FRED_API_KEY` in your environment before running.


# Known Limitations

- True ALFRED vintage reconstruction requires a FRED API key. Without it, the model falls back to current-vintage FRED data with explicit publication-lag heuristics.
- The current primary latent-state engines are GMM and Gaussian HMM. Dynamic Factor + HMM, Bayesian HMM, Markov-switching, and change-point models are challenger ideas rather than implemented production components.
- The financial-stress forecast is monthly. A true 1-week stress forecast requires a separate higher-frequency data pipeline.
- Monte Carlo transition simulations hold the most recently fitted HMM transition matrix fixed over the simulated horizon. Time-varying transition matrices are not yet implemented.
- Rare states can have small effective samples, especially in early walk-forward history.
- Bootstrap uncertainty measures the stability of the current fitted read rather than rerunning the complete historical walk-forward pipeline for every bootstrap draw.
- Synthetic fallback data validate estimation machinery, not realistic cross-series economic independence.
- The tactical portfolio is a demonstration application of the classifier, not evidence that the latent economic states are objectively "true".


## Documentation
- [Methodology](METHODOLOGY.md)
- [Validation](VALIDATION.md)
- [Limitations](LIMITATIONS.md)
- [GitHub Setup](GITHUB_SETUP.md)
- [Changelog](CHANGELOG.md)
