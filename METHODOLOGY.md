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
