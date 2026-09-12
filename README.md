# Macro Regime Classifier

A point-in-time macro regime research system that estimates **where the economy is, where it may be moving, how confident the model is, and what those regimes imply for cross-asset behaviour**.

The project uses two independent latent-state axes:

- **Growth × Inflation:** Stable Growth, Overheating, Stagflation, Contraction / Disinflation
- **Volatility × Liquidity:** Calm, Stress, Crisis

`Recession` is deliberately modeled separately as an NBER business-cycle event rather than used as a fourth Growth × Inflation quadrant.

## What makes the project different

The model is built as a research pipeline rather than a hard-coded macro playbook. It combines point-in-time macro data, walk-forward feature engineering, GMM/HMM latent-state estimation, probability-based confirmation, transition forecasting, uncertainty diagnostics, dedicated classifier validation, and empirical cross-asset regime mapping.

### Research architecture

1. Regime Definition & Taxonomy
2. Data Universe
3. Point-in-Time Data Engineering
4. Feature Engineering
5. Latent State Estimation
6. Regime Classification & Confirmation
7. Transition & Forecasting Engine
8. Confidence, Uncertainty & Regime Stability
9. Validation & Economic Reality Check
10. Cross-Asset Regime Mapping

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full methodology.

## Statistical engine

The current primary architecture is:

```text
Point-in-time macro data
        ↓
Block-level features + Level / Direction / Acceleration
        ↓
Walk-forward PCA factors
        ↓
GMM benchmark + causal Gaussian HMM
        ↓
Posterior probabilities
        ↓
Probability-based confirmation
        ↓
Transition forecasts + confidence diagnostics
        ↓
Validation + cross-asset mapping
```

Important implementation choices include:

- **Hungarian one-to-one state assignment** to prevent duplicate GMM economic labels.
- **Causal HMM forward filtering** rather than retrospective future-informed decoding.
- **Purged multi-horizon forecast training** to prevent target leakage.
- **ALFRED-aware YoY reconstruction** so numerator and denominator use vintages known at the same decision date.
- **Probability-first reporting** rather than relying on hard regime labels alone.

## Data

Macro data are sourced from FRED/ALFRED. Asset data are sourced from Yahoo Finance with fallback support through `pandas_datareader`/Stooq.

For full vintage-aware reconstruction, provide a FRED API key:

```bash
export FRED_API_KEY="your_key_here"
```

Without a key, the code falls back to current-vintage FRED series with explicit release-lag assumptions. This limitation is reported in the output rather than hidden.

## Installation

```bash
git clone <repository-url>
cd macro-regime-classifier
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows:

```bash
.venv\Scripts\activate
```

## Run

```bash
python run.py
```

or:

```bash
macro-regime
```

Outputs are written to `outputs/`. The script also builds an interactive Plotly dashboard showing the current regime, posterior probabilities, transition forecasts, uncertainty, recession risk, and cross-asset outlook.

## Validation

The project separates **classifier validation** from **portfolio validation**. Classifier diagnostics include probability calibration, Brier/log-loss metrics, detection lag, regime persistence, bootstrap state stability, model agreement, parameter stability, state-count sensitivity, structural-break analysis, and held-out economic reality checks.

See [`docs/VALIDATION.md`](docs/VALIDATION.md).

## Repository structure

```text
macro-regime-classifier/
├── README.md
├── pyproject.toml
├── requirements.txt
├── run.py
├── docs/
│   ├── METHODOLOGY.md
│   ├── VALIDATION.md
│   └── LIMITATIONS.md
├── examples/
├── outputs/
├── src/
│   └── macro_regime/
│       ├── __init__.py
│       └── engine.py
└── tests/
```

The initial release intentionally keeps the frozen research engine in one package module to minimize refactor risk. A later engineering-only refactor can split data, features, regimes, forecasts, validation and reporting into separate modules without changing model behaviour.

## Disclaimer

This repository is a quantitative research project and is not investment advice. Historical regime relationships and simulated/backtested results do not guarantee future performance.
