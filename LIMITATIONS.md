# Known Limitations

- True ALFRED vintage reconstruction requires a FRED API key. Without it, the model falls back to current-vintage FRED data with explicit publication-lag heuristics.
- The current primary latent-state engines are GMM and Gaussian HMM. Dynamic Factor + HMM, Bayesian HMM, Markov-switching, and change-point models are challenger ideas rather than implemented production components.
- The financial-stress forecast is monthly. A true 1-week stress forecast requires a separate higher-frequency data pipeline.
- Monte Carlo transition simulations hold the most recently fitted HMM transition matrix fixed over the simulated horizon. Time-varying transition matrices are not yet implemented.
- Rare states can have small effective samples, especially in early walk-forward history.
- Bootstrap uncertainty measures the stability of the current fitted read rather than rerunning the complete historical walk-forward pipeline for every bootstrap draw.
- Synthetic fallback data validate estimation machinery, not realistic cross-series economic independence.
- The tactical portfolio is a demonstration application of the classifier, not evidence that the latent economic states are objectively "true".
