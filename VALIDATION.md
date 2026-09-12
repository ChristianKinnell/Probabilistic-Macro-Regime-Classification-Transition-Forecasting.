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
