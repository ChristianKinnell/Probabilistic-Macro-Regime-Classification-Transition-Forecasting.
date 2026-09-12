"""
================================================================================
A hedge-fund-style macro regime classification system that sorts markets into
regimes along TWO INDEPENDENT AXES rather than a single taxonomy:

    Axis 1 — GROWTH x INFLATION quadrant   (Stable Growth / Overheating /
             Stagflation / Contraction-Disinflation), estimated with a Gaussian Mixture
             Model and cross-checked with a Gaussian Hidden Markov Model,
             both fit walk-forward (expanding window, periodic refit) so no
             regime label is ever informed by data that postdates it.

    Axis 2 — VOLATILITY x LIQUIDITY stress state (Calm / Stress / Crisis),
             estimated the same way but from an entirely separate feature
             set (VIX, credit spreads, TED-spread proxy, Fed balance sheet,
             M2 growth). This is deliberately NOT folded into the Growth x
             Inflation quadrant: liquidity/vol crunches (e.g. Sep-2019 repo
             spike, Mar-2020 dash-for-cash) can occur inside an otherwise
             benign growth/inflation quadrant, and collapsing the two axes
             into one classifier would hide exactly that kind of event.

"Recession" is deliberately NOT a quadrant label. A growth/inflation quadrant
is an unsupervised sign-of-the-gauge classification and a recession is an
NBER business-cycle EVENT -- collapsing the two invites exactly the taxonomy
error where a stagflationary downturn (weak growth, still-elevated inflation)
gets excluded from "recession" just because it doesn't land in the low-growth
low-inflation quadrant. So the fourth quadrant is named Contraction /
Disinflation (weak/contracting growth, moderating/falling inflation), and
"Recession" is reserved exclusively for the supervised NBER USREC validator
below, which is a separate model over a separate target.

Both axes feed:
    - A Random Forest supervised classifier validated against NBER USREC
      dates (USREC itself excluded from the feature set to avoid circular
      leakage), reported with full per-class precision/recall/F1.
    - A walk-forward, regime-conditioned Sharpe-weighted tactical asset
      allocation across a cross-asset ETF universe, with a volatility-
      targeting overlay and a liquidity-regime leverage throttle.
    - A regime-conditioned risk model: VaR/CVaR lookback windows and
      leverage caps that shorten/tighten automatically as the
      Volatility/Liquidity axis moves from Calm -> Stress -> Crisis.

Validation ("defensibility suite"):
    - Moving-block bootstrap Sharpe ratio confidence intervals
    - Paired block-bootstrap significance test vs. a buy-and-hold benchmark
    - Regime-label permutation test (is the regime label informative, or
      would a random relabeling produce the same tactical-allocation edge?)
    - Sub-period rank stability (Spearman) of regime-conditioned asset
      rankings across three equal sub-periods
    - Transaction-cost sensitivity sweep
    - Walk-forward hit-rate vs. the ex-post oracle top-N selection
    - Jennrich (1970) chi-square test for whether regime-conditioned
      correlation matrices are statistically distinguishable
    - Newey-West HAC t-stats on regime-conditioned mean returns, since
      rolling z-score / momentum features are serially correlated and a
      naive OLS t-stat would overstate significance

DATA:
    Macro data:  pandas_datareader's FRED client (a maintained API wrapper,
                 no manual CSV parsing, no API key required), with FRED's
                 direct fredgraph endpoint as a secondary retry if that
                 client itself fails.
    Asset data:  yfinance (Yahoo Finance's de facto standard Python API
                 client), with a stooq-via-pandas_datareader retry as a
                 secondary public source if Yahoo is unreachable.
    Both are wrapped in a synthetic-data fallback used ONLY if every real
    data path fails (e.g. sandboxed execution with no outbound network
    access at all): a regime-aware synthetic generator, so the full
    pipeline still runs end-to-end and is testable offline. The synthetic
    path also embeds a KNOWN ground-truth regime path, so the classifier's
    recovery accuracy against ground truth is printed as a sanity check —
    a cheap validation step that isn't available when using real data
    (nobody knows the "true" regime in real life). No local CSV file is
    ever read as a primary data source; the only CSV files this script
    writes are its own OUTPUT files (regime_panel.csv etc.) for your
    downstream use.
================================================================================
"""

import warnings 
warnings.filterwarnings("ignore", category=FutureWarning)
from dataclasses import dataclass, field

import os
import tempfile
import warnings
import webbrowser
from dataclasses import dataclass, field
from io import StringIO
from typing import Optional
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, f1_score, roc_auc_score,
                              average_precision_score, precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM
from scipy import stats
from scipy.stats import spearmanr, multivariate_normal
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment

import statsmodels.api as sm
import logging

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

RNG_SEED = 42
np.random.seed(RNG_SEED)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SECTION 1: REGIME DEFINITION
# CONFIGURATION
@dataclass
class Config:
    start_date: str = "2007-06-01"
    end_date: Optional[str] = None 
    use_live_data: bool = True
    random_state: int = RNG_SEED
    fred_api_key: Optional[str] = None

    # DATA UNIVERSE:
    growth_series: dict = field(default_factory=lambda: {
        "INDPRO":   ("Industrial Production Index", False, True, "Coincident"),
        "W875RX1":  ("Real Personal Income ex. Transfers", False, True, "Coincident"),
        "RRSFS":    ("Real Retail & Food Services Sales", False, True, "Coincident"),
    })
    labour_series: dict = field(default_factory=lambda: {
        "PAYEMS": ("Nonfarm Payrolls", False, True, "Coincident"),
        "UNRATE": ("Unemployment Rate", True, False, "Lagging"),
        "ICSA":   ("Initial Jobless Claims", True, False, "Leading"),
        "CCSA":   ("Continuing Jobless Claims", True, False, "Leading/Coincident"),
        "JTSJOL": ("JOLTS Job Openings", False, True, "Leading"),
    })
    inflation_series: dict = field(default_factory=lambda: {
        "CPIAUCSL": ("CPI, All Urban Consumers (headline)", False, True, "Lagging/Coincident"),
        "CPILFESL": ("Core CPI (ex. food & energy)", False, True, "Lagging/Coincident"),
        "PPIACO":   ("Producer Price Index", False, True, "Leading"),
        "PCEPI":    ("PCE Price Index (headline)", False, True, "Lagging/Coincident"),
        "PCEPILFE": ("Core PCE (Fed's preferred gauge)", False, True, "Lagging/Coincident"),
        "T5YIE":   ("5Y Breakeven Inflation Rate", False, False, "Leading/Real-time"),
        "T5YIFR":  ("5Y5Y Forward Inflation Expectation", False, False, "Leading/Real-time"),
    })
    housing_series: dict = field(default_factory=lambda: {
        "PERMIT":       ("Building Permits", False, True, "Leading"),
        "HOUST":        ("Housing Starts", False, True, "Leading"),
        "HSN1F":        ("New Home Sales", False, True, "Leading"),
        "MORTGAGE30US": ("30Y Mortgage Rate", True, False, "Leading/Real-time"),
    })
    credit_series: dict = field(default_factory=lambda: {
        "BAA10Y": ("BAA-10Y Credit Spread", False, False, "Leading/Real-time") 
    })
    liquidity_series: dict = field(default_factory=lambda: {
        "M2SL":     ("M2 Money Supply", True, True, "Leading"),
        "TEDRATE":  ("TED Spread proxy (CP-Tbill, post-LIBOR)", False, False, "Real-time"),
        "WALCL":    ("Fed Balance Sheet", True, True, "Real-time"),
        "WRESBAL":  ("Bank Reserves at the Fed", True, True, "Real-time"),
    })
    rates_series: dict = field(default_factory=lambda: {
        "FEDFUNDS": ("Effective Fed Funds Rate", False, False, "Coincident"),
        "DGS2":     ("2Y Treasury Yield", False, False, "Leading"),
        "DGS10":    ("10Y Treasury Yield", False, False, "Leading"),
        "T10Y2Y":   ("10Y-2Y Treasury Spread", False, False, "Leading"),
        "T10Y3M":   ("10Y-3M Treasury Spread", False, False, "Leading"),
        "DFII10":   ("10Y TIPS Real Yield", False, False, "Leading"),
    })
    financial_conditions_series: dict = field(default_factory=lambda: {
        "NFCI": ("Chicago Fed National Financial Conditions Index", False, False, "Real-time"),
    })
    volatility_series: dict = field(default_factory=lambda: {
        "VIXCLS": ("CBOE VIX", True, False, "Real-time"),
    })

    recession_series: str = "USREC"

    # Asset universe
    asset_universe: tuple = (
        "SPY", "IWM", "EFA", "EEM", "QQQ",   # equities
        "TLT", "IEF", "SHY", "LQD", "HYG",   # rates & credit
        "GLD", "DBC", "USO",                 # commodities
        "UUP",                                # dollar
    )
    benchmark_ticker: str = "SPY"

    # Regime taxonomy
    quadrant_labels: dict = field(default_factory=lambda: {
        (1, 1):   "Overheating",     # High growth, high inflation
        (1, -1):  "Stable Growth",   # High growth, low inflation
        (-1, 1):  "Stagflation",     # Low growth, high inflation
        (-1, -1): "Contraction / Disinflation",  # Low growth, low inflation
    })
    vol_liq_order: tuple = ("Calm", "Stress", "Crisis")
    quadrant_prototype_dmax: float = 1.5
    min_state_occupancy_months: int = 12

    confirm_economic_p_enter: float = 0.60     
    confirm_economic_p_exit: float = 0.40      
    confirm_economic_margin_min: float = 0.10  
    confirm_economic_n_confirm: int = 3        
    confirm_economic_score_min: float = 0.02
    confirm_economic_override_p: float = 0.85  

    confirm_stress_p_enter: float = 0.55    
    confirm_stress_p_exit: float = 0.35
    confirm_stress_margin_min: float = 0.05
    confirm_stress_n_confirm: int = 2
    confirm_stress_score_min: float = 0.05
    confirm_stress_override_p: float = 0.80
    confirm_crisis_entry_n_confirm: int = 1
    confirm_crisis_exit_n_confirm: int = 2

    # TRANSITION & FORECASTING ENGINE:
    forecast_horizons_economic: tuple = (1, 3, 6)
    forecast_horizons_stress: tuple = (1, 3)
    forecast_mc_paths: int = 10000
    forecast_mc_horizon: int = 12         
    forecast_logreg_C: float = 1.0        

    # CONFIDENCE, UNCERTAINTY & REGIME STABILITY:
    confidence_entropy_bands: tuple = (0.25, 0.50, 0.75)

    # Model-REFIT bootstrap draws 
    confidence_n_bootstrap: int = 150
    confidence_bootstrap_hmm_n_iter: int = 50 
    confidence_ood_percentile: float = 97.5   
    confidence_cluster_fit_bands: tuple = (0.50, 0.80, 0.95)  
    confidence_factor_evr_ratio_warn: float = 0.85  

    # VALIDATION & ECONOMIC REALITY CHECK:
    validation_false_transition_min_months: int = 2  
    validation_recession_detection_threshold: float = 0.30  
    validation_detection_lag_window: int = 12          
    validation_state_stability_n_boot: int = 80        
    validation_rolling_window_months: int = 60         
    validation_structural_break_dates: tuple = (
        "2009-06-01", "2012-01-01", "2017-01-01", "2020-01-01", "2020-07-01", "2022-01-01",
    )
    validation_calibration_bins: tuple = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    # CROSS-ASSET REGIME MAPPING 
    crossasset_min_obs: int = 10       
    crossasset_min_joint_obs: int = 6   
    crossasset_significance_maxlags: int = 6 

    z_window: int = 36               
    smoothing_window: int = 3        
    
    min_regime_history: int = 36     
    refit_every: int = 12            

    # Audit finding 
    hmm_n_iter: int = 500
    hmm_n_restarts: int = 3
   
    hmm_forgetting_floor: float = 0.02
    zscore_method: str = "standard"
    zscore_clip: Optional[float] = 5.0
    double_standardize_gauges: bool = True
    direction_window: int = 3

    vol_target: float = 0.10
    vol_scale_cap: float = 1.5
    vol_scale_floor: float = 0.3
    top_n_assets: int = 6
    txn_cost_bps: float = 10.0
    min_same_regime_months: int = 24 

    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    lookback_calm: int = 36
    lookback_stress: int = 12
    lookback_crisis: int = 6
    leverage_calm: float = 1.5
    leverage_stress: float = 1.0
    leverage_crisis: float = 0.5

    n_bootstrap: int = 2000
    n_permutation: int = 2000
    block_size_months: int = 12

    # SECTION 2: DATA UNIVERSE
    def indicator_blocks(self) -> dict:
        return {
            "growth": self.growth_series,
            "labour": self.labour_series,
            "inflation": self.inflation_series,
            "housing": self.housing_series,
            "credit": self.credit_series,
            "liquidity": self.liquidity_series,
            "rates": self.rates_series,
            "financial_conditions": self.financial_conditions_series,
            "volatility": self.volatility_series,
        }


# FEATURE ADMISSION FRAMEWORK:
FEATURE_ADMISSION_CRITERIA = (
    "Economic relevance — a plausible economic mechanism links it to the regime "
    "(Indicator -> Economic mechanism -> Regime information), not just backtest fit.",
    "Historical coverage — enough observations across several business cycles.",
    "Release frequency — preferably monthly, weekly or daily.",
    "Point-in-time availability — historical release timing/vintages can be "
    "reconstructed (current-vintage-only is a known limitation until this is met).",
    "Data reliability — reputable, reproducible source (FRED/ALFRED, exchange data).",
    "Incremental information — not a near-duplicate of another admitted series "
    "(the model should have ~9 blocks of real signal, not 9 blocks each hiding "
    "several redundant copies of the same underlying factor).",
    "Regime differentiation — behaviour differs meaningfully across economic states.",
    "Low structural fragility — definition/methodology hasn't changed so much "
    "that the historical series is no longer comparable across the sample "
    "(e.g. why TEDRATE needed a post-LIBOR CP-Tbill proxy).",
)


# FEATURE TRANSFORM POLICY:
TRANSFORM_POLICY_TABLE = (
    ("Price/index & real-activity series (CPI, PCE, PPI, Industrial "
     "Production, M2, Fed balance sheet, Payrolls, Housing Starts/Permits/"
     "Sales, JOLTS, Retail Sales, bank reserves)",
     "use_yoy=True -- growth rate (YoY)"),
    ("Rates (Fed Funds, Treasury yields, TIPS real yield, mortgage rate)",
     "use_yoy=False -- level (a rate isn't percentage-changed against itself)"),
    ("Spreads (BAA-10Y, TED, T10Y2Y, T10Y3M)",
     "use_yoy=False -- level"),
    ("Ratios/% and index variables (Unemployment rate, breakevens, NFCI, VIX)",
     "use_yoy=False -- level"),
    ("Flow/count series without a natural growth-rate reading "
     "(Initial/Continuing Claims)",
     "use_yoy=False -- level (already a flow; z-scoring the level is enough)"),
)

# ORIENTATION CONVENTION
BLOCK_ORIENTATION = {
    "growth":    "Higher = stronger growth",
    "labour":    "Higher = stronger labour market (UNRATE, ICSA, CCSA flipped)",
    "inflation": "Higher = more inflation / higher inflation expectations",
    "housing":   "Higher = stronger housing activity (MORTGAGE30US flipped)",
    "credit":    "Higher = MORE credit stress (wider spread) -- matches "
                 "STRESS_AXIS_GAUGES' convention below",
    "liquidity": "Higher = MORE liquidity stress (tighter/contracting) -- "
                 "matches STRESS_AXIS_GAUGES' convention below",
    "rates":     "No fixed convention yet -- not wired into an axis "
                 "(see Config.indicator_blocks docstring)",
    "financial_conditions": "Higher NFCI = tighter/more stressful conditions "
                             "(confirmation-only, not a PCA/axis input)",
    "volatility": "Higher = more stress (vix_z/equity_vol_z computed "
                  "directly from raw VIX/realized-vol, already this "
                  "orientation with no flip needed)",
}
# STRESS_AXIS_GAUGES:
CFG = Config()

# SECTION 3: POINT-IN-TIME DATA ENGINEERING

# DATA LAYER
def fetch_fred_series(series_id: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        import pandas_datareader.data as pdr
        s = pdr.DataReader(series_id, "fred", start, end)[series_id]
        s.index.name = "date"
        return s
    except Exception as e1:
        warnings.warn(f"[FRED] pandas_datareader fetch failed for {series_id}: {e1}. "
                       f"Retrying via the FRED CSV endpoint directly ...")
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df.columns = ["date", series_id]
        df["date"] = pd.to_datetime(df["date"])
        df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
        s = df.set_index("date")[series_id]
        return s.loc[start:end]
    except Exception as e2:
        warnings.warn(f"[FRED] direct CSV fetch also failed for {series_id}: {e2}")
        return None


def fetch_ted_spread_proxy(start, end) -> Optional[pd.Series]:
    cp = fetch_fred_series("DCPF3M", start, end)
    tbill = fetch_fred_series("DTB3", start, end)
    if cp is None or tbill is None:
        return None
    spread = (cp - tbill).dropna()
    spread.index.name = "date"
    return spread


# POINT-IN-TIME DATA ENGINEERING:
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.parquet")


FRED_API_BASE = "https://api.stlouisfed.org/fred/series/observations"

# Fallback-only:
PUBLICATION_LAG_DAYS: dict = {
    "INDPRO": 16, "W875RX1": 30, "RRSFS": 15,
    "PAYEMS": 5, "UNRATE": 5, "ICSA": 5, "CCSA": 12, "JTSJOL": 35,
    "CPIAUCSL": 13, "CPILFESL": 13, "PPIACO": 16, "PCEPI": 30, "PCEPILFE": 30,
    "T5YIE": 0, "T5YIFR": 0,
    "PERMIT": 18, "HOUST": 18, "HSN1F": 25, "MORTGAGE30US": 0,
    "BAA10Y": 0,
    "M2SL": 21, "TEDRATE": 0, "DCPF3M": 0, "DTB3": 0, "WALCL": 3, "WRESBAL": 3,
    "FEDFUNDS": 0, "DGS2": 0, "DGS10": 0, "T10Y2Y": 0, "T10Y3M": 0, "DFII10": 0,
    "NFCI": 5,
    "VIXCLS": 0,
    "USREC": 365,
}


def apply_publication_lag(raw: pd.Series, series_id: str) -> pd.Series:
    lag = PUBLICATION_LAG_DAYS.get(series_id, 0)
    s = raw.dropna().copy()
    s.index = s.index + pd.Timedelta(days=lag)
    return s


def fetch_fred_vintage_series(series_id: str, api_key: str, start: str,
                               use_cache: bool = True) -> pd.DataFrame:

    key = f"fredvintage_full_{series_id}_{start}"
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        return pd.read_parquet(path)

    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": start,
        "realtime_start": "1776-07-04", "realtime_end": "9999-12-31",
        "output_type": 1,
    }
    resp = requests.get(FRED_API_BASE, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if "error_message" in payload:
        raise RuntimeError(f"FRED API error for {series_id}: {payload['error_message']}")
    data = payload.get("observations", [])
    if not data:
        raise RuntimeError(f"No vintage observations returned for {series_id}")

    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        raise RuntimeError(f"All vintage observations for {series_id} were non-numeric/missing")

    full = df[["date", "realtime_start", "value"]].sort_values(["date", "realtime_start"])

    first_release = full.loc[full.groupby("date")["realtime_start"].idxmin()]
    n_dates = first_release["date"].nunique()
    n_avail = first_release["realtime_start"].nunique()
    max_batch_frac = first_release["realtime_start"].value_counts().iloc[0] / n_dates
    if n_avail < 0.5 * n_dates or max_batch_frac > 0.10:
        raise RuntimeError(
            f"{series_id}: ALFRED vintage history only covers {n_avail} distinct "
            f"release dates for {n_dates} observations (largest single batch = "
            f"{max_batch_frac:.0%}) -- likely starts too late for this model's "
            f"start_date; falling back to the release-lag heuristic")

    full = full.set_index("date").sort_index()
    full.to_parquet(path)
    return full


_LAST_VINTAGE_PANEL: Optional[pd.DataFrame] = None
_VINTAGE_SOURCE: dict = {}
_RAW_VINTAGE_TABLES: dict = {}


def _collapse_to_nowcast_series(vintage: pd.DataFrame) -> pd.Series:

    v = vintage.reset_index()[["date", "realtime_start", "value"]]
    v = v.sort_values(["realtime_start", "date"])
    latest_obs_date = pd.Timestamp.min
    idx, vals = [], []
    for date, rt, val in v.itertuples(index=False):
        if date >= latest_obs_date:
            latest_obs_date = date
            idx.append(rt)
            vals.append(val)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx))
    s = s[~s.index.duplicated(keep="last")]
    return s.sort_index()


def reconstruct_asof_history(series_id: str, asof) -> pd.Series:

    vintage = _RAW_VINTAGE_TABLES.get(series_id)
    if vintage is None:
        raise KeyError(f"{series_id}: no raw vintage table cached (call "
                        f"load_fred_series_point_in_time first; the lag-heuristic "
                        f"fallback path has no vintage table to reconstruct from)")
    v = vintage.reset_index()[["date", "realtime_start", "value"]]
    avail = v[v["realtime_start"] <= pd.Timestamp(asof)]
    if avail.empty:
        return pd.Series(dtype=float)
    idx = avail.groupby("date")["realtime_start"].idxmax()
    return avail.loc[idx].set_index("date")["value"].sort_index()


def pit_transform_value(series_id: str, asof, use_yoy: bool) -> float:
    curve = reconstruct_asof_history(series_id, asof)
    if curve.empty:
        return np.nan
    vals = curve.pct_change(12) if use_yoy else curve
    if vals.empty or pd.isna(vals.iloc[-1]):
        return np.nan
    return float(vals.iloc[-1])


def build_pit_feature_column(series_id: str, decision_dates: pd.DatetimeIndex,
                              use_yoy: bool) -> pd.Series:
    vals = {t: pit_transform_value(series_id, t, use_yoy) for t in decision_dates}
    return pd.Series(vals, index=decision_dates)

def load_fred_series_point_in_time(series_ids: list, start: str, end: str,
                                    api_key: Optional[str] = None,
                                    use_cache: bool = True) -> Optional[pd.DataFrame]:

    api_key = api_key or os.environ.get("FRED_API_KEY")
    cols = {}
    fallback_ids = []

    if api_key is None:
        warnings.warn(
            "https://fred.stlouisfed.org/docs/api/api_key.html for true ALFRED vintages.")
        fallback_ids = list(series_ids)
    else:
        for sid in series_ids:
            try:
                vintage = fetch_fred_vintage_series(sid, api_key, start=start, use_cache=use_cache)
                _RAW_VINTAGE_TABLES[sid] = vintage
                s = _collapse_to_nowcast_series(vintage)
                cols[sid] = s
                _VINTAGE_SOURCE[sid] = "alfred_vintage"
            except Exception as e:
                warnings.warn(f"[point-in-time] {sid}: {e} -- falling back to "
                               f"current-vintage + release-lag heuristic")
                fallback_ids.append(sid)

    for sid in fallback_ids:
        raw = fetch_fred_series(sid, start, end)
        if raw is None:
            continue
        cols[sid] = apply_publication_lag(raw, sid)
        _VINTAGE_SOURCE[sid] = f"lag_heuristic({PUBLICATION_LAG_DAYS.get(sid, 0)}d)"

    if not cols:
        return None
    panel = pd.DataFrame(cols).sort_index()
    global _LAST_VINTAGE_PANEL
    _LAST_VINTAGE_PANEL = panel.copy()
    return panel.loc[:end] if end else panel


def audit_snapshot(decision_date: str, block_map: Optional[dict] = None) -> pd.DataFrame:
    if _LAST_VINTAGE_PANEL is None:
        raise RuntimeError("audit_snapshot: no point-in-time panel in memory yet -- "
                            "call load_all_data() first.")
    decision_date = pd.Timestamp(decision_date)
    block_of = {}
    if block_map:
        for block, series in block_map.items():
            for sid in series:
                block_of[sid] = block

    rows = []
    for sid in _LAST_VINTAGE_PANEL.columns:
        s = _LAST_VINTAGE_PANEL[sid].loc[:decision_date].dropna()
        panel_value = float(s.iloc[-1]) if not s.empty else np.nan
        panel_date = s.index[-1].date() if not s.empty else pd.NaT
        asof_latest_obs, asof_value, n_revised_12m = pd.NaT, np.nan, np.nan
        if sid in _RAW_VINTAGE_TABLES:
            curve = reconstruct_asof_history(sid, decision_date)
            if not curve.empty:
                asof_latest_obs = curve.index[-1].date()
                asof_value = round(float(curve.iloc[-1]), 4)
                window_start = decision_date - pd.DateOffset(months=12)
                prior_curve = reconstruct_asof_history(sid, window_start)
                common = curve.index.intersection(prior_curve.index)
                common = common[common <= window_start]
                if len(common):
                    n_revised_12m = int((curve.loc[common] != prior_curve.loc[common]).sum())

        rows.append({
            "series_id": sid, "block": block_of.get(sid, ""),
            "available_date": panel_date,
            "value": round(panel_value, 4) if pd.notna(panel_value) else np.nan,
            "asof_latest_observation": asof_latest_obs, "asof_value": asof_value,
            "n_history_revised_asof_12m_ago": n_revised_12m,
            "source": _VINTAGE_SOURCE.get(sid, ""),
        })
    return pd.DataFrame(rows).set_index("series_id").sort_values("block")


def fetch_yahoo_prices(tickers, start, end) -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        data = yf.download(list(tickers), start=start, end=end,
                            auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        if data.empty:
            raise ValueError("empty frame returned")
        return data
    except Exception as e:
        warnings.warn(f"[Yahoo] live fetch failed: {e}. Retrying via stooq (pandas_datareader) ...")
    try:
        import pandas_datareader.data as pdr
        frames = {}
        for tk in tickers:
            try:
                d = pdr.DataReader(tk, "stooq", start, end)["Close"].sort_index()
                frames[tk] = d
            except Exception:
                continue
        if not frames:
            raise ValueError("stooq fallback returned no series")
        return pd.DataFrame(frames)
    except Exception as e2:
        warnings.warn(f"[Stooq] fallback also failed: {e2}")
        return None

def synthesize_regime_path(n_months: int, seed: int = RNG_SEED):
    rng = np.random.default_rng(seed)
    states = ["Stable Growth", "Overheating", "Stagflation", "Contraction / Disinflation"]
    P = np.array([
        [0.95, 0.02, 0.01, 0.02],
        [0.06, 0.87, 0.05, 0.02],
        [0.04, 0.05, 0.85, 0.06],
        [0.03, 0.01, 0.05, 0.91],
    ])
    path = [0]
    for _ in range(n_months - 1):
        path.append(rng.choice(4, p=P[path[-1]]))
    return [states[i] for i in path], states, P


def synthesize_vol_liq_path(regime_path, seed: int = RNG_SEED + 1):
    rng = np.random.default_rng(seed)
    states = list(CFG.vol_liq_order)
    base_p = {"Stable Growth": [0.85, 0.13, 0.02],
              "Overheating":   [0.70, 0.25, 0.05],
              "Stagflation":   [0.45, 0.40, 0.15],
              "Contraction / Disinflation": [0.25, 0.40, 0.35]}
    path = []
    prev = 0
    for r in regime_path:
        p = np.array(base_p[r])
        # small persistence nudge
        p = p * 0.7 + np.eye(3)[prev] * 0.3
        p = p / p.sum()
        prev = rng.choice(3, p=p)
        path.append(states[prev])
    return path


def synthesize_macro_and_asset_data(cfg: Config):
    dates = pd.date_range(cfg.start_date, periods=1, freq="MS")
    end = pd.Timestamp(cfg.end_date) if cfg.end_date else pd.Timestamp.today()
    dates = pd.date_range(cfg.start_date, end, freq="MS")
    n = len(dates)

    regime_path, regime_states, _ = synthesize_regime_path(n)
    vol_liq_path = synthesize_vol_liq_path(regime_path)
    truth_df = pd.DataFrame({"true_quadrant": regime_path,
                              "true_vol_liq": vol_liq_path}, index=dates)

    growth_level = {"Stable Growth": 1.0, "Overheating": 1.6,
                     "Stagflation": -0.8, "Contraction / Disinflation": -1.8}
    infl_level = {"Stable Growth": -0.3, "Overheating": 1.4,
                  "Stagflation": 1.8, "Contraction / Disinflation": -0.9}
    vol_level = {"Calm": -0.6, "Stress": 0.8, "Crisis": 2.5}
    liq_level = {"Calm": 0.5, "Stress": -0.5, "Crisis": -2.0}

    rng = np.random.default_rng(RNG_SEED + 2)
    g_lat = np.array([growth_level[r] for r in regime_path]) + \
        pd.Series(rng.normal(0, 0.35, n)).rolling(3, min_periods=1).mean().values
    i_lat = np.array([infl_level[r] for r in regime_path]) + \
        pd.Series(rng.normal(0, 0.35, n)).rolling(3, min_periods=1).mean().values
    v_lat = np.array([vol_level[s] for s in vol_liq_path]) + rng.normal(0, 0.3, n)
    l_lat = np.array([liq_level[s] for s in vol_liq_path]) + \
        pd.Series(rng.normal(0, 0.3, n)).rolling(2, min_periods=1).mean().values

    macro = pd.DataFrame(index=dates)
    macro_specs = {
        "INDPRO": (g_lat, 1.0, 100, 0.4),
        "PAYEMS": (g_lat, 0.8, 130000, 300),
        "UNRATE": (-g_lat, 0.9, 5.5, 0.4),
        "ICSA":   (-g_lat, 0.7, 300000, 25000),
        "PERMIT": (g_lat, 1.1, 1300, 120),
        "T10Y2Y": (g_lat, 0.5, 0.8, 0.5),
        "CPIAUCSL": (i_lat, 1.0, 220, 3.0),
        "PPIACO":   (i_lat, 1.1, 200, 4.0),
        "PCEPI":    (i_lat, 0.9, 110, 1.5),
        "M2SL":    (l_lat, 0.8, 15000, 400),
        "BAA10Y":  (-l_lat, 0.6, 2.2, 0.4),
        "TEDRATE": (-l_lat, 0.5, 0.4, 0.2),
        "WALCL":   (l_lat, 1.0, 4500000, 300000),
        "VIXCLS":  (v_lat, 1.4, 17, 4.0),
        "USREC":   (None, None, None, None),
    }
    for sid, (lat, beta, base, scale) in macro_specs.items():
        if sid == "USREC":
            continue
        noise = rng.normal(0, 1, n)
        level = base + scale * (beta * lat + 0.5 * noise)
        drift = np.cumsum(rng.normal(0, scale * 0.02, n))
        macro[sid] = level + drift
    
    macro["USREC"] = (pd.Series(regime_path, index=dates) == "Contraction / Disinflation").astype(int)

    # Asset prices, regime-conditioned drift/vol, correlated shocks
    regime_asset_mu = {
        "Stable Growth": {"eq": 0.13, "govt": 0.02, "credit": 0.05, "gold": 0.02, "cmdty": 0.03, "usd": 0.00},
        "Overheating":   {"eq": 0.10, "govt": -0.03, "credit": 0.03, "gold": 0.04, "cmdty": 0.10, "usd": -0.01},
        "Stagflation":   {"eq": -0.05, "govt": -0.05, "credit": -0.03, "gold": 0.12, "cmdty": 0.08, "usd": 0.02},
        "Contraction / Disinflation": {"eq": -0.15, "govt": 0.09, "credit": -0.06, "gold": 0.06, "cmdty": -0.10, "usd": 0.03},
    }
    vol_liq_vol_mult = {"Calm": 1.0, "Stress": 1.6, "Crisis": 2.6}
    sleeve_map = {
        "SPY": "eq", "IWM": "eq", "EFA": "eq", "EEM": "eq", "QQQ": "eq",
        "TLT": "govt", "IEF": "govt", "SHY": "govt",
        "LQD": "credit", "HYG": "credit",
        "GLD": "gold", "DBC": "cmdty", "USO": "cmdty",
        "UUP": "usd",
    }
    base_vol = {"eq": 0.16, "govt": 0.07, "credit": 0.09, "gold": 0.14, "cmdty": 0.20, "usd": 0.07}
    rng2 = np.random.default_rng(RNG_SEED + 3)
    market_shock = rng2.normal(0, 1, n)
    prices = pd.DataFrame(index=dates)
    for tk in cfg.asset_universe:
        sleeve = sleeve_map.get(tk, "eq")
        mu = np.array([regime_asset_mu[r][sleeve] for r in regime_path]) / 12
        vmult = np.array([vol_liq_vol_mult[s] for s in vol_liq_path])
        sigma_m = base_vol[sleeve] / np.sqrt(12) * vmult
        idio = rng2.normal(0, 1, n)
        beta_mkt = 0.5 if sleeve in ("eq", "credit") else (-0.2 if sleeve == "govt" else 0.1)
        ret = mu + sigma_m * (beta_mkt * market_shock + np.sqrt(max(1e-6, 1 - beta_mkt ** 2)) * idio)
        prices[tk] = 100 * np.cumprod(1 + ret)

    return macro, prices, truth_df


def load_all_data(cfg: Config):
    end = cfg.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    all_series = {}
    live_ok = cfg.use_live_data

    if live_ok:
        series_ids = [sid for block in cfg.indicator_blocks().values() for sid in block] \
            + [cfg.recession_series]
        fetch_ids = [sid for sid in series_ids if sid != "TEDRATE"]
        needs_ted = "TEDRATE" in series_ids
        if needs_ted:
            fetch_ids += ["DCPF3M", "DTB3"]

        raw_panel = load_fred_series_point_in_time(fetch_ids, cfg.start_date, end,
                                                     api_key=cfg.fred_api_key)
        if raw_panel is None:
            live_ok = False
        else:
            if needs_ted and "DCPF3M" in raw_panel.columns and "DTB3" in raw_panel.columns:
                raw_panel["TEDRATE"] = (raw_panel["DCPF3M"] - raw_panel["DTB3"]).dropna()
            raw_panel = raw_panel.drop(columns=["DCPF3M", "DTB3"], errors="ignore")
            missing = [sid for sid in series_ids if sid not in raw_panel.columns]
            if missing:
                warnings.warn(f"[point-in-time] missing series after fetch, "
                               f"falling back to synthetic data: {missing}")
                live_ok = False
            else:
                all_series = raw_panel

    if live_ok:
        macro = pd.DataFrame(all_series)
        macro = macro.resample("ME").last().loc[cfg.start_date:end]
        prices = fetch_yahoo_prices(cfg.asset_universe, cfg.start_date, end)
        if prices is None:
            live_ok = False
        else:
            prices = prices.resample("ME").last()
        truth_df = None

    if not live_ok:
        macro, prices, truth_df = synthesize_macro_and_asset_data(cfg)

    macro = macro.ffill().dropna(how="all")
    prices = prices.ffill().dropna(how="all")
    common_idx = macro.index.intersection(prices.index)
    macro = macro.loc[common_idx]
    prices = prices.loc[common_idx]
    if truth_df is not None:
        truth_df = truth_df.loc[common_idx]
    return macro, prices, truth_df

# SECTION 4: FEATURE ENGINEERING
def rolling_zscore(s: pd.Series, window: int, clip: Optional[float] = None) -> pd.Series:
    mu = s.rolling(window, min_periods=max(6, window // 2)).mean()
    sd = s.rolling(window, min_periods=max(6, window // 2)).std()
    z = (s - mu) / sd.replace(0, np.nan)
    return z.clip(-clip, clip) if clip is not None else z

def rolling_robust_zscore(s: pd.Series, window: int, clip: Optional[float] = None) -> pd.Series:

    min_p = max(6, window // 2)
    med = s.rolling(window, min_periods=min_p).median()
    mad = s.rolling(window, min_periods=min_p).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    return z.clip(-clip, clip) if clip is not None else z

def compute_zscore(s: pd.Series, window: int, cfg: Config) -> pd.Series:
    fn = rolling_robust_zscore if cfg.zscore_method == "robust" else rolling_zscore
    return fn(s, window, clip=cfg.zscore_clip)

def momentum_slope(s: pd.Series, window: int) -> pd.Series:
    def _slope(x):
        if np.isnan(x).any():
            return np.nan
        t = np.arange(len(x))
        return np.polyfit(t, x, 1)[0]
    return s.rolling(window).apply(_slope, raw=True)

def pca_composite_gauge_walkforward(macro: pd.DataFrame, series_map: dict,
                                     z_window: int, cfg: Config):
    raw_panel = pd.DataFrame(index=macro.index)
    for sid, (_, flip, use_yoy, _timing) in series_map.items():
        if sid not in macro.columns:
            continue
        if use_yoy and sid in _RAW_VINTAGE_TABLES:
            raw = build_pit_feature_column(sid, macro.index, use_yoy=True)
        else:
            raw = macro[sid].pct_change(12) if use_yoy else macro[sid]
        raw = -raw if flip else raw
        raw_panel[sid] = raw
    raw_panel = raw_panel.dropna()

    if raw_panel.shape[1] < 2:
        gauge = raw_panel.iloc[:, 0] if raw_panel.shape[1] == 1 else pd.Series(index=macro.index, dtype=float)
        return gauge.reindex(macro.index), 1.0, {}, np.nan, []

    z_panel = raw_panel.apply(lambda s: compute_zscore(s, z_window, cfg)).dropna()
    n = len(z_panel)

    if n < cfg.min_regime_history:
        gauge = z_panel.mean(axis=1)
        return gauge.reindex(macro.index), 1.0, {}, np.nan, []

    idx = z_panel.index
    start = cfg.min_regime_history
    checkpoints = list(range(start, n, cfg.refit_every)) + [n]

    gauge = pd.Series(index=idx, dtype=float)
    last_evr, last_loadings = 1.0, {}
    loading_history = [] 
    evr_history = []

    for i in range(len(checkpoints) - 1):
        train_end = checkpoints[i]
        pred_start, pred_end = checkpoints[i], checkpoints[i + 1]

        train_vals = z_panel.iloc[:train_end].values
        pca = PCA(n_components=1, random_state=cfg.random_state)
        train_pc1 = pca.fit_transform(train_vals).ravel()
        loadings = dict(zip(z_panel.columns, pca.components_[0]))

        train_mean = z_panel.iloc[:train_end].mean(axis=1).values
        flip_sign = np.corrcoef(train_pc1, train_mean)[0, 1] < 0

        pred_vals = z_panel.iloc[pred_start:pred_end].values
        if len(pred_vals) == 0:
            continue
        pc1_pred = pca.transform(pred_vals).ravel()
        if flip_sign:
            pc1_pred = -pc1_pred
            loadings = {k: -v for k, v in loadings.items()}

        gauge.iloc[pred_start:pred_end] = pc1_pred
        last_evr = pca.explained_variance_ratio_[0]
        last_loadings = loadings
        loading_history.append(np.array([loadings[c] for c in z_panel.columns]))
        evr_history.append(float(last_evr))

    if len(loading_history) >= 2:
        sims = [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
                for a, b in zip(loading_history[:-1], loading_history[1:])]
        stability = float(np.mean(sims))
    else:
        stability = np.nan

    return gauge.reindex(macro.index), last_evr, last_loadings, stability, evr_history

def cross_asset_stress_features(prices: pd.DataFrame, window: int = 6):
    rets = prices.pct_change()
    avg_corr, eig_share = [], []
    idx = rets.index
    for i in range(len(idx)):
        if i < window:
            avg_corr.append(np.nan)
            eig_share.append(np.nan)
            continue
        window_rets = rets.iloc[i - window + 1:i + 1].dropna(axis=1, how="any")
        if window_rets.shape[1] < 3:
            avg_corr.append(np.nan)
            eig_share.append(np.nan)
            continue
        c = window_rets.corr().values
        n = c.shape[0]
        off_diag = (c.sum() - n) / (n * (n - 1))
        eigvals = np.linalg.eigvalsh(c)
        avg_corr.append(off_diag)
        eig_share.append(eigvals.max() / eigvals.sum())
    return (pd.Series(avg_corr, index=idx, name="avg_pairwise_corr"),
            pd.Series(eig_share, index=idx, name="top_eig_share"))

def equity_realized_vol(prices: pd.DataFrame, cfg: Config, window: int = 6):

    bench = prices[cfg.benchmark_ticker] if cfg.benchmark_ticker in prices else prices.iloc[:, 0]
    return (bench.pct_change().rolling(window).std() * np.sqrt(12)).rename("equity_realized_vol")

def build_feature_panel(macro: pd.DataFrame, prices: pd.DataFrame, cfg: Config):

    gauge_info = {}
    gauges = {}
    for block_name, series_map in cfg.indicator_blocks().items():
        if block_name == "volatility" or not series_map:
            continue
        gauge, evr, loadings, stability, evr_history = pca_composite_gauge_walkforward(
            macro, series_map, cfg.z_window, cfg)
        gauges[block_name] = gauge
        gauge_info[f"{block_name}_explained_var"] = evr
        gauge_info[f"{block_name}_loadings"] = loadings
        gauge_info[f"{block_name}_loading_stability"] = stability
        gauge_info[f"{block_name}_evr_history"] = evr_history

    vix_z = compute_zscore(macro["VIXCLS"], cfg.z_window, cfg) if "VIXCLS" in macro else pd.Series(index=macro.index)
    avg_corr, eig_share = cross_asset_stress_features(prices)
    equity_vol_z = compute_zscore(equity_realized_vol(prices, cfg), cfg.z_window, cfg)

    feature_cols = {"vix_z": vix_z, "equity_vol_z": equity_vol_z,
                     "avg_pairwise_corr": avg_corr, "top_eig_share": eig_share}
    for block_name, gauge in gauges.items():
        feature_cols[f"{block_name}_gauge"] = gauge
        feature_cols[f"{block_name}_z"] = (
            compute_zscore(gauge, cfg.z_window, cfg) if cfg.double_standardize_gauges else gauge)

    level_cols = ["vix_z", "equity_vol_z", "avg_pairwise_corr", "top_eig_share"] + \
        [f"{b}_z" for b in gauges.keys()]
    for col in level_cols:
        if col not in feature_cols:
            continue
        base = col[:-2] if col.endswith("_z") else col
        direction = momentum_slope(feature_cols[col], cfg.direction_window)
        feature_cols[f"{base}_D"] = direction
        feature_cols[f"{base}_A"] = direction.diff(1)

    features = pd.DataFrame(feature_cols)

    rf_features = pd.DataFrame(index=macro.index)
    for sid in macro.columns:
        if sid == cfg.recession_series:
            continue
        rf_features[f"{sid}_z"] = compute_zscore(macro[sid], cfg.z_window, cfg)
        rf_features[f"{sid}_mom3"] = momentum_slope(macro[sid], 3)

    return features, rf_features, gauge_info

# SECTION 4 (continued): FEATURE ENGINEERING SENSITIVITY DIAGNOSTICS
def zscore_window_sensitivity(macro: pd.DataFrame, prices: pd.DataFrame, cfg: Config,
                               windows=(12, 36, 60),
                               cols=("growth_z", "inflation_z", "liquidity_z", "credit_z")) -> pd.DataFrame:
    panels = {}
    for w in windows:
        c = Config(**{**cfg.__dict__, "z_window": w})
        feats, _, _ = build_feature_panel(macro, prices, c)
        panels[w] = feats
    rows = []
    for col in cols:
        if col not in panels[windows[0]]:
            continue
        df = pd.DataFrame({w: panels[w][col] for w in windows}).dropna()
        for i, a in enumerate(windows):
            for b in windows[i + 1:]:
                rows.append({
                    "feature": col, "window_a": a, "window_b": b,
                    "correlation": df[a].corr(df[b]),
                    "sign_agreement": (np.sign(df[a]) == np.sign(df[b])).mean(),
                })
    return pd.DataFrame(rows)

def zscore_method_sensitivity(macro: pd.DataFrame, prices: pd.DataFrame, cfg: Config,
                               cols=("growth_z", "inflation_z", "vix_z", "liquidity_z")) -> pd.DataFrame:

    cfg_std = Config(**{**cfg.__dict__, "zscore_method": "standard"})
    cfg_rob = Config(**{**cfg.__dict__, "zscore_method": "robust"})
    feats_std, _, _ = build_feature_panel(macro, prices, cfg_std)
    feats_rob, _, _ = build_feature_panel(macro, prices, cfg_rob)
    rows = []
    for col in cols:
        if col not in feats_std or col not in feats_rob:
            continue
        a, b = feats_std[col].dropna(), feats_rob[col].dropna()
        common = a.index.intersection(b.index)
        rows.append({
            "feature": col, "correlation": a.loc[common].corr(b.loc[common]),
            "max_abs_standard": a.abs().max(), "max_abs_robust": b.abs().max(),
        })
    return pd.DataFrame(rows)

def double_standardization_sensitivity(macro: pd.DataFrame, prices: pd.DataFrame, cfg: Config,
                                        cols=("growth_z", "inflation_z", "liquidity_z", "credit_z")) -> pd.DataFrame:
    cfg_double = Config(**{**cfg.__dict__, "double_standardize_gauges": True})
    cfg_single = Config(**{**cfg.__dict__, "double_standardize_gauges": False})
    feats_double, _, _ = build_feature_panel(macro, prices, cfg_double)
    feats_single, _, _ = build_feature_panel(macro, prices, cfg_single)
    rows = []
    for col in cols:
        if col not in feats_double or col not in feats_single:
            continue
        d, s = feats_double[col].dropna(), feats_single[col].dropna()
        common = d.index.intersection(s.index)
        rows.append({
            "feature": col, "correlation": d.loc[common].corr(s.loc[common]),
            "ar1_double": d.autocorr(1), "ar1_single": s.autocorr(1),
        })
    return pd.DataFrame(rows)

# SECTION 5: LATENT STATE ESTIMATION
# REGIME CLASSIFICATION ENGINE
QUADRANT_PROTOTYPES = {
    "Stable Growth":              (1.0, -1.0),
    "Overheating":                (1.0, 1.0),
    "Stagflation":                (-1.0, 1.0),
    "Contraction / Disinflation": (-1.0, -1.0),
}

def assign_states_to_prototypes(means: np.ndarray, prototypes: dict):
    labels = list(prototypes.keys())
    proto_arr = np.array([prototypes[l] for l in labels])
    n_states, n_protos = means.shape[0], len(labels)
    size = max(n_states, n_protos)
    cost = np.full((size, size), 1e6)
    for k in range(n_states):
        for j in range(n_protos):
            cost[k, j] = float(np.linalg.norm(means[k, :2] - proto_arr[j]))
    row_ind, col_ind = linear_sum_assignment(cost)
    label_map, distances = {}, {}
    for k, j in zip(row_ind, col_ind):
        if k >= n_states or j >= n_protos:
            continue
        label_map[int(k)] = labels[j]
        distances[labels[j]] = cost[k, j]
    return label_map, distances


def relabel_quadrant_states(means: np.ndarray, cfg: Config):
    label_map, distances = assign_states_to_prototypes(means, QUADRANT_PROTOTYPES)
    quality = {label: {"distance_to_prototype": d,
                        "low_confidence": d > cfg.quadrant_prototype_dmax}
               for label, d in distances.items()}
    return label_map, quality


def relabel_vol_liq_states(means: np.ndarray, cfg: Config):
    stress_score = means.mean(axis=1)
    order = np.argsort(stress_score)
    label_map = {int(k): cfg.vol_liq_order[i] for i, k in enumerate(order)}
    return label_map, {}


def _majority_vote_smooth(labels: pd.Series, window: int) -> pd.Series:
    vals = labels.values
    out = np.empty(len(vals), dtype=object)
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        window_vals = vals[lo:i + 1]
        out[i] = pd.Series(window_vals).mode().iloc[0]
    return pd.Series(out, index=labels.index)


def _emission_logprob(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    n_states = means.shape[0]
    logB = np.zeros((X.shape[0], n_states))
    for j in range(n_states):
        logB[:, j] = multivariate_normal.logpdf(X, mean=means[j], cov=covars[j], allow_singular=True)
    return logB


def _mahalanobis_distance_all_states(X: np.ndarray, means: np.ndarray, covars: np.ndarray) -> np.ndarray:
    n_states = means.shape[0]
    D2 = np.zeros((X.shape[0], n_states))
    for j in range(n_states):
        diff = X - means[j]
        inv = np.linalg.pinv(covars[j])
        D2[:, j] = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return np.sqrt(np.clip(D2, 0, None))

def _hmm_causal_forward_filter(X: np.ndarray, transmat: np.ndarray, means: np.ndarray,
                                covars: np.ndarray, log_alpha_prev: Optional[np.ndarray] = None,
                                startprob: Optional[np.ndarray] = None,
                                forgetting_floor: float = 0.02) -> np.ndarray:
    n_states = means.shape[0]
    T = X.shape[0]
    logB = _emission_logprob(X, means, covars)
    log_transmat = np.log(np.clip(transmat, 1e-12, 1.0))
    log_alphas = np.zeros((T, n_states))

    def _floor(log_p):
        if forgetting_floor <= 0:
            return log_p
        p = np.exp(log_p)
        p = (1 - forgetting_floor) * p + forgetting_floor / n_states
        return np.log(p)

    if log_alpha_prev is None:
        log_start = np.log(np.clip(startprob, 1e-12, 1.0))
        log_joint = log_start + logB[0]
    else:
        log_pred = logsumexp(log_alpha_prev[:, None] + log_transmat, axis=0)
        log_joint = log_pred + logB[0]
    log_alphas[0] = _floor(log_joint - logsumexp(log_joint))

    for t in range(1, T):
        log_pred = logsumexp(log_alphas[t - 1][:, None] + log_transmat, axis=0)
        log_joint = log_pred + logB[t]
        log_alphas[t] = _floor(log_joint - logsumexp(log_joint))

    return log_alphas


def fit_regime_axis_walkforward(feat_df: pd.DataFrame, cols: list, n_states: int,
                                 relabel_fn, cfg: Config, label_prefix: str,
                                 canonical_labels: list,
                                 covariance_type: str = "full",
                                 init_means: Optional[np.ndarray] = None,
                                 rolling_window_months: Optional[int] = None):


    X_full = feat_df[cols].dropna()
    idx = X_full.index
    n = len(idx)

    gmm_labels = pd.Series(index=idx, dtype=object)
    gmm_proba_max = pd.Series(index=idx, dtype=float)
    gmm_proba_full = pd.DataFrame(index=idx, columns=canonical_labels, dtype=float)
    hmm_labels = pd.Series(index=idx, dtype=object)
    hmm_proba_full = pd.DataFrame(index=idx, columns=canonical_labels, dtype=float)
    mahal_full = pd.DataFrame(index=idx, columns=canonical_labels, dtype=float)
    active_transmat = pd.Series(index=idx, dtype=object)
    transition_matrices = []
    hmm_label_maps = []
    hmm_means_raw_by_refit = []
    state_quality_by_refit = []
    hmm_n_refits = 0
    hmm_convergence_failures = 0

    start = cfg.min_regime_history
    if start >= n:
        raise ValueError(f"Not enough history to fit {label_prefix} regime axis "
                          f"({n} obs < min_regime_history={start})")

    checkpoints = list(range(start, n, cfg.refit_every)) + [n]
    log_alpha_prev = None  

    for i in range(len(checkpoints) - 1):
        train_end = checkpoints[i]
        pred_start, pred_end = checkpoints[i], checkpoints[i + 1]

        train_start = max(0, train_end - rolling_window_months) if rolling_window_months else 0
        X_train = X_full.iloc[train_start:train_end].values
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)

        gmm_kwargs = dict(n_components=n_states, covariance_type=covariance_type,
                           random_state=cfg.random_state)
        if init_means is not None:
            gmm = GaussianMixture(n_init=1, means_init=init_means, **gmm_kwargs)
        else:
            gmm = GaussianMixture(n_init=5, **gmm_kwargs)
        gmm.fit(X_train_s)
        label_map, quality = relabel_fn(gmm.means_, cfg)
        state_quality_by_refit.append(quality)

        hmm, best_score = None, -np.inf
        hmm_n_refits += 1
        for restart in range(cfg.hmm_n_restarts):
            candidate = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                                     random_state=cfg.random_state + restart, n_iter=cfg.hmm_n_iter)
            candidate.fit(X_train_s)
            score = candidate.score(X_train_s)
            if score > best_score:
                hmm, best_score = candidate, score
        if not hmm.monitor_.converged:
            hmm_convergence_failures += 1
        hmm_label_map, _ = relabel_fn(hmm.means_, cfg)
        transition_matrices.append(hmm.transmat_.copy())
        hmm_label_maps.append(hmm_label_map)

        hmm_means_raw_by_refit.append(scaler.inverse_transform(hmm.means_))

        inv_map = {v: k for k, v in hmm_label_map.items()}
        perm = [inv_map[lab] for lab in canonical_labels]
        means_c = hmm.means_[perm]
        covars_c = hmm.covars_[perm]
        transmat_c = hmm.transmat_[np.ix_(perm, perm)]
        startprob_c = hmm.startprob_[perm]

        X_pred = X_full.iloc[pred_start:pred_end].values
        if len(X_pred) == 0:
            continue
        X_pred_s = scaler.transform(X_pred)
        g_states = gmm.predict(X_pred_s)
        g_proba = gmm.predict_proba(X_pred_s) 
        g_proba_full = np.zeros((len(X_pred_s), n_states))
        for k, lab in label_map.items():
            g_proba_full[:, canonical_labels.index(lab)] = g_proba[:, k]

        log_alphas = _hmm_causal_forward_filter(
            X_pred_s, transmat_c, means_c, covars_c,
            log_alpha_prev=log_alpha_prev, startprob=startprob_c,
            forgetting_floor=cfg.hmm_forgetting_floor)
        log_alpha_prev = log_alphas[-1]
        h_proba = np.exp(log_alphas)  
        h_states_idx = h_proba.argmax(axis=1)
        mahal = _mahalanobis_distance_all_states(X_pred_s, means_c, covars_c)

        pred_idx = idx[pred_start:pred_end]
        gmm_labels.loc[pred_idx] = [label_map[s] for s in g_states]
        gmm_proba_max.loc[pred_idx] = g_proba.max(axis=1)
        gmm_proba_full.loc[pred_idx, :] = g_proba_full
        hmm_labels.loc[pred_idx] = [canonical_labels[s] for s in h_states_idx]
        hmm_proba_full.loc[pred_idx, :] = h_proba
        mahal_full.loc[pred_idx, :] = mahal
        for d in pred_idx:
            active_transmat.loc[d] = transmat_c

    smoothed_hmm = hmm_labels.dropna()
    smoothed_hmm = _majority_vote_smooth(smoothed_hmm, cfg.smoothing_window)
    smoothed_gmm = gmm_labels.dropna()
    smoothed_gmm = _majority_vote_smooth(smoothed_gmm, cfg.smoothing_window)

    return {
        "gmm_label": gmm_labels, "gmm_confidence": gmm_proba_max,
        "gmm_proba": gmm_proba_full, "gmm_smoothed_label": smoothed_gmm,
        "hmm_label": hmm_labels, "hmm_proba": hmm_proba_full,
        "smoothed_label": smoothed_hmm,
        "last_transition_matrix": transition_matrices[-1] if transition_matrices else None,
        "last_label_map": hmm_label_maps[-1] if hmm_label_maps else None,
        "state_quality_by_refit": state_quality_by_refit,
        "active_transmat": active_transmat,
        "mahalanobis_distance": mahal_full,
        "transition_matrices": transition_matrices,
        "hmm_label_maps": hmm_label_maps,
        "hmm_means_raw_by_refit": hmm_means_raw_by_refit,
        "hmm_n_refits": hmm_n_refits,
        "hmm_convergence_failures": hmm_convergence_failures,
    }

def train_supervised_validator(rf_features: pd.DataFrame, target: pd.Series,
                                cfg: Config, exclude_cols=None):

    exclude_cols = exclude_cols or []
    X = rf_features.drop(columns=[c for c in exclude_cols if c in rf_features.columns],
                          errors="ignore")
    df = X.join(target.rename("target")).dropna()
    X_clean, y = df.drop(columns=["target"]), df["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    reports = []
    for train_i, test_i in tscv.split(X_clean):
        clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                      class_weight="balanced_subsample",
                                      random_state=cfg.random_state)
        clf.fit(X_clean.iloc[train_i], y.iloc[train_i])
        pred = clf.predict(X_clean.iloc[test_i])
        reports.append(f1_score(y.iloc[test_i], pred, average="macro", zero_division=0))

    clf = RandomForestClassifier(n_estimators=300, max_depth=8,
                                  class_weight="balanced_subsample",
                                  random_state=cfg.random_state)
    clf.fit(X_clean, y)
    full_pred = clf.predict(X_clean)
    report_txt = classification_report(y, full_pred, zero_division=0)
    importances = pd.Series(clf.feature_importances_, index=X_clean.columns) \
        .sort_values(ascending=False)

    return {
        "walkforward_macro_f1_mean": float(np.mean(reports)),
        "walkforward_macro_f1_folds": reports,
        "in_sample_report": report_txt,
        "top_features": importances.head(10),
    }


# AXIS WIRING 
ECONOMIC_AXIS_GAUGES = ["growth_z", "inflation_z"]
STRESS_AXIS_GAUGES = ["vix_z", "equity_vol_z", "liquidity_z", "credit_z",
                       "avg_pairwise_corr", "top_eig_share"]

def classify_regimes(features: pd.DataFrame, rf_features: pd.DataFrame,
                      macro: pd.DataFrame, cfg: Config):
    quadrant_canonical = list(cfg.quadrant_labels.values())
    quadrant_init_means = np.array([QUADRANT_PROTOTYPES[lab] for lab in quadrant_canonical])
    quad = fit_regime_axis_walkforward(
        features, ECONOMIC_AXIS_GAUGES, n_states=4,
        relabel_fn=relabel_quadrant_states, cfg=cfg, label_prefix="quadrant",
        canonical_labels=quadrant_canonical, init_means=quadrant_init_means)

    vol_liq_canonical = list(cfg.vol_liq_order)
    vol_liq = fit_regime_axis_walkforward(
        features, STRESS_AXIS_GAUGES, n_states=3,
        relabel_fn=relabel_vol_liq_states, cfg=cfg, label_prefix="vol_liquidity",
        canonical_labels=vol_liq_canonical, covariance_type="diag")

    validator = None
    if cfg.recession_series in macro.columns:
        validator = train_supervised_validator(
            rf_features, macro[cfg.recession_series],
            cfg, exclude_cols=[c for c in rf_features.columns
                                if c.startswith(cfg.recession_series)])

    quad_confirm = confirm_regime_sequence(
        quad["hmm_proba"].dropna(), quad["gmm_label"], quad["active_transmat"],
        quadrant_canonical, cfg, axis="economic")
    vol_liq_confirm = confirm_regime_sequence(
        vol_liq["hmm_proba"].dropna(), vol_liq["gmm_label"], vol_liq["active_transmat"],
        vol_liq_canonical, cfg, axis="stress")

    regime_panel = pd.DataFrame({
        "quadrant": quad_confirm["confirmed_regime"],
        "quadrant_status": quad_confirm["status"],
        "quadrant_confidence_label": quad_confirm["confidence"],
        "quadrant_raw_candidate": quad_confirm["raw_candidate"],
        "quadrant_posterior_candidate": quad_confirm["posterior_candidate"],
        "quadrant_margin": quad_confirm["margin"],
        "quadrant_persistence_count": quad_confirm["persistence_count"],
        "quadrant_model_agreement": quad_confirm["model_agreement"],
        "quadrant_majority_vote": quad["smoothed_label"], 
        "quadrant_confidence": quad["gmm_confidence"],
        "quadrant_gmm": quad["gmm_smoothed_label"],
        "quadrant_hmm": quad["hmm_label"],
        "vol_liquidity": vol_liq_confirm["confirmed_regime"],
        "vol_liquidity_status": vol_liq_confirm["status"],
        "vol_liquidity_confidence_label": vol_liq_confirm["confidence"],
        "vol_liquidity_raw_candidate": vol_liq_confirm["raw_candidate"],
        "vol_liquidity_posterior_candidate": vol_liq_confirm["posterior_candidate"],
        "vol_liquidity_margin": vol_liq_confirm["margin"],
        "vol_liquidity_persistence_count": vol_liq_confirm["persistence_count"],
        "vol_liquidity_model_agreement": vol_liq_confirm["model_agreement"],
        "vol_liquidity_majority_vote": vol_liq["smoothed_label"],
        "vol_liquidity_gmm": vol_liq["gmm_smoothed_label"],
    }).dropna()

    for lab in quadrant_canonical:
        regime_panel[f"quadrant_proba_{lab}"] = quad["hmm_proba"][lab]
    for lab in vol_liq_canonical:
        regime_panel[f"vol_liquidity_proba_{lab}"] = vol_liq["hmm_proba"][lab]

    return regime_panel, quad, vol_liq, validator


def regime_occupancy_stats(labels: pd.Series, cfg: Config) -> pd.DataFrame:
    labels = labels.dropna()
    segments = _regime_segments(labels)
    total = len(labels)
    rows = []
    for lab in labels.unique():
        durations = [len(labels.loc[seg_start:seg_end]) for seg_start, seg_end, seg_lab in segments
                     if seg_lab == lab]
        n_months = sum(durations)
        rows.append({
            "state": lab, "n_months": n_months, "occupancy_share": n_months / total if total else np.nan,
            "n_episodes": len(durations),
            "avg_duration_months": float(np.mean(durations)) if durations else np.nan,
            "median_duration_months": float(np.median(durations)) if durations else np.nan,
            "p75_duration_months": float(np.percentile(durations, 75)) if durations else np.nan,
            "min_duration_months": min(durations) if durations else 0,
            "longest_duration_months": max(durations) if durations else 0,
            "below_min_occupancy": n_months < cfg.min_state_occupancy_months,
        })
    return pd.DataFrame(rows).set_index("state").sort_values("n_months", ascending=False)


def compare_state_counts(feat_df: pd.DataFrame, cols: list, cfg: Config,
                          k_range=(2, 3, 4, 5, 6)) -> pd.DataFrame:

    X = feat_df[cols].dropna().values
    X_s = StandardScaler().fit_transform(X)
    rows = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               random_state=cfg.random_state, n_init=5)
        gmm.fit(X_s)
        labels = gmm.predict(X_s)
        occ = pd.Series(labels).value_counts(normalize=True)
        rows.append({
            "k": k, "bic": gmm.bic(X_s), "aic": gmm.aic(X_s),
            "smallest_state_occupancy": occ.min(),
            "n_states_below_5pct": int((occ < 0.05).sum()),
        })
    return pd.DataFrame(rows).set_index("k")

# SECTION 6: CLASSIFICATION & CONFIRMATION
def _confirmation_confidence_label(p_candidate: float, margin: float, agreement: float) -> str:

    score = 0.5 * p_candidate + 0.3 * margin + 0.2 * agreement
    if score >= 0.65:
        return "HIGH"
    if score >= 0.45:
        return "MODERATE"
    return "LOW"

def confirm_regime_sequence(posteriors: pd.DataFrame, gmm_labels: pd.Series,
                             active_transmat: pd.Series, canonical_labels: list,
                             cfg: Config, axis: str) -> pd.DataFrame:

    if axis == "economic":
        p_enter, p_exit = cfg.confirm_economic_p_enter, cfg.confirm_economic_p_exit
        margin_min, n_confirm = cfg.confirm_economic_margin_min, cfg.confirm_economic_n_confirm
        score_min, override_p = cfg.confirm_economic_score_min, cfg.confirm_economic_override_p
    else:
        p_enter, p_exit = cfg.confirm_stress_p_enter, cfg.confirm_stress_p_exit
        margin_min, n_confirm = cfg.confirm_stress_margin_min, cfg.confirm_stress_n_confirm
        score_min, override_p = cfg.confirm_stress_score_min, cfg.confirm_stress_override_p

    idx = posteriors.index
    incumbent = posteriors.iloc[0].astype(float).idxmax()
    streak_label, streak_count = None, 0
    rows = []

    for t in idx:
        p_t = posteriors.loc[t].astype(float)
        sorted_p = p_t.sort_values(ascending=False)
        candidate = sorted_p.index[0]
        p_candidate = float(sorted_p.iloc[0])
        margin = float(sorted_p.iloc[0] - sorted_p.iloc[1])
        p_incumbent = float(p_t[incumbent])

        transmat = active_transmat.loc[t] if t in active_transmat.index else None
        transition_score = np.nan
        if transmat is not None:
            i_idx, c_idx = canonical_labels.index(incumbent), canonical_labels.index(candidate)
            transition_score = float(p_candidate * transmat[i_idx, c_idx])

        gmm_lab = gmm_labels.loc[t] if t in gmm_labels.index else None
        agreement = 1.0 if gmm_lab == candidate else 0.0

        # Issue 12: Crisis gets its own asymmetric persistence requirement.
        this_n_confirm = n_confirm
        if axis == "stress":
            if candidate == "Crisis" and incumbent != "Crisis":
                this_n_confirm = cfg.confirm_crisis_entry_n_confirm
            elif incumbent == "Crisis" and candidate != "Crisis":
                this_n_confirm = cfg.confirm_crisis_exit_n_confirm

        if candidate == incumbent:
            status = "Confirmed"
            streak_label, streak_count = None, 0
            logged_persistence = 0
        else:
            passes_threshold = p_candidate > p_enter
            hysteresis_gap = p_enter - p_exit
            passes_hysteresis = (p_candidate - p_incumbent) > hysteresis_gap
            passes_margin = margin > margin_min
            passes_score = (transition_score > score_min) if not np.isnan(transition_score) else True
            if passes_threshold and passes_hysteresis and passes_margin and passes_score:
                streak_count = streak_count + 1 if streak_label == candidate else 1
                streak_label = candidate
                logged_persistence = streak_count 
                required_n = 1 if p_candidate > override_p else this_n_confirm
                if streak_count >= required_n:
                    incumbent = candidate
                    streak_label, streak_count = None, 0
                    status = "Confirmed transition"
                else:
                    status = "Transition Watch"
            else:
                streak_label, streak_count = None, 0
                logged_persistence = 0
                status = "Confirmed"

        rows.append({
            "date": t, "confirmed_regime": incumbent, "raw_candidate": candidate,
            "posterior_candidate": p_candidate, "margin": margin,
            "transition_score": transition_score, "model_agreement": agreement,
            "persistence_count": logged_persistence, "status": status,
            "confidence": _confirmation_confidence_label(p_candidate, margin, agreement),
        })

    return pd.DataFrame(rows).set_index("date")

# SECTION 7: TRANSITION FORECASTING
def structural_forecast_panel(posteriors: pd.DataFrame, active_transmat: pd.Series,
                               canonical_labels: list, horizons: tuple) -> dict:
    idx = posteriors.index
    max_h = max(horizons)
    out = {h: pd.DataFrame(index=idx, columns=canonical_labels, dtype=float) for h in horizons}
    for t in idx:
        if t not in active_transmat.index:
            continue
        A = active_transmat.loc[t]
        p = posteriors.loc[t].astype(float).values
        Ah = np.eye(len(canonical_labels))
        for h in range(1, max_h + 1):
            Ah = Ah @ A
            if h in horizons:
                out[h].loc[t] = p @ Ah
    return out

def transition_risk_from_structural(confirmed_regime: pd.Series, structural_panel: dict,
                                     canonical_labels: list) -> dict:
    out = {}
    for h, panel in structural_panel.items():
        idx = panel.index.intersection(confirmed_regime.index)
        cur = confirmed_regime.loc[idx]
        change_risk = pd.Series(index=idx, dtype=float)
        dest = pd.DataFrame(index=idx, columns=canonical_labels, dtype=float)
        for t in idx:
            row = panel.loc[t]
            if row.isna().all():
                continue
            cur_state = cur.loc[t]
            p_stay = row.get(cur_state, np.nan)
            change_risk.loc[t] = 1 - p_stay if not pd.isna(p_stay) else np.nan
            others = row.drop(labels=[cur_state], errors="ignore")
            total = others.sum()
            if total > 0:
                dest.loc[t] = (others / total).reindex(canonical_labels)
        out[h] = {"change_risk": change_risk, "conditional_destination": dest}
    return out


def _state_duration_series(labels: pd.Series) -> pd.Series:
    grp = (labels != labels.shift()).cumsum()
    return labels.groupby(grp).cumcount() + 1


def feature_conditioned_forecast_walkforward(features: pd.DataFrame, axis_cols: list,
                                              posteriors: pd.DataFrame, confirmed_regime: pd.Series,
                                              canonical_labels: list, horizons: tuple,
                                              cfg: Config) -> dict:
    duration = _state_duration_series(confirmed_regime)
    duration.name = "state_duration"

    X_full = features[axis_cols].join(posteriors.add_prefix("post_")).join(duration).dropna()
    idx = X_full.index
    n = len(idx)
    out = {h: pd.DataFrame(index=idx, columns=canonical_labels, dtype=float) for h in horizons}

    start = cfg.min_regime_history
    if start >= n:
        return out
    checkpoints = list(range(start, n, cfg.refit_every)) + [n]

    for h in horizons:
        y_full = confirmed_regime.reindex(idx).shift(-h)
        for i in range(len(checkpoints) - 1):
            train_end, pred_start, pred_end = checkpoints[i], checkpoints[i], checkpoints[i + 1]

            purged_end = train_end - h
            if purged_end <= 0:
                continue
            y_train = y_full.iloc[:purged_end].dropna()
            X_train = X_full.iloc[:purged_end].loc[y_train.index]
            X_pred = X_full.iloc[pred_start:pred_end]
            if len(X_pred) == 0 or y_train.nunique() < 2 or len(y_train) < 10:
                continue
            try:
                clf = LogisticRegression(max_iter=1000, C=cfg.forecast_logreg_C,
                                          class_weight="balanced")
                clf.fit(X_train.values, y_train.values)
                proba = clf.predict_proba(X_pred.values)
                proba_full = pd.DataFrame(0.0, index=X_pred.index, columns=canonical_labels)
                for k, cls in enumerate(clf.classes_):
                    proba_full[cls] = proba[:, k]
                out[h].loc[X_pred.index] = proba_full
            except Exception:
                continue
    return out

def forecast_agreement(structural_panel: dict, feature_panel: dict, canonical_labels: list) -> dict:
    out = {}
    for h in structural_panel:
        if h not in feature_panel:
            continue
        s, f = structural_panel[h], feature_panel[h]
        idx = s.index.intersection(f.index)
        s, f = s.loc[idx, canonical_labels], f.loc[idx, canonical_labels]
        mad = (s - f).abs().sum(axis=1) / 2.0
        out[h] = (1 - mad).dropna()
    return out

def simulate_regime_paths(start_state: str, transmat: np.ndarray, canonical_labels: list,
                           horizon: int, n_paths: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    start_idx = canonical_labels.index(start_state)
    cum = np.cumsum(transmat, axis=1)
    state = np.full(n_paths, start_idx, dtype=int)
    paths = np.zeros((n_paths, horizon), dtype=int)
    for h in range(horizon):
        r = rng.random(n_paths)
        nxt = (r[:, None] < cum[state]).argmax(axis=1)
        paths[:, h] = nxt
        state = nxt
    return paths

def summarize_mc_paths(paths: np.ndarray, start_state: str, canonical_labels: list,
                        report_horizons: tuple) -> dict:
    n_paths, horizon = paths.shape
    start_idx = canonical_labels.index(start_state)
    out = {"start_state": start_state, "n_paths": n_paths, "horizon": horizon}

    ever_visited = {lab: float((paths == k).any(axis=1).mean())
                    for k, lab in enumerate(canonical_labels)}
    out["ever_visited"] = ever_visited
    out["remain_throughout"] = float((paths == start_idx).all(axis=1).mean())

    hitting, conditional_destination = {}, {}
    for h in report_horizons:
        h = min(h, horizon)
        window = paths[:, :h]
        changed = (window != start_idx)
        transitioned = changed.any(axis=1)
        hitting[h] = float(transitioned.mean())
        if transitioned.any():
            first_change_col = changed.argmax(axis=1)  
            dest_states = window[np.arange(n_paths), first_change_col][transitioned]
            counts = np.bincount(dest_states, minlength=len(canonical_labels))
            conditional_destination[h] = {
                canonical_labels[k]: float(counts[k] / transitioned.sum())
                for k in range(len(canonical_labels))
            }
        else:
            conditional_destination[h] = {lab: np.nan for lab in canonical_labels}
    out["transition_within_h"] = hitting
    out["conditional_destination"] = conditional_destination
    return out

def expected_state_duration_analytic(transmat: np.ndarray, canonical_labels: list) -> dict:
    return {lab: float(1.0 / max(1e-6, 1 - transmat[i, i]))
            for i, lab in enumerate(canonical_labels)}

def regime_survival_curve(labels: pd.Series, state: str):

    labels = labels.dropna()
    segments = _regime_segments(labels)
    durations = []
    censored_current_duration = None
    for i, (seg_start, seg_end, seg_lab) in enumerate(segments):
        if seg_lab != state:
            continue
        d = len(labels.loc[seg_start:seg_end])
        if i == len(segments) - 1:
            censored_current_duration = d
        else:
            durations.append(d)
    if not durations:
        return pd.DataFrame(columns=["duration", "n_at_risk", "n_exit", "hazard", "survival"]), censored_current_duration
    max_d = max(durations)
    rows = []
    survival = 1.0
    for d in range(1, max_d + 1):
        n_at_risk = sum(1 for x in durations if x >= d)
        n_exit = sum(1 for x in durations if x == d)
        hazard = n_exit / n_at_risk if n_at_risk > 0 else np.nan
        rows.append({"duration": d, "n_at_risk": n_at_risk, "n_exit": n_exit,
                      "hazard": hazard, "survival": survival})
        survival = survival * (1 - hazard) if not np.isnan(hazard) else survival
    return pd.DataFrame(rows), censored_current_duration

def expected_remaining_duration_empirical(survival_curve: pd.DataFrame, current_duration: int,
                                           min_episodes: int = 3) -> float:
    if survival_curve.empty:
        return np.nan
    n_completed_episodes = int(survival_curve.iloc[0]["n_at_risk"])
    if n_completed_episodes < min_episodes:
        return np.nan
    sc = survival_curve.set_index("duration")["survival"]
    if current_duration not in sc.index:
        if current_duration > sc.index.max():
            return 0.0  
        current_duration = sc.index.min()
    s_now = sc.loc[current_duration]
    if s_now <= 0:
        return 0.0
    tail = sc.loc[current_duration:]
    return float(tail.sum() / s_now)

def forecast_recession_probability(rf_features: pd.DataFrame, recession: pd.Series,
                                    cfg: Config, horizons=(3, 6)) -> dict:
    
    exclude = [c for c in rf_features.columns if c.startswith(cfg.recession_series)]
    X_full = rf_features.drop(columns=[c for c in exclude if c in rf_features.columns],
                               errors="ignore").dropna()
    idx = X_full.index
    n = len(idx)
    out = {}
    start = cfg.min_regime_history
    if start >= n:
        return out
    checkpoints = list(range(start, n, cfg.refit_every)) + [n]
    for h in horizons:
        y_full = recession.reindex(idx).shift(-h)
        prob = pd.Series(index=idx, dtype=float)
        for i in range(len(checkpoints) - 1):
            train_end, pred_start, pred_end = checkpoints[i], checkpoints[i], checkpoints[i + 1]

            purged_end = train_end - h
            if purged_end <= 0:
                continue
            y_train = y_full.iloc[:purged_end].dropna()
            X_train = X_full.iloc[:purged_end].loc[y_train.index]
            X_pred = X_full.iloc[pred_start:pred_end]
            if len(X_pred) == 0 or y_train.nunique() < 2 or len(y_train) < 10:
                continue
            clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                          class_weight="balanced_subsample",
                                          random_state=cfg.random_state)
            clf.fit(X_train.values, y_train.values)
            classes = list(clf.classes_)
            if 1 not in classes:
                continue
            p1 = clf.predict_proba(X_pred.values)[:, classes.index(1)]
            prob.loc[X_pred.index] = p1
        out[h] = prob
    return out

def probability_momentum(prob_series: pd.Series, lookbacks=(1, 3)) -> pd.DataFrame:
    out = pd.DataFrame(index=prob_series.index)
    out["level"] = prob_series
    for lb in lookbacks:
        out[f"delta_{lb}m"] = prob_series - prob_series.shift(lb)
    return out

def forecast_calibration(prob_series: pd.Series, realized: pd.Series, bins=None) -> dict:
    bins = bins or (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    df = pd.DataFrame({"p": prob_series, "y": realized}).dropna()
    if df.empty or df["y"].nunique() < 2:
        return {"brier": np.nan, "log_loss": np.nan, "reliability": pd.DataFrame(), "n": len(df)}
    brier = float(np.mean((df["p"] - df["y"]) ** 2))
    eps = 1e-6
    p_clipped = df["p"].clip(eps, 1 - eps)
    ll = float(-np.mean(df["y"] * np.log(p_clipped) + (1 - df["y"]) * np.log(1 - p_clipped)))
    df["bin"] = pd.cut(df["p"], bins=bins, include_lowest=True)
    rel = df.groupby("bin", observed=True).agg(mean_forecast=("p", "mean"),
                                                 realized_freq=("y", "mean"),
                                                 n=("y", "size"))
    return {"brier": brier, "log_loss": ll, "reliability": rel, "n": len(df)}


def forecast_confidence_gradient(structural_panel: dict) -> dict:
    return {h: float(panel.max(axis=1).mean()) for h, panel in structural_panel.items()}

def current_forecast_confidence(structural_panel: dict, agreement_panel: dict, h: int) -> dict:
    out = {"max_proba": np.nan, "margin": np.nan, "entropy_norm": np.nan,
           "structural_feature_agreement": np.nan}
    panel = structural_panel.get(h)
    if panel is None:
        return out
    valid = panel.dropna(how="all")
    if valid.empty:
        return out
    row = valid.iloc[[-1]].astype(float)
    sorted_vals = np.sort(row.values, axis=1)[0, ::-1]
    out["max_proba"] = float(sorted_vals[0])
    out["margin"] = float(sorted_vals[0] - sorted_vals[1]) if len(sorted_vals) > 1 else np.nan
    out["entropy_norm"] = float(posterior_entropy(row, normalized=True).iloc[0])
    agree_series = (agreement_panel or {}).get(h)
    if agree_series is not None and not agree_series.dropna().empty:
        common = agree_series.dropna().index.intersection([valid.index[-1]])
        if len(common):
            out["structural_feature_agreement"] = float(agree_series.loc[common[-1]])
    return out

def build_forecast_engine(features: pd.DataFrame, regime_panel: pd.DataFrame,
                           quad_fit: dict, vol_liq_fit: dict, rf_features: pd.DataFrame,
                           macro: pd.DataFrame, cfg: Config) -> dict:
    quadrant_canonical = list(cfg.quadrant_labels.values())
    vol_liq_canonical = list(cfg.vol_liq_order)

    result = {}
    for axis, canonical_labels, gauges, fit, horizons in (
        ("quadrant", quadrant_canonical, ECONOMIC_AXIS_GAUGES, quad_fit, cfg.forecast_horizons_economic),
        ("vol_liquidity", vol_liq_canonical, STRESS_AXIS_GAUGES, vol_liq_fit, cfg.forecast_horizons_stress),
    ):
        posteriors = fit["hmm_proba"].dropna()
        confirmed = regime_panel[axis]
        active_transmat = fit["active_transmat"]

        structural = structural_forecast_panel(posteriors, active_transmat, canonical_labels, horizons)
        risk = transition_risk_from_structural(confirmed, structural, canonical_labels)

        axis_cols = []
        for g in gauges:
            base = g[:-2] if g.endswith("_z") else g
            axis_cols += [g, f"{base}_D", f"{base}_A"]
        axis_cols = [c for c in axis_cols if c in features.columns]
        feature_fcast = feature_conditioned_forecast_walkforward(
            features, axis_cols, posteriors, confirmed, canonical_labels, horizons, cfg)
        agreement = forecast_agreement(structural, feature_fcast, canonical_labels)

        last_transmat = fit["last_transition_matrix"]
        last_label_map = fit["last_label_map"] or {}
        if last_transmat is not None and last_label_map:
            inv_map = {v: k for k, v in last_label_map.items()}
            perm = [inv_map[lab] for lab in canonical_labels]
            transmat_c = last_transmat[np.ix_(perm, perm)]
        else:
            transmat_c = None

        cur_state = confirmed.iloc[-1] if len(confirmed) else None
        cur_duration = int(_state_duration_series(confirmed).iloc[-1]) if len(confirmed) else None

        mc_summary = None
        if transmat_c is not None and cur_state is not None:
            paths = simulate_regime_paths(cur_state, transmat_c, canonical_labels,
                                           cfg.forecast_mc_horizon, cfg.forecast_mc_paths,
                                           cfg.random_state)
            mc_summary = summarize_mc_paths(paths, cur_state, canonical_labels, horizons)

        exp_duration = expected_state_duration_analytic(transmat_c, canonical_labels) if transmat_c is not None else {}
        survival_curve, censored_duration = (regime_survival_curve(confirmed, cur_state)
                                              if cur_state is not None else (pd.DataFrame(), None))
        remaining_empirical = expected_remaining_duration_empirical(
            survival_curve, censored_duration if censored_duration is not None else (cur_duration or 1))

        confidence_gradient = forecast_confidence_gradient(structural)
        current_confidence = {h: current_forecast_confidence(structural, agreement, h) for h in horizons}

        result[axis] = {
            "canonical_labels": canonical_labels, "horizons": horizons,
            "structural": structural, "feature_conditioned": feature_fcast,
            "agreement": agreement, "transition_risk": risk,
            "mc_summary": mc_summary, "current_state": cur_state,
            "current_state_duration": cur_duration,
            "expected_duration_analytic": exp_duration,
            "survival_curve": survival_curve,
            "expected_remaining_duration_empirical": remaining_empirical,
            "confidence_gradient": confidence_gradient,
            "current_confidence": current_confidence,
        }

    recession_fcast = {}
    if cfg.recession_series in macro.columns:
        recession_prob = forecast_recession_probability(
            rf_features, macro[cfg.recession_series], cfg, horizons=(3, 6))
        for h, p in recession_prob.items():
            recession_fcast[h] = probability_momentum(p)
        realized_3m = macro[cfg.recession_series].reindex(
            recession_prob.get(3, pd.Series(dtype=float)).index).shift(-3)
        recession_fcast["calibration_3m"] = forecast_calibration(recession_prob.get(3, pd.Series(dtype=float)),
                                                                   realized_3m)
    result["recession"] = recession_fcast

    return result

# SECTION 8: CONFIDENCE & STABILITY
def posterior_margin(posteriors: pd.DataFrame) -> pd.Series:
    sorted_vals = np.sort(posteriors.values.astype(float), axis=1)[:, ::-1]
    return pd.Series(sorted_vals[:, 0] - sorted_vals[:, 1], index=posteriors.index)

def posterior_entropy(posteriors: pd.DataFrame, normalized: bool = True) -> pd.Series:
    p = np.clip(posteriors.values.astype(float), 1e-12, 1.0)
    H = -(p * np.log(p)).sum(axis=1)
    if normalized:
        H = H / np.log(p.shape[1])
    return pd.Series(H, index=posteriors.index)

def entropy_band(h_norm: float, bands: tuple) -> str:

    if h_norm is None or (isinstance(h_norm, float) and np.isnan(h_norm)):
        return "n/a"
    b1, b2, b3 = bands
    if h_norm < b1:
        return "Low"
    if h_norm < b2:
        return "Moderate"
    if h_norm < b3:
        return "Elevated"
    return "High"

def transition_entropy_from_matrix(transmat: np.ndarray, canonical_labels: list,
                                    normalized: bool = True) -> dict:
    out = {}
    K = len(canonical_labels)
    for i, lab in enumerate(canonical_labels):
        row = np.clip(transmat[i], 1e-12, 1.0)
        H = -(row * np.log(row)).sum()
        out[lab] = float(H / np.log(K)) if normalized else float(H)
    return out

def cluster_fit_label(current_distance: float, historical_distances: pd.Series, bands: tuple) -> str:

    hist = historical_distances.dropna()
    if hist.empty or current_distance is None or np.isnan(current_distance):
        return "n/a"
    pct = float((hist <= current_distance).mean())
    b1, b2, b3 = bands
    if pct <= b1:
        return "Strong"
    if pct <= b2:
        return "Normal"
    if pct <= b3:
        return "Weak"
    return "Outlier"

def gmm_membership_bootstrap_ci(features: pd.DataFrame, axis_cols: list, n_states: int,
                                 relabel_fn, cfg: Config, canonical_labels: list,
                                 n_boot: Optional[int] = None) -> dict:

    X_full = features[axis_cols].dropna()
    if len(X_full) < cfg.min_regime_history:
        return {}
    X_train = X_full.values
    x_current = X_train[-1:]
    n = len(X_train)
    block = cfg.block_size_months
    n_boot = n_boot or cfg.confidence_n_bootstrap
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.random_state + 7)

    gmm_kwargs = dict(n_components=n_states, covariance_type="full" if n_states == 4 else "diag",
                       random_state=cfg.random_state)
    draws = {lab: [] for lab in canonical_labels}
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        sample = np.concatenate([X_train[s:s + block] for s in starts])[:n]
        try:
            scaler = StandardScaler().fit(sample)
            sample_s = scaler.transform(sample)
            x_cur_s = scaler.transform(x_current)
            gmm = GaussianMixture(n_init=2, **gmm_kwargs)
            gmm.fit(sample_s)
            label_map, _ = relabel_fn(gmm.means_, cfg)
            proba = gmm.predict_proba(x_cur_s)[0]
            for k, lab in label_map.items():
                draws[lab].append(proba[k])
        except Exception:
            continue

    out = {}
    for lab in canonical_labels:
        vals = np.array(draws[lab])
        out[lab] = ((float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
                     if len(vals) >= 10 else (np.nan, np.nan))
    return out

def transition_matrix_bootstrap_ci(features: pd.DataFrame, axis_cols: list, n_states: int,
                                    relabel_fn, cfg: Config, canonical_labels: list,
                                    n_boot: Optional[int] = None) -> dict:
   
    X_full = features[axis_cols].dropna()
    if len(X_full) < cfg.min_regime_history:
        return {}
    X_train = X_full.values
    n = len(X_train)
    block = cfg.block_size_months
    n_boot = n_boot or cfg.confidence_n_bootstrap
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.random_state + 11)
    covariance_type = "full" if n_states == 4 else "diag"

    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)

        pieces = [X_train[s:s + block] for s in starts]
        lengths = [len(p) for p in pieces]
        sample = np.concatenate(pieces)
        if len(sample) > n:
            overflow = len(sample) - n
            sample = sample[:n]
            lengths[-1] -= overflow
        lengths = [l for l in lengths if l > 0]
        try:
            scaler = StandardScaler().fit(sample)
            sample_s = scaler.transform(sample)
            hmm = GaussianHMM(n_components=n_states, covariance_type=covariance_type,
                               random_state=cfg.random_state, n_iter=cfg.confidence_bootstrap_hmm_n_iter)
            hmm.fit(sample_s, lengths=lengths)
            label_map, _ = relabel_fn(hmm.means_, cfg)
            inv_map = {v: k for k, v in label_map.items()}
            perm = [inv_map[lab] for lab in canonical_labels]
            draws.append(hmm.transmat_[np.ix_(perm, perm)])
        except Exception:
            continue

    if not draws:
        return {}
    arr = np.array(draws)
    out = {}
    for i, from_lab in enumerate(canonical_labels):
        out[from_lab] = {}
        for j, to_lab in enumerate(canonical_labels):
            vals = arr[:, i, j]
            out[from_lab][to_lab] = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return out

def recession_probability_bootstrap_ci(rf_features: pd.DataFrame, recession: pd.Series, cfg: Config,
                                        horizon: int = 3, n_boot: Optional[int] = None) -> tuple:

    exclude = [c for c in rf_features.columns if c.startswith(cfg.recession_series)]
    X_full = rf_features.drop(columns=[c for c in exclude if c in rf_features.columns], errors="ignore")
    df = X_full.join(recession.rename("target")).dropna()
    y_full = df["target"].shift(-horizon)
    valid = y_full.dropna().index
    X_train_full, y_train_full = df.loc[valid, X_full.columns], y_full.loc[valid]
    if len(X_train_full) < cfg.min_regime_history or y_train_full.nunique() < 2:
        return (np.nan, np.nan)
    x_current = df[X_full.columns].iloc[[-1]].values
    n = len(X_train_full)
    block = cfg.block_size_months
    n_boot = n_boot or cfg.confidence_n_bootstrap
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.random_state + 13)
    Xv, yv = X_train_full.values, y_train_full.values
    idx_all = np.arange(n)

    probs = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(n - block, 1), size=n_blocks)
        take = np.concatenate([idx_all[s:s + block] for s in starts])[:n]
        Xs, ys = Xv[take], yv[take]
        if len(np.unique(ys)) < 2:
            continue
        try:
            clf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                          class_weight="balanced_subsample",
                                          random_state=cfg.random_state)
            clf.fit(Xs, ys)
            classes = list(clf.classes_)
            if 1 not in classes:
                continue
            probs.append(clf.predict_proba(x_current)[0, classes.index(1)])
        except Exception:
            continue

    if len(probs) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(probs, 2.5)), float(np.percentile(probs, 97.5)))

def regime_evidence_snapshot(features: pd.DataFrame, driving_cols: list, all_z_cols: list) -> dict:
    if features.empty:
        return {"driving": {}, "context": {}}
    last = features.iloc[-1]
    driving = {c: float(last[c]) for c in driving_cols if c in features.columns and pd.notna(last[c])}
    context_cols = [c for c in all_z_cols if c not in driving_cols]
    context = {c: float(last[c]) for c in context_cols if c in features.columns and pd.notna(last[c])}
    return {"driving": driving, "context": context}

def regime_stability_diagnostics(regime_panel: pd.DataFrame, axis_prefix: str, canonical_labels: list,
                                  typical_duration: dict, cfg: Config, trailing: int = 6) -> dict:
    proba_cols = [f"{axis_prefix}_proba_{lab}" for lab in canonical_labels]
    if not all(c in regime_panel.columns for c in proba_cols):
        return {}
    proba = regime_panel[proba_cols].dropna()
    proba.columns = canonical_labels
    if proba.empty:
        return {}

    confirmed = regime_panel[axis_prefix]
    state_age = int(_state_duration_series(confirmed).iloc[-1])
    top_prob_series = proba.max(axis=1).iloc[-trailing:]
    posterior_volatility = float(top_prob_series.std()) if len(top_prob_series) > 1 else np.nan
    entropy_now = float(posterior_entropy(proba, normalized=True).iloc[-1])
    margin_now = float(posterior_margin(proba).iloc[-1])
    agreement_now = float(regime_panel[f"{axis_prefix}_model_agreement"].iloc[-1])
    trailing_labels = confirmed.iloc[-trailing:]
    n_switches = int((trailing_labels != trailing_labels.shift()).sum() - 1) if len(trailing_labels) > 1 else 0
    n_switches = max(n_switches, 0)

    med = typical_duration.get("median_duration_months", np.nan)
    age_ratio = (state_age / med) if med and not np.isnan(med) and med > 0 else np.nan

    checks = {
        "state_age_typical_or_beyond": bool((not np.isnan(age_ratio)) and age_ratio >= 0.5),
        "low_posterior_volatility": bool((not np.isnan(posterior_volatility)) and posterior_volatility < 0.10),
        "low_entropy": bool(entropy_now < 0.50),
        "wide_margin": bool(margin_now >= 0.15),
        "models_agree": bool(agreement_now >= 0.5),
        "no_recent_switches": bool(n_switches == 0),
    }
    n_pass, n_total = sum(checks.values()), len(checks)
    label = "High" if n_pass == n_total else ("Low" if n_pass <= n_total // 3 else "Moderate")

    return {
        "state_age": state_age, "age_vs_median_ratio": age_ratio,
        "posterior_volatility_6m": posterior_volatility, "entropy_now": entropy_now,
        "margin_now": margin_now, "agreement_now": agreement_now,
        "n_switches_6m": n_switches, "checks": checks,
        "n_pass": n_pass, "n_total": n_total, "label": label,
    }

def classify_state_confidence(posterior: float, margin: float, entropy_norm: float,
                               agreement: float, cluster_fit: str, ood_flag: bool,
                               gmm_membership_ci_width: float) -> dict:
    checks = {
        "posterior_high": bool(posterior >= 0.55),
        "margin_wide": bool(margin >= 0.15),
        "entropy_low_or_moderate": bool(entropy_norm <= 0.50),
        "models_agree": bool(agreement >= 0.5),
        "cluster_fit_ok": cluster_fit in ("Strong", "Normal"),
        "not_ood": bool(not ood_flag),
    }

    if gmm_membership_ci_width is not None and not np.isnan(gmm_membership_ci_width):
        checks["gmm_membership_stable"] = bool(gmm_membership_ci_width <= 0.25)
    n_pass, n_total = sum(checks.values()), len(checks)
    label = "HIGH" if n_pass == n_total else ("LOW" if n_pass <= max(1, n_total // 3) else "MODERATE")
    return {"checks": checks, "n_pass": n_pass, "n_total": n_total, "label": label}

def build_confidence_engine(features: pd.DataFrame, regime_panel: pd.DataFrame,
                             quad_fit: dict, vol_liq_fit: dict, forecast: dict,
                             rf_features: pd.DataFrame, macro: pd.DataFrame,
                             gauge_info: dict, cfg: Config) -> dict:

    quadrant_canonical = list(cfg.quadrant_labels.values())
    vol_liq_canonical = list(cfg.vol_liq_order)
    all_z_cols = [c for c in features.columns if c.endswith("_z")]

    result = {}
    for axis, canonical_labels, driving_cols, fit, relabel_fn, n_states in (
        ("quadrant", quadrant_canonical, ECONOMIC_AXIS_GAUGES, quad_fit, relabel_quadrant_states, 4),
        ("vol_liquidity", vol_liq_canonical, STRESS_AXIS_GAUGES, vol_liq_fit, relabel_vol_liq_states, 3),
    ):
        proba_cols = [f"{axis}_proba_{lab}" for lab in canonical_labels]
        if not all(c in regime_panel.columns for c in proba_cols):
            result[axis] = {}
            continue
        proba = regime_panel[proba_cols].dropna()
        proba.columns = canonical_labels
        if proba.empty:
            result[axis] = {}
            continue

        confirmed = regime_panel[axis]
        cur_state = confirmed.iloc[-1]
        p_row = proba.iloc[-1]
        margin_now = float(posterior_margin(proba).iloc[-1])
        entropy_now = float(posterior_entropy(proba, normalized=True).iloc[-1])
        others = p_row.drop(index=cur_state).sort_values(ascending=False)
        challenger = (others.index[0], float(others.iloc[0])) if len(others) else (None, np.nan)
        agreement_now = float(regime_panel[f"{axis}_model_agreement"].iloc[-1])

        occ = regime_occupancy_stats(confirmed, cfg)
        typical_duration = occ.loc[cur_state].to_dict() if cur_state in occ.index else {}

        cluster_fit_lab, cluster_dist, ood_flag = "n/a", np.nan, False
        mahal = fit.get("mahalanobis_distance")
        if mahal is not None and not mahal.dropna(how="all").empty:
            common_idx = mahal.dropna(how="all").index.intersection(confirmed.index)
            dist_to_confirmed = pd.Series(
                {t: mahal.loc[t, confirmed.loc[t]] for t in common_idx
                 if confirmed.loc[t] in mahal.columns and pd.notna(mahal.loc[t, confirmed.loc[t]])}
            ).sort_index()
            if len(dist_to_confirmed) and confirmed.index[-1] in dist_to_confirmed.index:
                cluster_dist = float(dist_to_confirmed.iloc[-1])
                cluster_fit_lab = cluster_fit_label(cluster_dist, dist_to_confirmed, cfg.confidence_cluster_fit_bands)
                pct = float((dist_to_confirmed <= cluster_dist).mean())
                ood_flag = pct >= cfg.confidence_ood_percentile / 100.0

        last_transmat = fit["last_transition_matrix"]
        last_label_map = fit["last_label_map"] or {}
        transmat_c = None
        if last_transmat is not None and last_label_map:
            inv_map = {v: k for k, v in last_label_map.items()}
            perm = [inv_map[lab] for lab in canonical_labels]
            transmat_c = last_transmat[np.ix_(perm, perm)]
        trans_entropy = transition_entropy_from_matrix(transmat_c, canonical_labels) if transmat_c is not None else {}
        trans_entropy_now = trans_entropy.get(cur_state, np.nan)

        state_ci = gmm_membership_bootstrap_ci(
            features, driving_cols, n_states, relabel_fn, cfg, canonical_labels)
        ci_width_current = (state_ci[cur_state][1] - state_ci[cur_state][0]
                             if cur_state in state_ci and not np.isnan(state_ci[cur_state][0]) else np.nan)
        transmat_ci = transition_matrix_bootstrap_ci(
            features, driving_cols, n_states, relabel_fn, cfg, canonical_labels)

        stability = regime_stability_diagnostics(regime_panel, axis, canonical_labels, typical_duration, cfg)

        fdata = (forecast or {}).get(axis, {})
        horizons = fdata.get("horizons", ())
        fwd_h = 3 if 3 in horizons else (horizons[-1] if horizons else None)
        fwd_change_risk, fwd_dest, fwd_conf = np.nan, None, np.nan
        if fwd_h is not None:
            risk_h = fdata.get("transition_risk", {}).get(fwd_h)
            if risk_h is not None and not risk_h["change_risk"].dropna().empty:
                fwd_change_risk = float(risk_h["change_risk"].iloc[-1])
                dest_row = risk_h["conditional_destination"].iloc[-1].dropna().sort_values(ascending=False)
                if len(dest_row):
                    fwd_dest = (dest_row.index[0], float(dest_row.iloc[0]))

            fwd_conf = fdata.get("current_confidence", {}).get(fwd_h, {}).get("max_proba", np.nan)

        confidence_cls = classify_state_confidence(
            float(p_row[cur_state]), margin_now, entropy_now, agreement_now,
            cluster_fit_lab, ood_flag, ci_width_current)

        result[axis] = {
            "canonical_labels": canonical_labels, "confirmed_regime": cur_state,
            "posterior": p_row.to_dict(), "posterior_confirmed": float(p_row[cur_state]),
            "margin": margin_now, "entropy_norm": entropy_now,
            "entropy_band": entropy_band(entropy_now, cfg.confidence_entropy_bands),
            "most_likely_challenger": challenger,
            "state_age": stability.get("state_age"), "typical_duration": typical_duration,
            "gmm_hmm_agreement": agreement_now >= 0.5,
            "cluster_fit_distance": cluster_dist, "cluster_fit_label": cluster_fit_lab,
            "ood_flag": ood_flag,
            "gmm_membership_ci": state_ci, "gmm_membership_ci_width_current": ci_width_current,
            "transmat_ci": transmat_ci,
            "transition_entropy": trans_entropy, "transition_entropy_current": trans_entropy_now,
            "transition_entropy_band": entropy_band(trans_entropy_now, cfg.confidence_entropy_bands),
            "forward_horizon": fwd_h, "forward_change_risk": fwd_change_risk,
            "forward_destination": fwd_dest, "forward_confidence": fwd_conf,
            "stability": stability, "state_confidence": confidence_cls,
            "evidence": regime_evidence_snapshot(features, driving_cols, all_z_cols),
        }

    factor_uncertainty = {}
    for key, val in gauge_info.items():
        if not key.endswith("_explained_var"):
            continue
        block = key[:-len("_explained_var")]
        hist = gauge_info.get(f"{block}_evr_history", [])
        med = float(np.median(hist)) if hist else np.nan
        ratio = (val / med) if med and not np.isnan(med) and med > 0 else np.nan
        factor_uncertainty[block] = {
            "current_evr": val, "historical_median_evr": med,
            "loading_stability": gauge_info.get(f"{block}_loading_stability", np.nan),
            "low_reliability": bool((not np.isnan(ratio)) and ratio < cfg.confidence_factor_evr_ratio_warn),
        }
    result["factor_uncertainty"] = factor_uncertainty
    recession_ci = {}
    if cfg.recession_series in macro.columns and forecast and forecast.get("recession"):
        recession_ci["ci_3m"] = recession_probability_bootstrap_ci(
            rf_features, macro[cfg.recession_series], cfg, horizon=3)
    result["recession_ci"] = recession_ci

    return result

# SECTION 9: VALIDATION & DEFENSIBILITY
# SECTION 10: CROSS-ASSET APPLICATION
# 10A. TACTICAL ASSET ALLOCATION:
def walkforward_regime_allocation(prices: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config):

    rets = prices.pct_change().dropna(how="all")
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    quadrant = regime_panel.loc[idx, "quadrant"]
    vol_liq = regime_panel.loc[idx, "vol_liquidity"]

    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    strategy_gross_ret = pd.Series(0.0, index=idx)
    leverage_used = pd.Series(1.0, index=idx)
    selected_log = {}

    leverage_by_state = {"Calm": cfg.leverage_calm, "Stress": cfg.leverage_stress,
                          "Crisis": cfg.leverage_crisis}

    for i in range(1, len(idx)):
        t = idx[i]
        hist = rets.iloc[:i]
        cur_regime = quadrant.iloc[i - 1] 
        cur_vol_liq = vol_liq.iloc[i - 1]

        same_regime_hist = hist[quadrant.iloc[:i].values == cur_regime]
        use_hist = same_regime_hist if len(same_regime_hist) >= cfg.min_same_regime_months else hist

        mu = use_hist.mean() * 12
        sd = use_hist.std() * np.sqrt(12)
        sharpe = (mu / sd.replace(0, np.nan)).dropna()
        if sharpe.empty:
            continue
        top = sharpe.sort_values(ascending=False).head(cfg.top_n_assets)
        raw_w = top.clip(lower=0.01)
        w = raw_w / raw_w.sum()

        trailing = strategy_gross_ret.iloc[max(0, i - 12):i]
        realized = (float(trailing.std()) * np.sqrt(12)
                    if len(trailing) >= 6 and trailing.std() > 0 else cfg.vol_target)
        scale = np.clip(cfg.vol_target / max(realized, 1e-6), cfg.vol_scale_floor, cfg.vol_scale_cap)

        lev_cap = leverage_by_state.get(cur_vol_liq, 1.0)
        final_scale = min(scale, lev_cap)

        weights.loc[t, w.index] = w.values * final_scale
        strategy_gross_ret.loc[t] = (rets.loc[t, w.index] * w.values).sum() * final_scale
        leverage_used.loc[t] = final_scale
        selected_log[t] = list(w.index)

    turnover = weights.diff().abs().sum(axis=1) / 2
    txn_cost = turnover * (cfg.txn_cost_bps / 1e4)
    strategy_net_ret = strategy_gross_ret - txn_cost

    return {
        "weights": weights, "gross_returns": strategy_gross_ret,
        "net_returns": strategy_net_ret, "turnover": turnover,
        "leverage": leverage_used, "selection_log": selected_log,
    }

# 10B. REGIME-CONDITIONED RISK MODEL:
def historical_var_cvar(returns: pd.Series, confidence: float):
    if returns.dropna().empty:
        return np.nan, np.nan
    q = np.nanpercentile(returns, (1 - confidence) * 100)
    cvar = returns[returns <= q].mean()
    return q, cvar

def regime_conditioned_risk_model(strategy_returns: pd.Series, regime_panel: pd.DataFrame, cfg: Config):
    idx = strategy_returns.index.intersection(regime_panel.index)
    rets = strategy_returns.loc[idx]
    quadrant = regime_panel.loc[idx, "quadrant"]
    vol_liq = regime_panel.loc[idx, "vol_liquidity"]

    by_quadrant = {}
    for regime in quadrant.dropna().unique():
        r = rets[quadrant == regime]
        var, cvar = historical_var_cvar(r, cfg.var_confidence)
        by_quadrant[regime] = {
            "n_months": len(r), "ann_return": r.mean() * 12,
            "ann_vol": r.std() * np.sqrt(12),
            f"VaR_{int(cfg.var_confidence*100)}": var,
            f"CVaR_{int(cfg.cvar_confidence*100)}": cvar,
        }
    quadrant_risk_table = pd.DataFrame(by_quadrant).T

    lookback_by_state = {"Calm": cfg.lookback_calm, "Stress": cfg.lookback_stress,
                          "Crisis": cfg.lookback_crisis}
    leverage_by_state = {"Calm": cfg.leverage_calm, "Stress": cfg.leverage_stress,
                          "Crisis": cfg.leverage_crisis}

    pit_var, pit_cvar, pit_lookback, pit_lev_cap = [], [], [], []
    for t in idx:
        state = vol_liq.loc[t]
        lb = lookback_by_state.get(state, cfg.lookback_stress)

        window = rets.loc[:t].iloc[:-1].tail(lb)
        v, c = historical_var_cvar(window, cfg.var_confidence)
        pit_var.append(v)
        pit_cvar.append(c)
        pit_lookback.append(lb)
        pit_lev_cap.append(leverage_by_state.get(state, 1.0))

    pit_risk = pd.DataFrame({
        "vol_liquidity_state": vol_liq.values,
        f"VaR_{int(cfg.var_confidence*100)}_dynamic": pit_var,
        f"CVaR_{int(cfg.cvar_confidence*100)}_dynamic": pit_cvar,
        "lookback_months_used": pit_lookback,
        "leverage_cap_used": pit_lev_cap,
    }, index=idx)

    return quadrant_risk_table, pit_risk

# SECTION 9 (continued): VALIDATION TESTS & DEFENSIBILITY SUITE
def moving_block_bootstrap_sharpe_ci(returns: pd.Series, cfg: Config, n_boot=None):
    r = returns.dropna().values
    n = len(r)
    block = cfg.block_size_months
    n_boot = n_boot or cfg.n_bootstrap
    n_blocks = int(np.ceil(n / block))
    sharpes = np.empty(n_boot)
    rng = np.random.default_rng(cfg.random_state)
    for b in range(n_boot):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        sharpes[b] = (sample.mean() / sample.std()) * np.sqrt(12) if sample.std() > 0 else np.nan
    sharpes = sharpes[~np.isnan(sharpes)]
    return {
        "sharpe_point": (r.mean() / r.std()) * np.sqrt(12) if r.std() > 0 else np.nan,
        "sharpe_ci_95": (np.nanpercentile(sharpes, 2.5), np.nanpercentile(sharpes, 97.5)),
    }

def paired_block_bootstrap_test(strategy_ret: pd.Series, benchmark_ret: pd.Series, cfg: Config):
    """Block-bootstrap test of H0: mean(strategy - benchmark) <= 0."""
    df = pd.concat([strategy_ret, benchmark_ret], axis=1).dropna()
    diff = (df.iloc[:, 0] - df.iloc[:, 1]).values
    n = len(diff)
    block = cfg.block_size_months
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(cfg.random_state + 1)
    boot_means = np.empty(cfg.n_bootstrap)
    for b in range(cfg.n_bootstrap):
        starts = rng.integers(0, n - block, size=n_blocks)
        sample = np.concatenate([diff[s:s + block] for s in starts])[:n]
        boot_means[b] = sample.mean()
    observed = diff.mean()
    centered = boot_means - boot_means.mean()
    p_value = (np.sum(centered >= observed) + 1) / (cfg.n_bootstrap + 1)
    return {"observed_mean_monthly_diff": observed, "p_value_one_sided": p_value}

def _proxy_regime_allocation_returns(quadrant_labels: pd.Series, rets: pd.DataFrame, cfg: Config):
    perm_rets = []
    for lab in quadrant_labels.dropna().unique():
        mask = quadrant_labels == lab
        sub = rets.loc[mask]
        if sub.empty:
            continue
        mu = sub.mean()
        top = mu.sort_values(ascending=False).head(cfg.top_n_assets)
        w = top.clip(lower=1e-4)
        w = w / w.sum()
        perm_rets.append((sub[w.index] * w.values).sum(axis=1))
    if not perm_rets:
        return np.nan
    return pd.concat(perm_rets).mean()

def regime_label_permutation_test(strategy_net_returns: pd.Series, regime_panel: pd.DataFrame,
                                   prices: pd.DataFrame, cfg: Config):
    quadrant = regime_panel["quadrant"].dropna()

    grp = (quadrant != quadrant.shift()).cumsum()
    episode_list = [g.values.tolist() for _, g in quadrant.groupby(grp)]
    episode_labels = [g.iloc[0] for _, g in quadrant.groupby(grp)]
    episode_lens = [len(e) for e in episode_list]
    rets = prices.pct_change().reindex(quadrant.index)
    observed_proxy_mean = _proxy_regime_allocation_returns(quadrant, rets, cfg)
    observed_strategy_mean = strategy_net_returns.reindex(quadrant.index).dropna().mean()

    rng = np.random.default_rng(cfg.random_state + 2)
    permuted_means = np.empty(cfg.n_permutation)

    for p in range(cfg.n_permutation):
        shuffled_labels_seq = list(episode_labels)
        rng.shuffle(shuffled_labels_seq)
        shuffled_full = []
        for lab, ln in zip(shuffled_labels_seq, episode_lens):
            shuffled_full.extend([lab] * ln)
        shuffled_quadrant = pd.Series(shuffled_full[:len(quadrant)], index=quadrant.index)
        permuted_means[p] = _proxy_regime_allocation_returns(shuffled_quadrant, rets, cfg)

    permuted_means = permuted_means[~np.isnan(permuted_means)]
    p_value = (np.sum(permuted_means >= observed_proxy_mean) + 1) / (len(permuted_means) + 1)
    return {
        "observed_proxy_mean_monthly_return": observed_proxy_mean,
        "observed_strategy_mean_monthly_return": observed_strategy_mean,
        "permuted_mean": np.mean(permuted_means), "permuted_std": np.std(permuted_means),
        "p_value": p_value,
        "n_episodes": len(episode_labels),
    }

def subperiod_rank_stability(prices: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config):
    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets, quadrant = rets.loc[idx], regime_panel.loc[idx, "quadrant"]
    thirds = np.array_split(idx, 3)

    results = {}
    for regime in quadrant.dropna().unique():
        rankings = []
        for third in thirds:
            sub = rets.loc[third][quadrant.loc[third] == regime]
            if len(sub) < 6:
                rankings.append(None)
                continue
            sharpe = (sub.mean() / sub.std().replace(0, np.nan)).dropna()
            rankings.append(sharpe.rank(ascending=False))
        pairs = [(a, b) for a, b in [(rankings[0], rankings[1]), (rankings[1], rankings[2]),
                                       (rankings[0], rankings[2])] if a is not None and b is not None]
        rhos = []
        for a, b in pairs:
            common = a.index.intersection(b.index)
            if len(common) >= 3:
                rho, _ = spearmanr(a.loc[common], b.loc[common])
                rhos.append(rho)
        results[regime] = np.nanmean(rhos) if rhos else np.nan
    return results

def transaction_cost_sensitivity(prices, regime_panel, cfg: Config, bps_grid=(0, 5, 10, 20, 30)):
    results = {}
    for bps in bps_grid:
        c = Config(**{**cfg.__dict__, "txn_cost_bps": bps})
        alloc = walkforward_regime_allocation(prices, regime_panel, c)
        r = alloc["net_returns"].dropna()
        if r.std() > 0:
            sharpe = (r.mean() / r.std()) * np.sqrt(12)
        else:
            sharpe = np.nan
        results[bps] = {"ann_return": r.mean() * 12, "sharpe": sharpe}
    return pd.DataFrame(results).T

def walkforward_hit_rate(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                          selection_log: dict, cfg: Config):
    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    quadrant = regime_panel.loc[idx, "quadrant"]

    oracle_top = {}
    for regime in quadrant.dropna().unique():
        sub = rets.loc[idx][quadrant == regime]
        mu = sub.mean()
        oracle_top[regime] = set(mu.sort_values(ascending=False).head(cfg.top_n_assets).index)

    overlaps = []
    for t, selected in selection_log.items():
        if t not in quadrant.index:
            continue
        regime = quadrant.loc[t]
        if regime not in oracle_top or pd.isna(regime):
            continue
        overlap = len(set(selected) & oracle_top[regime])
        overlaps.append(overlap)
    if not overlaps:
        return {"mean_overlap": np.nan, "max_possible": cfg.top_n_assets, "n_periods": 0}
    return {"mean_overlap": np.mean(overlaps), "max_possible": cfg.top_n_assets,
            "n_periods": len(overlaps)}

def jennrich_correlation_test(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                               regime_a: str, regime_b: str, axis: str = "quadrant"):
    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    quadrant = regime_panel.loc[idx, axis]
    Ra = rets.loc[idx][quadrant == regime_a].dropna()
    Rb = rets.loc[idx][quadrant == regime_b].dropna()
    if len(Ra) < 10 or len(Rb) < 10:
        return {"chi2": np.nan, "p_value": np.nan, "note": "insufficient sample in one regime"}
    n1, n2 = len(Ra), len(Rb)
    R1, R2 = Ra.corr().values, Rb.corr().values
    p = R1.shape[0]
    Rbar = (n1 * R1 + n2 * R2) / (n1 + n2)
    try:
        Rbar_inv = np.linalg.inv(Rbar)
    except np.linalg.LinAlgError:
        return {"chi2": np.nan, "p_value": np.nan, "note": "singular average correlation matrix"}
    diff = R1 - R2
    chi2_stat = 0.5 * n1 * n2 / (n1 + n2) * np.trace(Rbar_inv @ diff @ Rbar_inv @ diff)
    dof = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    return {"chi2": chi2_stat, "dof": dof, "p_value": p_value}

def newey_west_regime_tstat(strategy_returns: pd.Series, regime_panel: pd.DataFrame, maxlags=6):
    idx = strategy_returns.index.intersection(regime_panel.index)
    rets, quadrant = strategy_returns.loc[idx], regime_panel.loc[idx, "quadrant"]
    results = {}
    for regime in quadrant.dropna().unique():
        y = rets[quadrant == regime].dropna()
        if len(y) < 8:
            continue
        X = np.ones(len(y))
        model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(y) // 3 + 1)})
        results[regime] = {"mean_monthly": model.params[0], "hac_tstat": model.tvalues[0],
                            "hac_pvalue": model.pvalues[0], "n": len(y)}
    return pd.DataFrame(results).T

def holm_bonferroni(pvals: dict, alpha: float = 0.05) -> dict:

    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = {}
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        running_max = max(running_max, min((m - i) * p, 1.0))
        adjusted[name] = {"p_raw": p, "p_holm": running_max,
                           "significant_at_alpha": running_max < alpha}
    return adjusted

def run_defensibility_suite(prices, regime_panel, alloc, cfg: Config):
    bench_rets = prices[cfg.benchmark_ticker].pct_change()
    suite = {}
    suite["bootstrap_sharpe_ci_strategy"] = moving_block_bootstrap_sharpe_ci(alloc["net_returns"], cfg)
    suite["bootstrap_sharpe_ci_benchmark"] = moving_block_bootstrap_sharpe_ci(bench_rets, cfg)
    suite["paired_bootstrap_vs_benchmark"] = paired_block_bootstrap_test(
        alloc["net_returns"], bench_rets, cfg)
    suite["regime_permutation_test"] = regime_label_permutation_test(
        alloc["net_returns"], regime_panel, prices, cfg)
    suite["subperiod_rank_stability"] = subperiod_rank_stability(prices, regime_panel, cfg)
    suite["txn_cost_sensitivity"] = transaction_cost_sensitivity(prices, regime_panel, cfg)
    suite["walkforward_hit_rate"] = walkforward_hit_rate(
        prices, regime_panel, alloc["selection_log"], cfg)
    quadrants_present = regime_panel["quadrant"].dropna().unique().tolist()
    if "Stable Growth" in quadrants_present and "Contraction / Disinflation" in quadrants_present:
        suite["jennrich_stablegrowth_vs_contraction"] = jennrich_correlation_test(
            prices, regime_panel, "Stable Growth", "Contraction / Disinflation")
    suite["newey_west_regime_tstats"] = newey_west_regime_tstat(alloc["net_returns"], regime_panel)

    pvals = {
        "paired_bootstrap_vs_benchmark": suite["paired_bootstrap_vs_benchmark"]["p_value_one_sided"],
        "regime_permutation_test": suite["regime_permutation_test"]["p_value"],
    }
    if "jennrich_stablegrowth_vs_contraction" in suite:
        jp = suite["jennrich_stablegrowth_vs_contraction"]["p_value"]
        if not np.isnan(jp):
            pvals["jennrich_stablegrowth_vs_contraction"] = jp
    for regime, row in suite["newey_west_regime_tstats"].iterrows():
        if not np.isnan(row["hac_pvalue"]):
            pvals[f"newey_west_{regime}"] = row["hac_pvalue"]
    suite["holm_bonferroni"] = holm_bonferroni(pvals)

    return suite

# VALIDATION & ECONOMIC REALITY CHECK ENGINE:
# DASHBOARD A: CLASSIFICATION QUALITY
def empirical_persistence(labels: pd.Series) -> dict:
    labels = labels.dropna()
    out = {}
    for lab in labels.unique():
        mask = labels.shift() == lab
        if mask.sum() == 0:
            out[lab] = np.nan
            continue
        out[lab] = float((labels[mask] == lab).mean())
    return out


def short_episode_rate(confirmed_regime: pd.Series, cfg: Config) -> dict:

    segments = _regime_segments(confirmed_regime.dropna())
    if len(segments) <= 1:
        return {"n_transitions": 0, "n_likely_short": 0, "short_episode_rate": np.nan,
                "short_episodes": []}
    transitions = segments[1:]
    durations = [len(confirmed_regime.loc[s:e]) for s, e, _ in transitions]
    n_short = sum(1 for d in durations if d < cfg.validation_false_transition_min_months)
    short_eps = [(s, e, lab, len(confirmed_regime.loc[s:e])) for s, e, lab in transitions
                 if len(confirmed_regime.loc[s:e]) < cfg.validation_false_transition_min_months]
    return {
        "n_transitions": len(transitions), "n_likely_short": n_short,
        "short_episode_rate": n_short / len(transitions),
        "short_episodes": short_eps,
    }


def regime_reversion_rate(confirmed_regime: pd.Series, cfg: Config) -> dict:

    segments = _regime_segments(confirmed_regime.dropna())
    if len(segments) < 3:
        return {"n_candidate_transitions": 0, "n_reversions": 0, "reversion_rate": np.nan,
                "reverted_episodes": []}
    n_candidates = len(segments) - 2
    n_reversions, reverted = 0, []
    for k in range(n_candidates):
        prev_lab = segments[k][2]
        mid_start, mid_end, mid_lab = segments[k + 1]
        next_lab = segments[k + 2][2]
        mid_duration = len(confirmed_regime.loc[mid_start:mid_end])
        if next_lab == prev_lab and mid_duration <= cfg.validation_false_transition_min_months:
            n_reversions += 1
            reverted.append((mid_start, mid_end, mid_lab, mid_duration))
    return {
        "n_candidate_transitions": n_candidates, "n_reversions": n_reversions,
        "reversion_rate": n_reversions / n_candidates if n_candidates else np.nan,
        "reverted_episodes": reverted,
    }


def nber_recession_onsets(recession: pd.Series) -> list:
    """0->1 transitions in USREC -- the reference event dates for detection-lag scoring."""
    r = recession.dropna()
    return list(r.index[(r == 1) & (r.shift(1) == 0)])


def detection_lag(event_dates: list, signal: pd.Series, threshold: float, window: int) -> pd.DataFrame:

    s = signal.dropna().sort_index()
    prev = s.shift(1)
    crossings = s.index[(s >= threshold) & (prev < threshold)]
    rows = []
    for ev in event_dates:
        win_start, win_end = ev - pd.DateOffset(months=window), ev + pd.DateOffset(months=window)
        win_crossings = crossings[(crossings >= win_start) & (crossings <= win_end)]
        if len(win_crossings) == 0:
            rows.append({"event_date": ev, "first_cross_date": None, "lag_months": np.nan})
            continue
        first = win_crossings.min()
        lag = (first.year - ev.year) * 12 + (first.month - ev.month)
        rows.append({"event_date": ev, "first_cross_date": first, "lag_months": lag})
    return pd.DataFrame(rows)

def model_agreement_by_status(regime_panel: pd.DataFrame, axis: str) -> dict:

    agree = regime_panel[f"{axis}_model_agreement"]
    status = regime_panel[f"{axis}_status"]
    out = {"overall": float(agree.mean())}
    for s in status.unique():
        out[s] = float(agree[status == s].mean())
    return out

def _illustrative_dates_per_state(confirmed_regime: pd.Series) -> list:

    labels = confirmed_regime.dropna()
    segments = _regime_segments(labels)
    best_by_state = {}
    for s, e, lab in segments:
        d = len(labels.loc[s:e])
        if lab not in best_by_state or d > best_by_state[lab][2]:
            best_by_state[lab] = (s, e, d)
    dates = []
    for lab, (s, e, d) in best_by_state.items():
        seg_idx = labels.loc[s:e].index
        dates.append(seg_idx[len(seg_idx) // 2])
    if len(labels):
        dates.append(labels.index[-1])
    return sorted(set(dates))

def bootstrap_state_stability_at_dates(features: pd.DataFrame, axis_cols: list, n_states: int,
                                        relabel_fn, cfg: Config, canonical_labels: list,
                                        confirmed_regime: pd.Series, target_dates: list,
                                        n_boot: Optional[int] = None) -> dict:

    X_full = features[axis_cols].dropna()
    n_boot = n_boot or cfg.validation_state_stability_n_boot
    block = cfg.block_size_months
    covariance_type = "full" if n_states == 4 else "diag"
    out = {}
    for target in target_dates:
        if target not in X_full.index:
            continue
        train_end = X_full.index.get_loc(target) + 1
        if train_end < cfg.min_regime_history:
            continue
        X_train = X_full.iloc[:train_end].values
        x_target = X_train[-1:]
        assigned = confirmed_regime.loc[target] if target in confirmed_regime.index else None
        n = len(X_train)
        n_blocks = int(np.ceil(n / block))
        rng = np.random.default_rng(cfg.random_state + int(target.strftime("%Y%m%d")) % 100000)
        matches, n_valid = 0, 0
        for _ in range(n_boot):
            starts = rng.integers(0, max(n - block, 1), size=n_blocks)
            sample = np.concatenate([X_train[s:s + block] for s in starts])[:n]
            try:
                scaler = StandardScaler().fit(sample)
                sample_s = scaler.transform(sample)
                x_s = scaler.transform(x_target)
                gmm = GaussianMixture(n_init=2, n_components=n_states,
                                       covariance_type=covariance_type, random_state=cfg.random_state)
                gmm.fit(sample_s)
                label_map, _ = relabel_fn(gmm.means_, cfg)
                draw_label = label_map[gmm.predict(x_s)[0]]
                n_valid += 1
                if draw_label == assigned:
                    matches += 1
            except Exception:
                continue
        out[target] = {"assigned_state": assigned,
                        "stability_pct": (matches / n_valid) if n_valid else np.nan,
                        "n_valid_draws": n_valid}
    return out

# DASHBOARD B: FORECAST QUALITY
def multiclass_forecast_score(prob_panel: pd.DataFrame, confirmed_regime: pd.Series,
                               canonical_labels: list, h: int) -> dict:

    realized = confirmed_regime.reindex(prob_panel.index).shift(-h)
    df = prob_panel.join(realized.rename("y"), how="inner").dropna()
    if df.empty:
        return {"brier": np.nan, "log_loss": np.nan, "n": 0}
    p = df[canonical_labels].values.astype(float)
    y_idx = np.array([canonical_labels.index(v) for v in df["y"]])
    y_onehot = np.zeros_like(p)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0
    brier = float(np.mean(np.sum((p - y_onehot) ** 2, axis=1)))
    p_true = np.clip(p[np.arange(len(y_idx)), y_idx], 1e-9, 1.0)
    ll = float(-np.mean(np.log(p_true)))
    return {"brier": brier, "log_loss": ll, "n": len(df)}

def calibration_slope_intercept(prob_series: pd.Series, realized: pd.Series) -> dict:

    df = pd.DataFrame({"p": prob_series, "y": realized}).dropna()
    df = df[(df["p"] > 0.001) & (df["p"] < 0.999)]
    if len(df) < 20 or df["y"].nunique() < 2:
        return {"intercept": np.nan, "slope": np.nan, "n": len(df)}
    logit_p = np.log(df["p"] / (1 - df["p"]))
    X = sm.add_constant(logit_p.values)
    try:
        model = sm.Logit(df["y"].values, X).fit(disp=0)
        return {"intercept": float(model.params[0]), "slope": float(model.params[1]), "n": len(df)}
    except Exception:
        return {"intercept": np.nan, "slope": np.nan, "n": len(df)}

def transition_hit_rate_validation(change_risk: pd.Series, confirmed_regime: pd.Series, h: int) -> dict:

    idx = change_risk.index
    cur = confirmed_regime.reindex(idx)
    fut = confirmed_regime.reindex(idx).shift(-h)
    df = pd.DataFrame({"risk": change_risk, "cur": cur, "fut": fut}).dropna(subset=["risk", "cur", "fut"])
    if df.empty:
        return {"auc": np.nan, "n": 0, "base_rate": np.nan}
    changed = (df["cur"] != df["fut"]).astype(float)
    if changed.nunique() < 2:
        return {"auc": np.nan, "n": len(df), "base_rate": float(changed.mean())}
    return {"auc": float(roc_auc_score(changed, df["risk"])), "n": len(df), "base_rate": float(changed.mean())}

def destination_accuracy(conditional_destination: pd.DataFrame, confirmed_regime: pd.Series, h: int) -> dict:

    idx = conditional_destination.index
    cur = confirmed_regime.reindex(idx)
    fut = confirmed_regime.reindex(idx).shift(-h)
    n_correct, n_total = 0, 0
    for t in idx:
        if pd.isna(cur.loc[t]) or pd.isna(fut.loc[t]) or cur.loc[t] == fut.loc[t]:
            continue
        row = conditional_destination.loc[t].dropna()
        if row.empty:
            continue
        n_total += 1
        if row.idxmax() == fut.loc[t]:
            n_correct += 1
    return {"n_correct": n_correct, "n_total": n_total, "accuracy": (n_correct / n_total) if n_total else np.nan}

def recession_classification_metrics(prob_series: pd.Series, recession: pd.Series,
                                      h: int, threshold: float = 0.5) -> dict:
    
    realized = recession.reindex(prob_series.index).shift(-h)
    df = pd.DataFrame({"p": prob_series, "y": realized}).dropna()
    if df.empty or df["y"].nunique() < 2:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan,
                "roc_auc": np.nan, "pr_auc": np.nan, "n": len(df), "n_positive": 0}
    pred = (df["p"] >= threshold).astype(int)
    return {
        "precision": float(precision_score(df["y"], pred, zero_division=0)),
        "recall": float(recall_score(df["y"], pred, zero_division=0)),
        "f1": float(f1_score(df["y"], pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(df["y"], df["p"])),
        "pr_auc": float(average_precision_score(df["y"], df["p"])),
        "n": len(df), "n_positive": int(df["y"].sum()),
    }

# DASHBOARD C: ROBUSTNESS
def parameter_stability_report(fit: dict, canonical_labels: list) -> dict:

    transmats = fit.get("transition_matrices", [])
    means_raw = fit.get("hmm_means_raw_by_refit", [])
    label_maps = fit.get("hmm_label_maps", [])
    canon_means, canon_transmats = [], []
    for means, tm, lm in zip(means_raw, transmats, label_maps):
        inv_map = {v: k for k, v in lm.items()}
        if not all(lab in inv_map for lab in canonical_labels):
            continue
        perm = [inv_map[lab] for lab in canonical_labels]
        canon_means.append(means[perm])
        canon_transmats.append(tm[np.ix_(perm, perm)])
    out = {"centroid_drift": [], "transmat_frobenius_drift": []}
    for i in range(1, len(canon_means)):
        out["centroid_drift"].append({
            canonical_labels[k]: float(np.linalg.norm(canon_means[i][k] - canon_means[i - 1][k]))
            for k in range(len(canonical_labels))})
    for i in range(1, len(canon_transmats)):
        out["transmat_frobenius_drift"].append(float(np.linalg.norm(canon_transmats[i] - canon_transmats[i - 1])))
    return out

def leave_one_gauge_out(features: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    full = regime_panel["vol_liquidity"]
    rows = []
    for drop_col in STRESS_AXIS_GAUGES:
        cols = [c for c in STRESS_AXIS_GAUGES if c != drop_col]
        try:
            fit = fit_regime_axis_walkforward(
                features, cols, n_states=3, relabel_fn=relabel_vol_liq_states, cfg=cfg,
                label_prefix="vol_liquidity_loo", canonical_labels=list(cfg.vol_liq_order),
                covariance_type="diag")
            confirm = confirm_regime_sequence(
                fit["hmm_proba"].dropna(), fit["gmm_label"], fit["active_transmat"],
                list(cfg.vol_liq_order), cfg, axis="stress")
            reduced = confirm["confirmed_regime"]
            idx = reduced.index.intersection(full.index)
            agreement = float((reduced.loc[idx] == full.loc[idx]).mean())
            occ_shift = (reduced.value_counts(normalize=True) -
                         full.value_counts(normalize=True)).abs().sum() / 2
            rows.append({"gauge_removed": drop_col, "agreement_with_full_model": agreement,
                         "occupancy_shift_l1": occ_shift, "n": len(idx)})
        except Exception as e:
            rows.append({"gauge_removed": drop_col, "agreement_with_full_model": np.nan,
                         "occupancy_shift_l1": np.nan, "n": 0, "error": str(e)})
    return pd.DataFrame(rows).set_index("gauge_removed")

def rolling_vs_expanding_comparison(features: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config) -> dict:

    full = regime_panel["vol_liquidity"]
    fit = fit_regime_axis_walkforward(
        features, STRESS_AXIS_GAUGES, n_states=3, relabel_fn=relabel_vol_liq_states, cfg=cfg,
        label_prefix="vol_liquidity_rolling", canonical_labels=list(cfg.vol_liq_order),
        covariance_type="diag", rolling_window_months=cfg.validation_rolling_window_months)
    confirm = confirm_regime_sequence(
        fit["hmm_proba"].dropna(), fit["gmm_label"], fit["active_transmat"],
        list(cfg.vol_liq_order), cfg, axis="stress")
    rolling = confirm["confirmed_regime"]
    idx = rolling.index.intersection(full.index)
    agreement = float((rolling.loc[idx] == full.loc[idx]).mean())
    occ_shift = (rolling.value_counts(normalize=True) - full.value_counts(normalize=True)).abs().sum() / 2
    return {"window_months": cfg.validation_rolling_window_months,
            "agreement_with_expanding": agreement, "occupancy_shift_l1": occ_shift, "n": len(idx)}

def structural_break_breakdown(regime_panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    bounds = [pd.Timestamp(d) for d in cfg.validation_structural_break_dates]
    edges = [regime_panel.index.min()] + bounds + [regime_panel.index.max() + pd.Timedelta(days=1)]
    rows = []
    for i in range(len(edges) - 1):
        seg = regime_panel.loc[(regime_panel.index >= edges[i]) & (regime_panel.index < edges[i + 1])]
        if seg.empty:
            continue
        rows.append({
            "period_start": edges[i].date(), "period_end": (edges[i + 1] - pd.Timedelta(days=1)).date(),
            "n_months": len(seg),
            "modal_quadrant": seg["quadrant"].mode().iloc[0] if not seg["quadrant"].empty else None,
            "modal_vol_liquidity": seg["vol_liquidity"].mode().iloc[0] if not seg["vol_liquidity"].empty else None,
            "quadrant_model_agreement": float(seg["quadrant_model_agreement"].mean()),
            "vol_liquidity_model_agreement": float(seg["vol_liquidity_model_agreement"].mean()),
        })
    return pd.DataFrame(rows)

def vintage_sensitivity_naive_vs_pit(cfg: Config) -> dict:

    if not cfg.use_live_data:
        raise ValueError("vintage_sensitivity_naive_vs_pit needs live data -- "
                          "synthetic mode has no vintage history to compare against")
    end = cfg.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    series_ids = [sid for block in cfg.indicator_blocks().values() for sid in block] + [cfg.recession_series]
    fetch_ids = [sid for sid in series_ids if sid != "TEDRATE"]
    needs_ted = "TEDRATE" in series_ids
    if needs_ted:
        fetch_ids += ["DCPF3M", "DTB3"]

    naive_cols = {}
    for sid in fetch_ids:
        raw = fetch_fred_series(sid, cfg.start_date, end)
        if raw is not None:
            naive_cols[sid] = raw 
    naive_panel = pd.DataFrame(naive_cols)
    if needs_ted and "DCPF3M" in naive_panel.columns and "DTB3" in naive_panel.columns:
        naive_panel["TEDRATE"] = (naive_panel["DCPF3M"] - naive_panel["DTB3"]).dropna()
    naive_panel = naive_panel.drop(columns=["DCPF3M", "DTB3"], errors="ignore")
    naive_macro = naive_panel.resample("ME").last().loc[cfg.start_date:end].ffill()

    naive_prices = fetch_yahoo_prices(cfg.asset_universe, cfg.start_date, end)
    naive_prices = naive_prices.resample("ME").last().ffill()
    common_idx = naive_macro.index.intersection(naive_prices.index)
    naive_macro, naive_prices = naive_macro.loc[common_idx], naive_prices.loc[common_idx]

    naive_features, naive_rf_features, _ = build_feature_panel(naive_macro, naive_prices, cfg)
    naive_regime_panel, _, _, _ = classify_regimes(naive_features, naive_rf_features, naive_macro, cfg)

    pit_macro, pit_prices, _ = load_all_data(cfg)
    pit_features, pit_rf_features, _ = build_feature_panel(pit_macro, pit_prices, cfg)
    pit_regime_panel, _, _, _ = classify_regimes(pit_features, pit_rf_features, pit_macro, cfg)

    idx = naive_regime_panel.index.intersection(pit_regime_panel.index)
    return {
        "n_common_months": len(idx),
        "quadrant_agreement_naive_vs_pit": float((naive_regime_panel.loc[idx, "quadrant"] ==
                                                    pit_regime_panel.loc[idx, "quadrant"]).mean()),
        "vol_liquidity_agreement_naive_vs_pit": float((naive_regime_panel.loc[idx, "vol_liquidity"] ==
                                                         pit_regime_panel.loc[idx, "vol_liquidity"]).mean()),
        "naive_regime_panel": naive_regime_panel, "pit_regime_panel": pit_regime_panel,
    }

# DASHBOARD D: ECONOMIC REALITY
def newey_west_conditional_mean(series: pd.Series, regime_panel: pd.DataFrame, axis: str, maxlags=6) -> pd.DataFrame:

    idx = series.index.intersection(regime_panel.index)
    y_full, labels = series.loc[idx], regime_panel.loc[idx, axis]
    results = {}
    for regime in labels.dropna().unique():
        y = y_full[labels == regime].dropna()
        if len(y) < 8:
            continue
        X = np.ones(len(y))
        model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(y) // 3 + 1)})
        results[regime] = {"mean": model.params[0], "hac_tstat": model.tvalues[0],
                            "hac_pvalue": model.pvalues[0], "n": len(y)}
    return pd.DataFrame(results).T

def held_out_economic_reality_check(features: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config) -> dict:

    held_out_cols = [c for c in ("labour_z", "housing_z", "rates_z") if c in features.columns]
    out = {}
    for col in held_out_cols:
        out[col] = {
            "quadrant": newey_west_conditional_mean(features[col], regime_panel, "quadrant"),
            "vol_liquidity": newey_west_conditional_mean(features[col], regime_panel, "vol_liquidity"),
        }
    orderings = {}
    if "labour_z" in out:
        q = out["labour_z"]["quadrant"]
        if "Stable Growth" in q.index and "Contraction / Disinflation" in q.index:
            orderings["labour_z: E[.|StableGrowth] > E[.|Contraction]"] = bool(
                q.loc["Stable Growth", "mean"] > q.loc["Contraction / Disinflation", "mean"])
    if "housing_z" in out:
        q = out["housing_z"]["quadrant"]
        if "Overheating" in q.index and "Contraction / Disinflation" in q.index:
            orderings["housing_z: E[.|Overheating] > E[.|Contraction]"] = bool(
                q.loc["Overheating", "mean"] > q.loc["Contraction / Disinflation", "mean"])
    if "rates_z" in out:
        v = out["rates_z"]["vol_liquidity"]
        if "Crisis" in v.index and "Calm" in v.index:
            orderings["rates_z: Crisis vs. Calm means (flight-to-quality direction, no fixed sign prediction)"] = {
                "Crisis_mean": float(v.loc["Crisis", "mean"]), "Calm_mean": float(v.loc["Calm", "mean"])}
    out["_orderings"] = orderings
    return out

NAMED_EVENT_WINDOWS = {
    "2008 GFC": ("2008-01-01", "2009-06-30"),
    "2011 Eurozone stress": ("2011-06-01", "2012-06-30"),
    "2020 COVID shock": ("2020-02-01", "2020-06-30"),
    "2022-23 inflation shock": ("2022-01-01", "2023-06-30"),
}

def event_study_snapshot(regime_panel: pd.DataFrame, event_windows: dict = NAMED_EVENT_WINDOWS) -> dict:

    out = {}
    for name, (start, end) in event_windows.items():
        seg = regime_panel.loc[start:end]
        if seg.empty:
            out[name] = {"available": False}
            continue
        row = {"available": True, "n_months": len(seg),
               "quadrant_path": seg["quadrant"].tolist(),
               "vol_liquidity_path": seg["vol_liquidity"].tolist()}
        contraction_col = next((c for c in seg.columns if c.startswith("quadrant_proba_") and "Contraction" in c), None)
        crisis_col = next((c for c in seg.columns if c.startswith("vol_liquidity_proba_") and "Crisis" in c), None)
        if contraction_col:
            row["max_contraction_probability"] = float(seg[contraction_col].max())
        if crisis_col:
            row["max_crisis_probability"] = float(seg[crisis_col].max())
        out[name] = row
    return out

# STEP 9 ORCHESTRATION
def build_validation_engine(features: pd.DataFrame, regime_panel: pd.DataFrame,
                             quad_fit: dict, vol_liq_fit: dict, forecast: dict,
                             rf_features: pd.DataFrame, macro: pd.DataFrame,
                             suite: dict, cfg: Config) -> dict:
    """Step 9 orchestration -- Dashboards A-D. See the module-level comment
    above for what's deliberately excluded and why."""
    quadrant_canonical = list(cfg.quadrant_labels.values())
    vol_liq_canonical = list(cfg.vol_liq_order)
    result = {"A": {}, "B": {}, "C": {}, "D": {}}

    for axis, canonical_labels, fit in (("quadrant", quadrant_canonical, quad_fit),
                                          ("vol_liquidity", vol_liq_canonical, vol_liq_fit)):
        confirmed = regime_panel[axis]
        last_transmat = fit["last_transition_matrix"]
        last_label_map = fit["last_label_map"] or {}
        fitted_pers = {}
        if last_transmat is not None and last_label_map:
            inv_map = {v: k for k, v in last_label_map.items()}
            for lab in canonical_labels:
                if lab in inv_map:
                    fitted_pers[lab] = float(last_transmat[inv_map[lab], inv_map[lab]])

        relabel_fn = relabel_quadrant_states if axis == "quadrant" else relabel_vol_liq_states
        n_states = 4 if axis == "quadrant" else 3
        axis_cols = ECONOMIC_AXIS_GAUGES if axis == "quadrant" else STRESS_AXIS_GAUGES
        target_dates = _illustrative_dates_per_state(confirmed)

        result["A"][axis] = {
            "occupancy": regime_occupancy_stats(confirmed, cfg),
            "empirical_persistence": empirical_persistence(confirmed),
            "fitted_persistence_most_recent_refit": fitted_pers,
            "short_episodes": short_episode_rate(confirmed, cfg),
            "reversions": regime_reversion_rate(confirmed, cfg),
            "model_agreement_by_status": model_agreement_by_status(regime_panel, axis),
            "bootstrap_state_stability": bootstrap_state_stability_at_dates(
                features, axis_cols, n_states, relabel_fn, cfg, canonical_labels, confirmed, target_dates),
        }

    if cfg.recession_series in macro.columns:
        onsets = nber_recession_onsets(macro[cfg.recession_series])
        recession_prob_3m = forecast.get("recession", {}).get(3)
        if recession_prob_3m is not None and onsets:
            result["A"]["recession_detection_lag"] = detection_lag(
                onsets, recession_prob_3m["level"], cfg.validation_recession_detection_threshold,
                cfg.validation_detection_lag_window)
        contraction_label = next((v for v in cfg.quadrant_labels.values() if "Contraction" in v), None)
        contraction_col = f"quadrant_proba_{contraction_label}" if contraction_label else None
        if contraction_col and contraction_col in regime_panel.columns and onsets:
            result["A"]["contraction_probability_detection_lag"] = detection_lag(
                onsets, regime_panel[contraction_col], cfg.validation_recession_detection_threshold,
                cfg.validation_detection_lag_window)

    for axis, canonical_labels in (("quadrant", quadrant_canonical), ("vol_liquidity", vol_liq_canonical)):
        fdata = forecast.get(axis, {})
        confirmed = regime_panel[axis]
        axis_b = {}
        for h, panel in fdata.get("structural", {}).items():
            axis_b[f"structural_h{h}"] = multiclass_forecast_score(panel, confirmed, canonical_labels, h)
            risk_h = fdata.get("transition_risk", {}).get(h)
            if risk_h is not None:
                axis_b[f"transition_hit_rate_h{h}"] = transition_hit_rate_validation(risk_h["change_risk"], confirmed, h)
                axis_b[f"destination_accuracy_h{h}"] = destination_accuracy(risk_h["conditional_destination"], confirmed, h)
        for h, panel in fdata.get("feature_conditioned", {}).items():
            axis_b[f"feature_conditioned_h{h}"] = multiclass_forecast_score(panel, confirmed, canonical_labels, h)
        horizons = fdata.get("horizons", ())
        if horizons:
            h0 = horizons[0]
            panel0 = fdata.get("structural", {}).get(h0)
            if panel0 is not None:
                realized0 = confirmed.reindex(panel0.index).shift(-h0)
                axis_b[f"calibration_h{h0}_by_state"] = {
                    lab: forecast_calibration(panel0[lab], (realized0 == lab).astype(float),
                                               bins=cfg.validation_calibration_bins)
                    for lab in canonical_labels if lab in panel0.columns}
        result["B"][axis] = axis_b

    if cfg.recession_series in macro.columns and forecast.get("recession"):
        recession_b = {}
        for h in (3, 6):
            p = forecast["recession"].get(h)
            if p is None:
                continue
            level = p["level"]
            recession_b[f"h{h}"] = recession_classification_metrics(level, macro[cfg.recession_series], h)
            realized_h = macro[cfg.recession_series].reindex(level.index).shift(-h)
            recession_b[f"h{h}_calibration_slope_intercept"] = calibration_slope_intercept(level, realized_h)
        result["B"]["recession"] = recession_b

    result["C"]["parameter_stability_quadrant"] = parameter_stability_report(quad_fit, quadrant_canonical)
    result["C"]["parameter_stability_vol_liquidity"] = parameter_stability_report(vol_liq_fit, vol_liq_canonical)
    result["C"]["feature_stability_note"] = (
        "Reuses Section 4.J loading_stability / Step 8 evr_history already in gauge_info -- "
        "see FACTOR (PCA) UNCERTAINTY in the tearsheet; not recomputed here.")
    result["C"]["state_count_sensitivity_quadrant"] = compare_state_counts(
        features, ECONOMIC_AXIS_GAUGES, cfg, k_range=(2, 3, 4, 5, 6))
    result["C"]["state_count_sensitivity_vol_liquidity"] = compare_state_counts(
        features, STRESS_AXIS_GAUGES, cfg, k_range=(2, 3, 4, 5))
    result["C"]["structural_break_breakdown"] = structural_break_breakdown(regime_panel, cfg)
    result["C"]["leave_one_gauge_out_stress"] = leave_one_gauge_out(features, regime_panel, cfg)
    result["C"]["rolling_vs_expanding_stress"] = rolling_vs_expanding_comparison(features, regime_panel, cfg)

    # ---- Dashboard D: Economic Reality ----
    result["D"]["held_out_economic_reality"] = held_out_economic_reality_check(features, regime_panel, cfg)
    result["D"]["event_studies"] = event_study_snapshot(regime_panel)
    if "jennrich_stablegrowth_vs_contraction" in suite:
        result["D"]["cross_asset_jennrich"] = suite["jennrich_stablegrowth_vs_contraction"]
    result["D"]["nber_recession_reference"] = {
        "detection_lag": result["A"].get("recession_detection_lag"),
        "classification_metrics": result["B"].get("recession", {}),
    }

    return result

# 10. CROSS-ASSET REGIME MAPPING
ASSET_SLEEVES = {
    "SPY": "Equities", "IWM": "Equities", "EFA": "Equities", "EEM": "Equities", "QQQ": "Equities",
    "TLT": "Rates", "IEF": "Rates", "SHY": "Rates",
    "LQD": "Credit", "HYG": "Credit",
    "GLD": "Gold",
    "DBC": "Commodities", "USO": "Commodities",
    "UUP": "FX",
}


def _downside_deviation(r: pd.Series, target: float = 0.0) -> float:
    downside = r[r < target] - target
    return downside.std() * np.sqrt(12) if len(downside) > 1 else np.nan


def _sortino_ratio(r: pd.Series, target: float = 0.0) -> float:
    dd = _downside_deviation(r, target)
    if not dd or np.isnan(dd) or dd == 0:
        return np.nan
    return (r.mean() * 12 - target * 12) / dd


def _max_drawdown_concat(r: pd.Series) -> float:
    r = r.dropna()
    if r.empty:
        return np.nan
    curve = (1 + r).cumprod()
    return (curve / curve.cummax() - 1).min()


def _max_drawdown_by_episode(asset_returns: pd.Series, labels: pd.Series, state: str) -> dict:

    segments = _regime_segments(labels.dropna())
    episode_dd = []
    for s, e, lab in segments:
        if lab != state:
            continue
        r = asset_returns.loc[s:e].dropna()
        if len(r) < 2:
            continue
        episode_dd.append(_max_drawdown_concat(r))
    if not episode_dd:
        return {"worst_episode_drawdown": np.nan, "mean_episode_drawdown": np.nan, "n_episodes": 0}
    return {"worst_episode_drawdown": float(min(episode_dd)),
            "mean_episode_drawdown": float(np.mean(episode_dd)),
            "n_episodes": len(episode_dd)}

def regime_conditioned_asset_profile(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                                      axis: str, cfg: Config) -> pd.DataFrame:

    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    labels = regime_panel.loc[idx, axis]

    rows = []
    for state in labels.dropna().unique():
        sub = rets[labels == state]
        for asset in prices.columns:
            r = sub[asset].dropna()
            if len(r) < 2:
                continue
            ann_ret = r.mean() * 12
            ann_vol = r.std() * np.sqrt(12)
            var, cvar = historical_var_cvar(r, cfg.var_confidence)
            dd_episodes = _max_drawdown_by_episode(rets[asset], labels, state)
            rows.append({
                "regime": state, "asset": asset, "sleeve": ASSET_SLEEVES.get(asset, "Other"),
                "n_obs": len(r), "ann_return": ann_ret, "ann_vol": ann_vol,
                "sharpe": ann_ret / ann_vol if ann_vol else np.nan,
                "sortino": _sortino_ratio(r), "downside_deviation": _downside_deviation(r),
                "max_drawdown": dd_episodes["worst_episode_drawdown"],
                "max_drawdown_mean_episode": dd_episodes["mean_episode_drawdown"],
                "max_drawdown_n_episodes": dd_episodes["n_episodes"],
                "max_drawdown_concat_caveat": _max_drawdown_concat(r),
                f"VaR_{int(cfg.var_confidence*100)}": var,
                f"CVaR_{int(cfg.cvar_confidence*100)}": cvar,
                "hit_rate": (r > 0).mean(),
                "insufficient_sample": len(r) < cfg.crossasset_min_obs,
            })
    return pd.DataFrame(rows)

def cross_asset_correlation_structure(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                                       axis: str, cfg: Config) -> dict:

    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    labels = regime_panel.loc[idx, axis]

    eq_cols = [a for a in prices.columns if ASSET_SLEEVES.get(a) == "Equities"]
    credit_cols = [a for a in prices.columns if ASSET_SLEEVES.get(a) == "Credit"]
    rates_cols = [a for a in prices.columns if ASSET_SLEEVES.get(a) == "Rates"]

    by_state = {}
    for state in labels.dropna().unique():
        sub = rets[labels == state].dropna(how="all")
        n = len(sub)
        if n < cfg.crossasset_min_obs:
            by_state[state] = {"n_obs": n, "insufficient_sample": True}
            continue
        corr = sub.corr()
        mask = ~np.eye(len(corr), dtype=bool)
        by_state[state] = {
            "n_obs": n, "insufficient_sample": False,
            "avg_pairwise_corr": corr.values[mask].mean(),
            "corr_equities_credit": corr.loc[eq_cols, credit_cols].values.mean() if eq_cols and credit_cols else np.nan,
            "corr_equities_rates": corr.loc[eq_cols, rates_cols].values.mean() if eq_cols and rates_cols else np.nan,
        }

    states = [s for s in labels.dropna().unique()
              if by_state.get(s, {}).get("n_obs", 0) >= cfg.crossasset_min_obs]
    pairwise_jennrich = {}
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            a, b = states[i], states[j]
            pairwise_jennrich[f"{a} vs {b}"] = jennrich_correlation_test(prices, regime_panel, a, b, axis=axis)
    return {"by_state": by_state, "pairwise_jennrich": pairwise_jennrich}

def regime_conditional_beta(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                             axis: str, cfg: Config) -> pd.DataFrame:

    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    bench = rets[cfg.benchmark_ticker]
    labels = regime_panel.loc[idx, axis]

    rows = []
    for state in labels.dropna().unique():
        mask = (labels == state).values
        if mask.sum() < cfg.crossasset_min_obs:
            continue
        for asset in prices.columns:
            if asset == cfg.benchmark_ticker:
                continue
            y = rets.loc[mask, asset].dropna()
            x = bench.loc[y.index]
            if len(y) < cfg.crossasset_min_obs or x.std() == 0:
                continue
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit(
                cov_type="HAC", cov_kwds={"maxlags": min(cfg.crossasset_significance_maxlags, len(y) // 3 + 1)})
            rows.append({"regime": state, "asset": asset, "sleeve": ASSET_SLEEVES.get(asset, "Other"),
                         "beta": model.params[1], "hac_pvalue": model.pvalues[1], "n": len(y)})
    return pd.DataFrame(rows)

def joint_axis_sleeve_matrix(prices: pd.DataFrame, regime_panel: pd.DataFrame, cfg: Config) -> pd.DataFrame:

    rets = prices.pct_change()
    sleeve_rets = rets.T.groupby(lambda t: ASSET_SLEEVES.get(t, "Other")).mean().T
    idx = sleeve_rets.index.intersection(regime_panel.index)
    sleeve_rets = sleeve_rets.loc[idx]
    quadrant = regime_panel.loc[idx, "quadrant"]
    vol_liq = regime_panel.loc[idx, "vol_liquidity"]

    rows = []
    for q in quadrant.dropna().unique():
        for v in vol_liq.dropna().unique():
            mask = ((quadrant == q) & (vol_liq == v)).values
            n = int(mask.sum())
            for sleeve in sleeve_rets.columns:
                r = sleeve_rets.loc[mask, sleeve].dropna()
                rows.append({
                    "quadrant": q, "vol_liquidity": v, "sleeve": sleeve, "n_obs": n,
                    "ann_return": r.mean() * 12 if n >= cfg.crossasset_min_joint_obs else np.nan,
                    "insufficient_sample": n < cfg.crossasset_min_joint_obs,
                })
    return pd.DataFrame(rows)

def forward_conditional_returns(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                                 axis: str, horizons: tuple, cfg: Config) -> dict:

    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    labels = regime_panel.loc[idx, axis]
    cum = (1 + rets).cumprod()

    out = {}
    for h in horizons:
        fwd = cum.shift(-h) / cum - 1
        rows = []
        for state in labels.dropna().unique():
            mask = (labels == state).values
            for asset in prices.columns:
                r = fwd.loc[mask, asset].dropna()
                if len(r) < cfg.crossasset_min_obs:
                    continue
                rows.append({"regime": state, "asset": asset, "sleeve": ASSET_SLEEVES.get(asset, "Other"),
                             "n_obs": len(r), "mean_fwd_return": r.mean(), "vol_fwd_return": r.std(),
                             "sharpe_like": r.mean() / r.std() if r.std() else np.nan})
        out[h] = pd.DataFrame(rows)
    return out


def _posterior_weighted_mean(post: pd.Series, mu: pd.DataFrame, min_coverage: float = 0.5) -> pd.Series:

    out = pd.Series(index=mu.columns, dtype=float)
    for asset in mu.columns:
        valid = mu[asset].dropna()
        if valid.empty:
            out[asset] = np.nan
            continue
        w = post.reindex(valid.index).fillna(0.0)
        coverage = float(w.sum())
        out[asset] = float((w @ valid) / coverage) if coverage >= min_coverage else np.nan
    return out


def probability_weighted_outlook(prices: pd.DataFrame, regime_panel: pd.DataFrame, forecast: dict,
                                  axis: str, canonical_labels: list, horizons: tuple,
                                  fwd_tables: dict, cfg: Config) -> pd.DataFrame:

    latest = regime_panel.index[-1]
    post_cols = [f"{axis}_proba_{lab}" for lab in canonical_labels]
    if not all(c in regime_panel.columns for c in post_cols):
        return pd.DataFrame()
    current_post = regime_panel.loc[latest, post_cols]
    current_post.index = canonical_labels

    out = {}
    h1 = horizons[0]
    if h1 in fwd_tables and not fwd_tables[h1].empty:
        mu1 = fwd_tables[h1].pivot(index="regime", columns="asset", values="mean_fwd_return").reindex(canonical_labels)
        out[f"current_posterior_h{h1}"] = _posterior_weighted_mean(current_post, mu1)

    struct = (forecast or {}).get(axis, {}).get("structural", {})
    for h, panel in struct.items():
        if h not in fwd_tables or fwd_tables[h].empty or latest not in panel.index:
            continue
        muh = fwd_tables[h].pivot(index="regime", columns="asset", values="mean_fwd_return").reindex(canonical_labels)
        p_future = panel.loc[latest].reindex(canonical_labels).fillna(0.0)
        out[f"transition_aware_h{h}"] = _posterior_weighted_mean(p_future, muh)

    return pd.DataFrame(out).T if out else pd.DataFrame()


def regime_pairwise_mean_diff_test(series: pd.Series, labels: pd.Series, regime_a: str,
                                    regime_b: str, maxlags: int = 6) -> dict:

    idx = series.index.intersection(labels.index)
    y_all, lab_all = series.loc[idx].dropna(), labels.loc[idx]
    mask = lab_all.isin([regime_a, regime_b])
    lab = lab_all.loc[mask]
    y = y_all.reindex(lab.index).dropna()
    lab = lab.loc[y.index]
    if len(y) < 12 or lab.nunique() < 2:
        return {"mean_a": np.nan, "mean_b": np.nan, "diff": np.nan,
                "hac_pvalue": np.nan, "n": len(y)}
    dummy = (lab == regime_a).astype(float).values
    X = sm.add_constant(dummy)
    model = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(maxlags, len(y) // 3 + 1)})
    return {"mean_b": model.params[0], "mean_a": model.params[0] + model.params[1],
            "diff": model.params[1], "hac_tstat": model.tvalues[1],
            "hac_pvalue": model.pvalues[1], "n": len(y)}


def regime_information_content(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                                axis: str, cfg: Config) -> tuple:

    rets = prices.pct_change()
    idx = rets.index.intersection(regime_panel.index)
    rets = rets.loc[idx]
    labels = regime_panel.loc[idx, axis]
    states = [s for s in labels.dropna().unique() if (labels == s).sum() >= cfg.crossasset_min_obs]
    pairs = [(states[i], states[j]) for i in range(len(states)) for j in range(i + 1, len(states))]

    rows, pvals = [], {}
    for asset in prices.columns:
        r = rets[asset].dropna()
        if r.empty:
            continue
        _, uncond_cvar = historical_var_cvar(r, cfg.cvar_confidence)
        best_pair, best_gap, best_test = None, -1.0, None
        for a, b in pairs:
            test = regime_pairwise_mean_diff_test(r, labels, a, b, cfg.crossasset_significance_maxlags)
            if np.isnan(test["diff"]):
                continue
            pvals[f"{asset}::{a} vs {b}"] = test["hac_pvalue"]
            if abs(test["diff"]) > best_gap:
                best_gap, best_pair, best_test = abs(test["diff"]), (a, b), test
        rows.append({
            "asset": asset, "sleeve": ASSET_SLEEVES.get(asset, "Other"),
            "unconditional_ann_return": r.mean() * 12, "unconditional_ann_vol": r.std() * np.sqrt(12),
            "unconditional_skew": r.skew(), f"unconditional_CVaR_{int(cfg.cvar_confidence*100)}": uncond_cvar,
            "widest_regime_pair": best_pair,
            "widest_pair_mean_diff_annualized": best_gap * 12 if best_pair else np.nan,
            "widest_pair_hac_pvalue": best_test["hac_pvalue"] if best_test else np.nan,
        })
    table = pd.DataFrame(rows).set_index("asset") if rows else pd.DataFrame()
    corrected = holm_bonferroni(pvals) if pvals else {}
    if not table.empty:
        table["significant_after_holm_bonferroni"] = [
            any(v["significant_at_alpha"] for k, v in corrected.items() if k.startswith(f"{a}::"))
            for a in table.index
        ]
    return table, corrected


def cross_asset_outlook_summary(regime_panel: pd.DataFrame, outlook: pd.DataFrame,
                                 asset_profile: pd.DataFrame, axis: str, canonical_labels: list,
                                 cfg: Config) -> pd.DataFrame:

    if outlook.empty or "current_posterior_h1" not in outlook.index:
        return pd.DataFrame()
    row = outlook.loc["current_posterior_h1"]
    sleeve_of = {a: ASSET_SLEEVES.get(a, "Other") for a in row.index}
    exp_ret = row.groupby(sleeve_of).mean()

    latest = regime_panel.index[-1]
    post_cols = [f"{axis}_proba_{lab}" for lab in canonical_labels]
    if all(c in regime_panel.columns for c in post_cols) and not asset_profile.empty:
        current_post = regime_panel.loc[latest, post_cols]
        current_post.index = canonical_labels
        piv_vol = asset_profile.pivot_table(index="regime", columns="sleeve",
                                             values="ann_vol", aggfunc="mean").reindex(canonical_labels)
        piv_n = asset_profile.pivot_table(index="regime", columns="sleeve",
                                           values="n_obs", aggfunc="mean").reindex(canonical_labels)
        vol_by_sleeve = current_post @ piv_vol.fillna(0.0)
        n_by_sleeve = current_post @ piv_n.fillna(0.0)
    else:
        vol_by_sleeve, n_by_sleeve = pd.Series(dtype=float), pd.Series(dtype=float)

    def _direction(x):
        if pd.isna(x):
            return "n/a"
        return "Positive" if x > 0.02 else ("Negative" if x < -0.02 else "Neutral")

    def _risk(v):
        if pd.isna(v):
            return "n/a"
        return "High" if v > 0.18 else ("Low" if v < 0.08 else "Moderate")

    def _confidence(n):
        if pd.isna(n):
            return "n/a"
        return "High" if n >= cfg.crossasset_min_obs * 3 else ("Moderate" if n >= cfg.crossasset_min_obs else "Low")

    rows = []
    for sleeve in exp_ret.index:
        rows.append({
            "sleeve": sleeve, "exp_return_h1": exp_ret.get(sleeve, np.nan),
            "direction": _direction(exp_ret.get(sleeve, np.nan)),
            "ann_vol": vol_by_sleeve.get(sleeve, np.nan), "risk": _risk(vol_by_sleeve.get(sleeve, np.nan)),
            "confidence": _confidence(n_by_sleeve.get(sleeve, np.nan)),
        })
    return pd.DataFrame(rows).set_index("sleeve")


def build_cross_asset_engine(prices: pd.DataFrame, regime_panel: pd.DataFrame,
                              forecast: dict, suite: dict, cfg: Config) -> dict:

    quadrant_canonical = list(cfg.quadrant_labels.values())
    vol_liq_canonical = list(cfg.vol_liq_order)

    profile_econ = regime_conditioned_asset_profile(prices, regime_panel, "quadrant", cfg)
    profile_stress = regime_conditioned_asset_profile(prices, regime_panel, "vol_liquidity", cfg)

    corr_econ = cross_asset_correlation_structure(prices, regime_panel, "quadrant", cfg)
    corr_stress = cross_asset_correlation_structure(prices, regime_panel, "vol_liquidity", cfg)

    beta_econ = regime_conditional_beta(prices, regime_panel, "quadrant", cfg)
    beta_stress = regime_conditional_beta(prices, regime_panel, "vol_liquidity", cfg)

    joint_matrix = joint_axis_sleeve_matrix(prices, regime_panel, cfg)

    fwd_econ = forward_conditional_returns(prices, regime_panel, "quadrant", cfg.forecast_horizons_economic, cfg)
    fwd_stress = forward_conditional_returns(prices, regime_panel, "vol_liquidity", cfg.forecast_horizons_stress, cfg)

    outlook_econ = probability_weighted_outlook(
        prices, regime_panel, forecast, "quadrant", quadrant_canonical,
        cfg.forecast_horizons_economic, fwd_econ, cfg)
    outlook_stress = probability_weighted_outlook(
        prices, regime_panel, forecast, "vol_liquidity", vol_liq_canonical,
        cfg.forecast_horizons_stress, fwd_stress, cfg)

    info_econ, holm_econ = regime_information_content(prices, regime_panel, "quadrant", cfg)
    info_stress, holm_stress = regime_information_content(prices, regime_panel, "vol_liquidity", cfg)

    outlook_summary = cross_asset_outlook_summary(
        regime_panel, outlook_econ, profile_econ, "quadrant", quadrant_canonical, cfg)

    return {
        "asset_profile": {"quadrant": profile_econ, "vol_liquidity": profile_stress},
        "correlation_structure": {"quadrant": corr_econ, "vol_liquidity": corr_stress},
        "conditional_beta": {"quadrant": beta_econ, "vol_liquidity": beta_stress},
        "joint_axis_matrix": joint_matrix,
        "forward_conditional_returns": {"quadrant": fwd_econ, "vol_liquidity": fwd_stress},
        "probability_weighted_outlook": {"quadrant": outlook_econ, "vol_liquidity": outlook_stress},
        "regime_information_content": {"quadrant": (info_econ, holm_econ),
                                        "vol_liquidity": (info_stress, holm_stress)},
        "outlook_summary": outlook_summary,
        "rank_stability_reused": suite.get("subperiod_rank_stability", {}),
    }

# REPORTING / TEARSHEET
QUADRANT_COLORS = {"Stable Growth": "#2ca02c", "Overheating": "#ff7f0e",
                    "Stagflation": "#d62728", "Contraction / Disinflation": "#7f7f7f"}
VOL_LIQ_COLORS = {"Calm": "#2ca02c", "Stress": "#ff7f0e", "Crisis": "#d62728"}


def _shade_regimes(ax, labels: pd.Series, color_map: dict):
    grp = (labels != labels.shift()).cumsum()
    for _, seg in labels.groupby(grp):
        ax.axvspan(seg.index[0], seg.index[-1], color=color_map.get(seg.iloc[0], "#cccccc"), alpha=0.25)


def _regime_segments(labels: pd.Series):

    labels = labels.dropna()
    grp = (labels != labels.shift()).cumsum()
    return [(seg.index[0], seg.index[-1], seg.iloc[0]) for _, seg in labels.groupby(grp)]


def plot_regime_timeline(regime_panel: pd.DataFrame, prices: pd.DataFrame, cfg: Config):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    bench = prices[cfg.benchmark_ticker].reindex(regime_panel.index)

    axes[0].plot(bench.index, bench.values, color="black", lw=1.2)
    axes[0].set_yscale("log")
    _shade_regimes(axes[0], regime_panel["quadrant"], QUADRANT_COLORS)
    axes[0].set_title(f"Growth x Inflation Quadrant vs. {cfg.benchmark_ticker} (log scale)")
    handles = [plt.Line2D([0], [0], color=c, lw=8, alpha=0.4) for c in QUADRANT_COLORS.values()]
    axes[0].legend(handles, QUADRANT_COLORS.keys(), loc="upper left", ncol=4, fontsize=8)

    axes[1].plot(bench.index, bench.values, color="black", lw=1.2)
    axes[1].set_yscale("log")
    _shade_regimes(axes[1], regime_panel["vol_liquidity"], VOL_LIQ_COLORS)
    axes[1].set_title(f"Volatility x Liquidity Stress State vs. {cfg.benchmark_ticker} (log scale)")
    handles2 = [plt.Line2D([0], [0], color=c, lw=8, alpha=0.4) for c in VOL_LIQ_COLORS.values()]
    axes[1].legend(handles2, VOL_LIQ_COLORS.keys(), loc="upper left", ncol=3, fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/01_regime_timeline.png", dpi=140)
    plt.close(fig)

def plot_transition_heatmap(transmat: np.ndarray, label_map: dict, title: str, fname: str):
    if transmat is None:
        return
    n = transmat.shape[0]
    labels = [label_map.get(i, str(i)) for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(transmat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center",
                     color="white" if transmat[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{fname}", dpi=140)
    plt.close(fig)

def plot_equity_curve(alloc, prices, cfg: Config):
    net = alloc["net_returns"].dropna()
    bench = prices[cfg.benchmark_ticker].pct_change().reindex(net.index).fillna(0)
    strat_curve = (1 + net).cumprod()
    bench_curve = (1 + bench).cumprod()

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(strat_curve.index, strat_curve.values, label="Regime-Conditioned Tactical Allocation", lw=1.6)
    axes[0].plot(bench_curve.index, bench_curve.values, label=cfg.benchmark_ticker, lw=1.2, color="gray")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper left")
    axes[0].set_title("Cumulative Growth of $1 (log scale)")

    strat_dd = strat_curve / strat_curve.cummax() - 1
    bench_dd = bench_curve / bench_curve.cummax() - 1
    axes[1].fill_between(strat_dd.index, strat_dd.values, 0, color="steelblue", alpha=0.6, label="Strategy DD")
    axes[1].plot(bench_dd.index, bench_dd.values, color="gray", lw=1, label=f"{cfg.benchmark_ticker} DD")
    axes[1].legend(loc="lower left")
    axes[1].set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/04_equity_curve_drawdown.png", dpi=140)
    plt.close(fig)

def plot_regime_risk_bars(quadrant_risk_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    order = [r for r in ["Stable Growth", "Overheating", "Stagflation", "Contraction / Disinflation"]
             if r in quadrant_risk_table.index]
    tbl = quadrant_risk_table.loc[order]
    x = np.arange(len(tbl))
    ax.bar(x - 0.2, tbl["ann_return"] * 100, width=0.4, label="Ann. Return %",
           color=[QUADRANT_COLORS[r] for r in order])
    ax.bar(x + 0.2, tbl.iloc[:, -1] * 100, width=0.4, label="CVaR (95%) monthly %", color="#555555")
    ax.set_xticks(x); ax.set_xticklabels(order, rotation=20)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend()
    ax.set_title("Strategy Return & Tail Risk by Growth x Inflation Regime")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/05_regime_risk_bars.png", dpi=140)
    plt.close(fig)

# INTERACTIVE PLOTLY DASHBOARD
DASH_BG = "#0b0e14"     
CARD_BG = "#151922"     
GRID_COLOR = "#2a2f3a"
FONT_COLOR = "#e5e7eb"
MUTED_COLOR = "#9ca3af"
ASSET_PALETTE = [
    "#4dd0e1", "#ff8a65", "#aed581", "#ba68c8", "#fff176", "#f06292",
    "#7986cb", "#4db6ac", "#ffb74d", "#90a4ae", "#81c784", "#e57373",
    "#64b5f6", "#dce775",
]

def _apply_dark_layout(fig, **extra_layout):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
        font=dict(color=FONT_COLOR), **extra_layout)
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR)
    return fig

def _add_regime_timeline_row(fig, row: int, bench: pd.Series, labels: pd.Series,
                              color_map: dict, cfg: Config, legendgroup: str):
    for start, end, label in _regime_segments(labels):
        fig.add_vrect(x0=start, x1=end, fillcolor=color_map.get(label, "#cccccc"),
                      opacity=0.30, line_width=0, row=row, col=1, exclude_empty_subplots=False)
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, mode="lines", name=cfg.benchmark_ticker,
                              line=dict(color=FONT_COLOR, width=1.3), showlegend=False), row=row, col=1)
    for label, color in color_map.items():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                  marker=dict(size=10, color=color, symbol="square", opacity=0.85),
                                  name=label, legendgroup=legendgroup), row=row, col=1)
    fig.update_yaxes(type="log", row=row, col=1, title_text=cfg.benchmark_ticker)


def _current_regime_probabilities_data(regime_panel: pd.DataFrame, cfg: Config):
    latest = regime_panel.index[-1]
    quad_labels = list(cfg.quadrant_labels.values())
    vol_labels = list(cfg.vol_liq_order)
    quad_cols = [f"quadrant_proba_{l}" for l in quad_labels if f"quadrant_proba_{l}" in regime_panel.columns]
    vol_cols = [f"vol_liquidity_proba_{l}" for l in vol_labels if f"vol_liquidity_proba_{l}" in regime_panel.columns]
    if not quad_cols or not vol_cols:
        return None
    quad_p = regime_panel.loc[latest, quad_cols]
    quad_p.index = [c.replace("quadrant_proba_", "") for c in quad_p.index]
    vol_p = regime_panel.loc[latest, vol_cols]
    vol_p.index = [c.replace("vol_liquidity_proba_", "") for c in vol_p.index]
    return {"latest": latest, "quad_p": quad_p, "vol_p": vol_p,
            "quad_state": regime_panel["quadrant"].iloc[-1],
            "vol_state": regime_panel["vol_liquidity"].iloc[-1]}


def _add_current_regime_probabilities(fig, row: int, data: dict):
    fig.add_trace(go.Bar(x=data["quad_p"].index, y=data["quad_p"].values,
                          marker_color=[QUADRANT_COLORS.get(l, MUTED_COLOR) for l in data["quad_p"].index],
                          showlegend=False), row=row, col=1)
    fig.add_trace(go.Bar(x=data["vol_p"].index, y=data["vol_p"].values,
                          marker_color=[VOL_LIQ_COLORS.get(l, MUTED_COLOR) for l in data["vol_p"].index],
                          showlegend=False), row=row, col=2)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=row, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=row, col=2)

def _transition_forecast_data(forecast: Optional[dict], cfg: Config):
    quad_labels = list(cfg.quadrant_labels.values())
    vol_labels = list(cfg.vol_liq_order)
    out, any_trace = {}, False
    for axis, labels in (("quadrant", quad_labels), ("vol_liquidity", vol_labels)):
        struct = (forecast or {}).get(axis, {}).get("structural", {})
        horizons = sorted(struct.keys())
        series_list = []
        for lab in labels:
            ys = []
            for h in horizons:
                panel = struct[h]
                valid = panel[lab].dropna() if (not panel.empty and lab in panel.columns) else pd.Series(dtype=float)
                ys.append(float(valid.iloc[-1]) if len(valid) else np.nan)
            if not all(np.isnan(y) for y in ys):
                series_list.append((lab, ys))
                any_trace = True
        out[axis] = {"horizons": horizons, "series": series_list}
    return out if any_trace else None

def _add_transition_forecast(fig, row: int, data: dict):
    for col, axis in enumerate(("quadrant", "vol_liquidity"), start=1):
        horizons = data[axis]["horizons"]
        for lab, ys in data[axis]["series"]:
            colors = QUADRANT_COLORS if axis == "quadrant" else VOL_LIQ_COLORS
            fig.add_trace(go.Bar(x=[f"+{h}M" for h in horizons], y=ys, name=lab,
                                  marker_color=colors.get(lab, MUTED_COLOR),
                                  showlegend=False), row=row, col=col)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=row, col=1)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", row=row, col=2)

def _confidence_entropy_data(regime_panel: pd.DataFrame, cfg: Config):
    out = []
    for axis, labels, color in (("quadrant", list(cfg.quadrant_labels.values()), "#4dd0e1"),
                                 ("vol_liquidity", list(cfg.vol_liq_order), "#ff8a65")):
        cols = [f"{axis}_proba_{l}" for l in labels]
        if not all(c in regime_panel.columns for c in cols):
            continue
        proba = regime_panel[cols].dropna()
        proba.columns = labels
        if proba.empty:
            continue
        name = "Growth x Inflation" if axis == "quadrant" else "Vol x Liquidity"
        out.append({"axis": axis, "name": name, "color": color,
                    "margin": posterior_margin(proba),
                    "entropy": posterior_entropy(proba, normalized=True)})
    return out if out else None

def _add_confidence_entropy(fig, margin_row: int, entropy_row: int, data: list):
    for d in data:
        fig.add_trace(go.Scatter(x=d["margin"].index, y=d["margin"].values, name=d["name"], mode="lines",
                                  line=dict(color=d["color"], width=1.6), legendgroup=d["axis"]),
                      row=margin_row, col=1)
        fig.add_trace(go.Scatter(x=d["entropy"].index, y=d["entropy"].values, name=d["name"], mode="lines",
                                  line=dict(color=d["color"], width=1.6), legendgroup=d["axis"],
                                  showlegend=False), row=entropy_row, col=1)
    fig.update_yaxes(range=[0, 1], row=margin_row, col=1)
    fig.update_yaxes(range=[0, 1], row=entropy_row, col=1)

def _recession_forecast_data(forecast: Optional[dict]):
    recession = (forecast or {}).get("recession", {})
    horizons = sorted(h for h in recession.keys() if isinstance(h, int))
    series = []
    for h in horizons:
        df = recession[h]
        if df is None or df.empty or "level" not in df.columns:
            continue
        s = df["level"].dropna()
        if not s.empty:
            series.append((h, s))
    return series if series else None

def _add_recession_forecast(fig, row: int, col: int, data: list):
    palette = {3: "#ffd54f", 6: "#f06292"}
    for h, s in data:
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name=f"{h}-month-ahead P(recession)",
                                  mode="lines", line=dict(width=1.8, color=palette.get(h, MUTED_COLOR))),
                      row=row, col=col)
    fig.add_hline(y=0.5, line_dash="dot", line_color=MUTED_COLOR, row=row, col=col)
    fig.update_yaxes(range=[0, 1], tickformat=".0%", title_text="Forecast probability", row=row, col=col)

def _cross_asset_outlook_data(cross_asset: Optional[dict]):
    summary = (cross_asset or {}).get("outlook_summary")
    if summary is None or summary.empty:
        return None
    return summary.sort_values("exp_return_h1")

def _add_cross_asset_outlook(fig, row: int, col: int, df: pd.DataFrame):
    risk_color = {"High": "#d62728", "Moderate": "#ff7f0e", "Low": "#2ca02c", "n/a": MUTED_COLOR}
    colors = [risk_color.get(r, MUTED_COLOR) for r in df["risk"]]
    labels = [f"{d} / {r} risk / {c} confidence"
              for d, r, c in zip(df["direction"], df["risk"], df["confidence"])]
    fig.add_trace(go.Bar(x=df["exp_return_h1"] * 100, y=df.index, orientation="h",
                          marker_color=colors, text=labels, textposition="outside",
                          showlegend=False), row=row, col=col)
    fig.add_vline(x=0, line_color=MUTED_COLOR, row=row, col=col)
    fig.update_xaxes(title_text="Probability-weighted expected 1-month return (%)", row=row, col=col)


def _add_equity_curve(fig, growth_row: int, dd_row: int, alloc: dict, prices: pd.DataFrame, cfg: Config):
    net = alloc["net_returns"].dropna()
    bench = prices[cfg.benchmark_ticker].pct_change().reindex(net.index).fillna(0)
    strat_curve = (1 + net).cumprod()
    bench_curve = (1 + bench).cumprod()
    strat_dd = strat_curve / strat_curve.cummax() - 1
    bench_dd = bench_curve / bench_curve.cummax() - 1

    fig.add_trace(go.Scatter(x=strat_curve.index, y=strat_curve.values,
                              name="Regime-Conditioned Tactical Allocation",
                              line=dict(width=2, color="#4dd0e1")), row=growth_row, col=1)
    fig.add_trace(go.Scatter(x=bench_curve.index, y=bench_curve.values, name=cfg.benchmark_ticker,
                              line=dict(width=1.3, color=MUTED_COLOR)), row=growth_row, col=1)
    fig.update_yaxes(type="log", row=growth_row, col=1)

    fig.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values, name="Strategy DD",
                              fill="tozeroy", line=dict(color="#4dd0e1")), row=dd_row, col=1)
    fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd.values, name=f"{cfg.benchmark_ticker} DD",
                              line=dict(color=MUTED_COLOR, width=1)), row=dd_row, col=1)

def _add_asset_cumulative_returns(fig, row: int, prices: pd.DataFrame, cfg: Config):
    norm = prices / prices.bfill().iloc[0]
    bench = cfg.benchmark_ticker
    others = [tk for tk in norm.columns if tk != bench]
    for i, tk in enumerate(others):
        fig.add_trace(go.Scatter(x=norm.index, y=norm[tk], name=tk, mode="lines",
                                  line=dict(width=1.3, color=ASSET_PALETTE[i % len(ASSET_PALETTE)]),
                                  opacity=0.85), row=row, col=1)
    if bench in norm.columns:
        fig.add_trace(go.Scatter(x=norm.index, y=norm[bench], name=f"{bench} (benchmark)",
                                  mode="lines", line=dict(width=3.2, color="#ffd54f")), row=row, col=1)
    fig.update_yaxes(type="log", title_text="Cumulative growth of $1 (log scale)", row=row, col=1)


def _add_transition_heatmap(fig, row: int, col: int, transmat: np.ndarray, label_map: dict):
    n = transmat.shape[0]
    labels = [label_map.get(i, str(i)) for i in range(n)]
    fig.add_trace(go.Heatmap(
        z=transmat, x=labels, y=labels, colorscale="Viridis", zmin=0, zmax=1,
        text=np.round(transmat, 2), texttemplate="%{text}",
        hovertemplate="from %{y} -> %{x}: %{z:.3f}<extra></extra>",
        showscale=False), row=row, col=col)
    fig.update_yaxes(autorange="reversed", row=row, col=col)


def _add_regime_risk_bars(fig, row: int, col: int, quadrant_risk_table: pd.DataFrame):
    order = [r for r in ["Stable Growth", "Overheating", "Stagflation", "Contraction / Disinflation"]
             if r in quadrant_risk_table.index]
    tbl = quadrant_risk_table.loc[order]
    cvar_col = tbl.columns[-1]
    fig.add_trace(go.Bar(x=order, y=tbl["ann_return"] * 100, name="Ann. Return %",
                          marker_color=[QUADRANT_COLORS[r] for r in order], showlegend=False),
                  row=row, col=col)
    fig.add_trace(go.Bar(x=order, y=tbl[cvar_col] * 100, name=f"{cvar_col} monthly %",
                          marker_color=MUTED_COLOR, showlegend=False), row=row, col=col)


def _feature_importance_data(validator: Optional[dict]):
    if validator is None:
        return None
    return validator["top_features"].sort_values(ascending=True)


def _add_feature_importance(fig, row: int, col: int, imp: pd.Series):
    fig.add_trace(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#4dd0e1",
                          showlegend=False), row=row, col=col)


def build_plotly_dashboard(cfg: Config, regime_panel: pd.DataFrame, prices: pd.DataFrame,
                            alloc: dict, quad_fit: dict, vol_liq_fit: dict,
                            quadrant_risk_table: pd.DataFrame, validator: Optional[dict],
                            forecast: Optional[dict] = None,
                            cross_asset: Optional[dict] = None) -> str:
    quad_label_map = quad_fit["last_label_map"] or {}
    vol_label_map = vol_liq_fit["last_label_map"] or {}

    bench = prices[cfg.benchmark_ticker].reindex(regime_panel.index)
    curr_regime_data = _current_regime_probabilities_data(regime_panel, cfg)
    transition_data = _transition_forecast_data(forecast, cfg)
    confidence_data = _confidence_entropy_data(regime_panel, cfg)
    recession_data = _recession_forecast_data(forecast)
    cross_asset_data = _cross_asset_outlook_data(cross_asset)
    quad_transmat = quad_fit["last_transition_matrix"]
    vol_transmat = vol_liq_fit["last_transition_matrix"]
    feature_imp = _feature_importance_data(validator)

    row_plan = [
        {"kind": "full", "weight": 1.4,
         "title": f"Growth x Inflation Quadrant vs. {cfg.benchmark_ticker} (log scale)",
         "add": lambda fig, row: _add_regime_timeline_row(
             fig, row, bench, regime_panel["quadrant"], QUADRANT_COLORS, cfg, "quad")},
        {"kind": "full", "weight": 1.4,
         "title": f"Volatility x Liquidity Stress State vs. {cfg.benchmark_ticker} (log scale)",
         "add": lambda fig, row: _add_regime_timeline_row(
             fig, row, bench, regime_panel["vol_liquidity"], VOL_LIQ_COLORS, cfg, "volliq")},
    ]
    if curr_regime_data is not None:
        row_plan.append({"kind": "pair", "weight": 0.9,
            "left": {"title": f"Growth x Inflation — current: {curr_regime_data['quad_state']}",
                     "add": lambda fig, row, col: _add_current_regime_probabilities(fig, row, curr_regime_data)},
            "right": {"title": f"Volatility x Liquidity — current: {curr_regime_data['vol_state']}",
                      "add": lambda fig, row, col: None}})
    if transition_data is not None:
        row_plan.append({"kind": "pair", "weight": 1.0,
            "left": {"title": "Growth x Inflation — forecast P(state) by horizon",
                     "add": lambda fig, row, col: _add_transition_forecast(fig, row, transition_data)},
            "right": {"title": "Volatility x Liquidity — forecast P(state) by horizon",
                      "add": lambda fig, row, col: None}})
    if confidence_data is not None:
        row_plan.append({"kind": "full", "weight": 0.9,
                          "title": "Posterior Margin (top state - runner-up)", "add": None})
        row_plan.append({"kind": "full", "weight": 0.9,
                          "title": "Normalized Posterior Entropy", "add": None})
    if recession_data is not None or cross_asset_data is not None:
        row_plan.append({"kind": "pair", "weight": 1.0,
            "left": ({"title": "Recession Forecast Probability (walk-forward, purged horizon)",
                      "add": lambda fig, row, col: _add_recession_forecast(fig, row, col, recession_data)}
                     if recession_data is not None else None),
            "right": ({"title": "Cross-Asset Outlook by Sleeve (posterior-weighted)",
                       "add": lambda fig, row, col: _add_cross_asset_outlook(fig, row, col, cross_asset_data)}
                      if cross_asset_data is not None else None)})
    row_plan.append({"kind": "full", "weight": 1.3, "title": "Cumulative Growth of $1 (log scale)", "add": None})
    row_plan.append({"kind": "full", "weight": 0.6, "title": "Drawdown", "add": None})
    row_plan.append({"kind": "full", "weight": 1.3,
                      "title": f"Cumulative Returns by Asset vs. {cfg.benchmark_ticker}",
                      "add": lambda fig, row: _add_asset_cumulative_returns(fig, row, prices, cfg)})
    if quad_transmat is not None or vol_transmat is not None:
        row_plan.append({"kind": "pair", "weight": 1.1,
            "left": ({"title": "Growth x Inflation HMM Transition Matrix (most recent refit)",
                      "add": lambda fig, row, col: _add_transition_heatmap(fig, row, col, quad_transmat, quad_label_map)}
                     if quad_transmat is not None else None),
            "right": ({"title": "Volatility x Liquidity HMM Transition Matrix (most recent refit)",
                       "add": lambda fig, row, col: _add_transition_heatmap(fig, row, col, vol_transmat, vol_label_map)}
                      if vol_transmat is not None else None)})
    row_plan.append({"kind": "pair", "weight": 1.0,
        "left": {"title": "Strategy Return & Tail Risk by Growth x Inflation Regime",
                 "add": lambda fig, row, col: _add_regime_risk_bars(fig, row, col, quadrant_risk_table)},
        "right": ({"title": "Top RF Feature Importances (in-sample)",
                   "add": lambda fig, row, col: _add_feature_importance(fig, row, col, feature_imp)}
                  if feature_imp is not None else None)})

    # subplot_titles takes exactly one entry per REAL subplot cell (row-major,
    # skipping every cell a colspan marks None) -- NOT a dense rows*cols list
    # with "" placeholders for the spanned-away cell. Supplying a placeholder
    # there silently pushes every later title one slot down the list Plotly
    # actually consumes, so titles for the last few rows get dropped off the
    # end with no error (confirmed by direct reproduction against Plotly's
    # own make_subplots before landing this fix).
    specs, subplot_titles, weights = [], [], []
    for entry in row_plan:
        if entry["kind"] == "full":
            specs.append([{"colspan": 2}, None])
            subplot_titles.append(entry["title"])
        else:
            specs.append([{}, {}])
            subplot_titles.append(entry["left"]["title"] if entry["left"] else "")
            subplot_titles.append(entry["right"]["title"] if entry["right"] else "")
        weights.append(entry["weight"])

    n_rows = len(row_plan)
    vspacing = min(0.02, 0.9 / max(n_rows - 1, 1))
    fig = make_subplots(rows=n_rows, cols=2, specs=specs, subplot_titles=subplot_titles,
                         row_heights=[w / sum(weights) for w in weights],
                         vertical_spacing=vspacing)

    confidence_rows, equity_rows = [], []
    for i, entry in enumerate(row_plan, start=1):
        if entry["kind"] == "full" and entry["title"] == "Posterior Margin (top state - runner-up)":
            confidence_rows.append(i)
        elif entry["kind"] == "full" and entry["title"] == "Normalized Posterior Entropy":
            confidence_rows.append(i)
        elif entry["kind"] == "full" and entry["title"] == "Cumulative Growth of $1 (log scale)":
            equity_rows.append(i)
        elif entry["kind"] == "full" and entry["title"] == "Drawdown":
            equity_rows.append(i)
    if confidence_data is not None and len(confidence_rows) == 2:
        _add_confidence_entropy(fig, confidence_rows[0], confidence_rows[1], confidence_data)
    if len(equity_rows) == 2:
        _add_equity_curve(fig, equity_rows[0], equity_rows[1], alloc, prices, cfg)

    for i, entry in enumerate(row_plan, start=1):
        if entry["kind"] == "full":
            if entry["add"] is not None:
                entry["add"](fig, i)
        else:
            if entry["left"] is not None:
                entry["left"]["add"](fig, i, 1)
            if entry["right"] is not None:
                entry["right"]["add"](fig, i, 2)

    total_height = max(1400, int(sum(weights) * 230))
    _apply_dark_layout(fig, height=total_height, hovermode="x unified", barmode="group",
                        title="Macro Regime Classifier",
                        legend=dict(groupclick="togglegroup"),
                        margin=dict(l=90, r=40, t=70, b=40))

    html = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False})
    parts = [
        "<html><head><meta charset='utf-8'><title>Macro Regime Classifier — Dashboard</title>",
        "<style>",
        f"  body {{ background:{DASH_BG}; color:{FONT_COLOR}; "
        "font-family:Arial,Helvetica,sans-serif; margin:0; padding:24px 28px 40px; }",
        f"  p.subtitle {{ color:{MUTED_COLOR}; margin-top:0; margin-bottom:16px; }}",
        "</style></head>",
        "<body>",
        f"<p class='subtitle'>Sample: {regime_panel.index.min().date()} -&gt; "
        f"{regime_panel.index.max().date()} ({len(regime_panel)} months)</p>",
        html,
        "</body></html>",
    ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False,
                                      prefix="macro_regime_dashboard_") as f:
        f.write("\n".join(parts))
        out_path = f.name
    webbrowser.open(f"file://{out_path}")
    return out_path

def print_tearsheet(cfg, regime_panel, quad_fit, vol_liq_fit, validator,
                     alloc, quadrant_risk_table, pit_risk, suite, gauge_info,
                     truth_df=None, forecast=None, confidence=None, validation=None,
                     cross_asset=None):
    lines = []
    W = 82
    def hr(): lines.append("-" * W)
    def title(t): lines.append(t.center(W))

    lines.append("=" * W)
    title("INSTITUTIONAL MACRO REGIME CLASSIFIER — TEARSHEET")
    lines.append("=" * W)
    lines.append(f"Sample: {regime_panel.index.min().date()} -> {regime_panel.index.max().date()}  "
                 f"({len(regime_panel)} months)")
    hr()

    lines.append("REGIME DISTRIBUTION — Growth x Inflation Quadrant (HMM-filtered, primary)")
    dist = regime_panel["quadrant"].value_counts(normalize=True).mul(100).round(1)
    for k, v in dist.items():
        lines.append(f"  {k:<16} {v:>5.1f}%")
    lines.append("")
    lines.append("REGIME DISTRIBUTION — Volatility x Liquidity Stress State (HMM-filtered, primary)")
    dist2 = regime_panel["vol_liquidity"].value_counts(normalize=True).mul(100).round(1)
    for k, v in dist2.items():
        lines.append(f"  {k:<16} {v:>5.1f}%")
    hr()

    # Section 5, issue 6: occupancy/episode diagnostics per state.
    lines.append("STATE OCCUPANCY / EPISODE DIAGNOSTICS (Section 5, issue 6)")
    occ_q = regime_occupancy_stats(regime_panel["quadrant"], cfg)
    occ_v = regime_occupancy_stats(regime_panel["vol_liquidity"], cfg)
    for label, occ in (("Quadrant", occ_q), ("Vol/Liquidity", occ_v)):
        lines.append(f"  {label}:")
        for state, row in occ.iterrows():
            flag = "  ** below min_state_occupancy_months **" if row["below_min_occupancy"] else ""
            lines.append(f"    {state:<28} n={row['n_months']:>4.0f} "
                         f"({row['occupancy_share']*100:4.1f}%)  episodes={row['n_episodes']:>3.0f}  "
                         f"avg={row['avg_duration_months']:4.1f}mo  "
                         f"longest={row['longest_duration_months']:>3.0f}mo{flag}")
    hr()

    lines.append("HMM FIT DIAGNOSTICS -- convergence (n_iter="
                 f"{cfg.hmm_n_iter}, n_restarts={cfg.hmm_n_restarts}, best-of-restarts kept)")
    for label, fit in (("Quadrant", quad_fit), ("Vol/Liquidity", vol_liq_fit)):
        n_refits = fit.get("hmm_n_refits", 0)
        n_fail = fit.get("hmm_convergence_failures", 0)
        flag = "  ** some refits did not fully converge -- see KNOWN LIMITATIONS **" if n_fail else ""
        lines.append(f"  {label:<16} {n_refits - n_fail}/{n_refits} refits converged within "
                     f"n_iter={cfg.hmm_n_iter}{flag}")
    hr()

    lines.append("QUADRANT ASSIGNMENT CONFIDENCE (Section 5, issue 1 caveat, most recent refit)")
    last_quality = quad_fit["state_quality_by_refit"][-1] if quad_fit.get("state_quality_by_refit") else {}
    for label, q in last_quality.items():
        flag = "  LOW CONFIDENCE" if q["low_confidence"] else ""
        lines.append(f"  {label:<28} distance_to_prototype={q['distance_to_prototype']:.2f}{flag}")
    hr()

    def _emit_confirmation_snapshot(axis_label, prefix):
        if regime_panel.empty:
            return
        row = regime_panel.iloc[-1]
        proba_prefix = f"{prefix}_proba_"
        proba_labels = [c[len(proba_prefix):] for c in regime_panel.columns if c.startswith(proba_prefix)]
        lines.append(axis_label.upper())
        lines.append(f"  Confirmed: {row[prefix]}")
        lines.append(f"  Candidate: {row[f'{prefix}_raw_candidate']}")
        lines.append("")
        lines.append("  Posterior:")
        for lab in proba_labels:
            lines.append(f"    {lab:<28} {row[proba_prefix + lab]*100:5.1f}%")
        lines.append("")
        lines.append(f"  Margin:          {row[f'{prefix}_margin']*100:5.1f}%")
        lines.append(f"  Persistence:     {int(row[f'{prefix}_persistence_count'])}")
        lines.append(f"  Model agreement: {row[f'{prefix}_model_agreement']*100:5.0f}%")
        lines.append("")
        lines.append("  Classification status:")
        lines.append(f"    {row[f'{prefix}_status'].upper()}")
        lines.append("")
        lines.append("  Confidence:")
        lines.append(f"    {row[f'{prefix}_confidence_label']}")

    lines.append("REGIME CONFIRMATION SNAPSHOT (Step 6, most recent month)")
    _emit_confirmation_snapshot("Economic Regime", "quadrant")
    lines.append("")
    _emit_confirmation_snapshot("Financial-Stress Regime", "vol_liquidity")
    hr()

    lines.append("PCA COMPOSITE GAUGES — explained variance of PC1 (data-driven weighting), "
                 "Section 4.J loading stability")
    for key, val in gauge_info.items():
        if key.endswith("_explained_var"):
            block_name = key[:-len("_explained_var")]
            stab = gauge_info.get(f"{block_name}_loading_stability", np.nan)
            stab_str = f", loading stability {stab:+.2f}" if not (isinstance(stab, float) and np.isnan(stab)) else ""
            lines.append(f"  {block_name.capitalize() + ' gauge:':<17} {val*100:5.1f}% of basket variance{stab_str}")
    hr()

    if validator is not None:
        lines.append("SUPERVISED VALIDATOR — Random Forest vs. NBER USREC (USREC excluded from features)")
        lines.append(f"  Walk-forward macro-F1 (5-fold TimeSeriesSplit): "
                     f"{validator['walkforward_macro_f1_mean']:.3f}")
        lines.append(f"  Fold F1 scores: {[round(f,3) for f in validator['walkforward_macro_f1_folds']]}")
        lines.append("  Top predictive features (in-sample importance):")
        for feat, imp in validator["top_features"].head(6).items():
            lines.append(f"    {feat:<20} {imp:.4f}")
        hr()

    lines.append("TACTICAL ALLOCATION (Application / demonstration portfolio -- built ON TOP "
                 "OF Step 10's cross-asset research below, not the research itself)")
    lines.append("  walk-forward, regime-conditioned, vol-targeted, liquidity-throttled")
    net = alloc["net_returns"].dropna()
    bench = None
    sharpe = (net.mean() / net.std()) * np.sqrt(12) if net.std() > 0 else np.nan
    lines.append(f"  Annualized return (net of {cfg.txn_cost_bps:.0f}bps costs): {net.mean()*12*100:6.2f}%")
    lines.append(f"  Annualized volatility:                          {net.std()*np.sqrt(12)*100:6.2f}%")
    lines.append(f"  Sharpe ratio:                                    {sharpe:6.3f}")
    lines.append(f"  Average leverage/exposure scale used:            {alloc['leverage'].mean():6.2f}x")
    lines.append(f"  Average monthly turnover:                        {alloc['turnover'].mean()*100:6.2f}%")
    hr()

    lines.append("REGIME-CONDITIONED RISK (VaR / CVaR by Growth x Inflation quadrant)")
    lines.append(quadrant_risk_table.round(4).to_string())
    lines.append("")
    lines.append("DYNAMIC RISK LOOKBACK / LEVERAGE CAP BY CURRENT VOL/LIQUIDITY STATE")
    lines.append(f"  Calm:   {cfg.lookback_calm}mo lookback, leverage cap {cfg.leverage_calm}x")
    lines.append(f"  Stress: {cfg.lookback_stress}mo lookback, leverage cap {cfg.leverage_stress}x")
    lines.append(f"  Crisis: {cfg.lookback_crisis}mo lookback, leverage cap {cfg.leverage_crisis}x")
    hr()

    lines.append("DEFENSIBILITY SUITE")
    bci = suite["bootstrap_sharpe_ci_strategy"]
    lines.append(f"  [1] Block-bootstrap Sharpe 95% CI (strategy): "
                 f"{bci['sharpe_point']:.3f}  [{bci['sharpe_ci_95'][0]:.3f}, {bci['sharpe_ci_95'][1]:.3f}]")
    bciB = suite["bootstrap_sharpe_ci_benchmark"]
    lines.append(f"      Block-bootstrap Sharpe 95% CI (benchmark):  "
                 f"{bciB['sharpe_point']:.3f}  [{bciB['sharpe_ci_95'][0]:.3f}, {bciB['sharpe_ci_95'][1]:.3f}]")
    pbt = suite["paired_bootstrap_vs_benchmark"]
    lines.append(f"  [2] Paired block-bootstrap vs. benchmark: mean monthly diff "
                 f"{pbt['observed_mean_monthly_diff']*100:+.3f}%, p={pbt['p_value_one_sided']:.4f}")
    perm = suite["regime_permutation_test"]
    lines.append(f"  [3] Regime-label permutation test ({perm['n_episodes']} episodes shuffled, "
                 f"{cfg.n_permutation} draws): real-label proxy "
                 f"{perm['observed_proxy_mean_monthly_return']*100:.3f}% vs. shuffled-label proxy "
                 f"{perm['permuted_mean']*100:.3f}% +/- {perm['permuted_std']*100:.3f}%, "
                 f"p={perm['p_value']:.4f}")
    lines.append(f"      (both sides use the same proxy allocation — only the label assignment "
                 f"differs; actual net-of-cost strategy return, shown for context and NOT part of "
                 f"this test: {perm['observed_strategy_mean_monthly_return']*100:.3f}%/mo)")
    lines.append("  [4] Sub-period rank stability (mean Spearman rho across thirds, by regime):")
    for regime, rho in suite["subperiod_rank_stability"].items():
        lines.append(f"        {regime:<16} rho={rho:.3f}" if not np.isnan(rho) else f"        {regime:<16} n/a")
    lines.append("  [5] Transaction cost sensitivity:")
    lines.append("      " + suite["txn_cost_sensitivity"].round(4).to_string().replace("\n", "\n      "))
    hr_res = suite["walkforward_hit_rate"]
    lines.append(f"  [6] Walk-forward hit rate vs. ex-post oracle top-{cfg.top_n_assets}: "
                 f"mean overlap {hr_res['mean_overlap']:.2f}/{hr_res['max_possible']} "
                 f"over {hr_res['n_periods']} periods")
    if "jennrich_stablegrowth_vs_contraction" in suite:
        j = suite["jennrich_stablegrowth_vs_contraction"]
        if not np.isnan(j.get("chi2", np.nan)):
            lines.append(f"  [7] Jennrich test, Stable Growth vs. Contraction/Disinflation correlation matrices: "
                         f"chi2={j['chi2']:.2f}, dof={j['dof']:.0f}, p={j['p_value']:.4f}")
    lines.append("  [8] Newey-West HAC t-stats, mean monthly return by regime:")
    lines.append("      " + suite["newey_west_regime_tstats"].round(4).to_string().replace("\n", "\n      "))
    lines.append("  [9] Holm-Bonferroni correction across all p-values above (family-wise alpha=0.05):")
    for name, res in suite["holm_bonferroni"].items():
        flag = "significant" if res["significant_at_alpha"] else "not significant"
        lines.append(f"        {name:<38} p_raw={res['p_raw']:.4f}  p_holm={res['p_holm']:.4f}  ({flag})")
    hr()

    if forecast:
        def _emit_forecast_block(axis_label, axis_key):
            fdata = forecast.get(axis_key)
            if not fdata:
                return
            canonical_labels, horizons = fdata["canonical_labels"], fdata["horizons"]
            lines.append(axis_label.upper())
            lines.append(f"  Current confirmed state: {fdata['current_state']}  "
                         f"(in-state for {fdata['current_state_duration']} months)")
            lines.append("")

            def _table(panel_dict, title_):
                lines.append(f"  {title_}")
                lines.append("  " + f"{'':<28}" + "".join(f"{f'{h}M':>8}" for h in horizons))
                for lab in canonical_labels:
                    row = "  " + f"{lab:<28}"
                    for h in horizons:
                        panel = panel_dict.get(h)
                        v = panel[lab].iloc[-1] if panel is not None and len(panel) else np.nan
                        row += f"{v*100:7.1f}%" if not (v is None or (isinstance(v, float) and np.isnan(v))) else f"{'n/a':>8}"
                    lines.append(row)
                lines.append("")

            _table(fdata["structural"], "STRUCTURAL FORECAST (p_t x A^h, HMM transition-matrix propagation)")
            _table(fdata["feature_conditioned"], "FEATURE-CONDITIONED FORECAST (multinomial logistic regression, challenger)")

            agree_bits = [f"{h}M={fdata['agreement'][h].iloc[-1]*100:.0f}%" for h in horizons
                          if h in fdata["agreement"] and len(fdata["agreement"][h].dropna())]
            if agree_bits:
                lines.append("  Structural vs. feature-conditioned agreement: " + ", ".join(agree_bits))
                lines.append("")

            for h in horizons:
                risk_h = fdata["transition_risk"].get(h)
                if risk_h is None or risk_h["change_risk"].dropna().empty:
                    continue
                chg = risk_h["change_risk"].iloc[-1]
                dest = risk_h["conditional_destination"].iloc[-1].dropna().sort_values(ascending=False)
                dest_str = ", ".join(f"{k} {v*100:.0f}%" for k, v in dest.items())
                lines.append(f"  {h}M end-horizon change risk (P(S_t+{h} != S_t)): {chg*100:5.1f}%"
                             + (f"   conditional destination: {dest_str}" if dest_str else ""))
            lines.append("")

            mc = fdata.get("mc_summary")
            if mc:
                lines.append(f"  MONTE CARLO PATH SIMULATION ({mc['n_paths']:,} paths, {mc['horizon']}M horizon, "
                             f"from current state)")
                lines.append(f"    Remain {mc['start_state']} throughout: {mc['remain_throughout']*100:5.1f}%")
                for lab, p in mc["ever_visited"].items():
                    if lab != mc["start_state"]:
                        lines.append(f"    Enter {lab} at least once:  {p*100:5.1f}%")
                for h in horizons:
                    if h in mc["transition_within_h"]:
                        lines.append(f"    Any-passage transition within {h}M (hitting probability): "
                                     f"{mc['transition_within_h'][h]*100:5.1f}%")
                lines.append("")

            cur = fdata["current_state"]
            exp_dur = fdata.get("expected_duration_analytic", {}).get(cur, np.nan)
            lines.append(f"  Expected state duration -- analytic (1/(1-p_ii)): {exp_dur:.1f} months")
            cur_dur = fdata.get("current_state_duration")
            if (not np.isnan(exp_dur)) and cur_dur is not None and cur_dur > 2 * max(exp_dur, 1e-6):
                lines.append(f"    NOTE: this is well below the {cur_dur}mo the CONFIRMED regime has "
                             f"actually persisted -- 1/(1-p_ii) uses the most recent refit's RAW filtered-"
                             f"posterior transition matrix, which can show rapid flip-flopping between "
                             f"adjacent states even while Step 6's hysteresis/persistence engine holds the "
                             f"reported confirmed_regime steady through that noise. Both numbers are "
                             f"honestly computed; they answer different questions (raw HMM dynamics vs. "
                             f"the smoothed regime actually reported).")
            rem = fdata.get("expected_remaining_duration_empirical", np.nan)
            if not (rem is None or np.isnan(rem)):
                lines.append(f"  Expected remaining duration -- empirical, duration-dependent "
                             f"(already {fdata['current_state_duration']}mo in state): {rem:.1f} months")
            else:
                lines.append("  Expected remaining duration -- empirical: n/a (fewer than 3 completed "
                             "historical episodes of this state -- not enough data for a trustworthy "
                             "hazard curve; trust the analytic estimate above instead)")

            grad = fdata.get("confidence_gradient", {})
            grad_str = ", ".join(f"{h}M={grad[h]*100:.1f}%" for h in horizons if h in grad)
            if grad_str:
                lines.append(f"  Forecast confidence gradient (mean max-probability, should decay "
                             f"with horizon): {grad_str}")

        lines.append("TRANSITION & FORECASTING ENGINE (Step 7)")
        _emit_forecast_block("Economic Regime Forecast", "quadrant")
        lines.append("")
        _emit_forecast_block("Financial-Stress Regime Forecast", "vol_liquidity")
        hr()

        rec = forecast.get("recession", {})
        if rec:
            lines.append("RECESSION FORECAST (Random Forest, walk-forward, horizon-shifted NBER USREC "
                         "target -- separate from the Contraction/Disinflation quadrant-state forecast "
                         "above, per Section 1's taxonomy)")
            for h in (3, 6):
                df = rec.get(h)
                if df is None or df["level"].dropna().empty:
                    continue
                last = df.dropna(subset=["level"]).iloc[-1]
                d1 = last.get("delta_1m", np.nan)
                d3 = last.get("delta_3m", np.nan)
                lines.append(f"  {h}M probability: {last['level']*100:5.1f}%   "
                             f"(delta 1m: {d1*100:+5.1f}pp, delta 3m: {d3*100:+5.1f}pp)" if not np.isnan(d1)
                             else f"  {h}M probability: {last['level']*100:5.1f}%")
                if h == 3 and not np.isnan(d1):
                    lines.append(f"    Previous month:  {(last['level']-d1)*100:5.1f}%")
                if h == 3 and confidence:
                    ci = confidence.get("recession_ci", {}).get("ci_3m")
                    if ci and not (np.isnan(ci[0]) or np.isnan(ci[1])):
                        lines.append(f"    95% CI (block-bootstrap RF refit, "
                                     f"n={cfg.confidence_n_bootstrap}): [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
            cal = rec.get("calibration_3m")
            if cal and not (cal.get("brier") is None or np.isnan(cal.get("brier", np.nan))):
                lines.append(f"  3M forecast calibration (out-of-sample): Brier={cal['brier']:.4f}, "
                             f"LogLoss={cal['log_loss']:.4f}, n={cal['n']}")
            hr()

    if confidence:
        def _fmt_pct(v):
            return f"{v*100:5.1f}%" if v is not None and not (isinstance(v, float) and np.isnan(v)) else "  n/a"

        def _emit_confidence_block(axis_label, axis_key):
            cdata = confidence.get(axis_key)
            if not cdata:
                return
            canonical_labels = cdata["canonical_labels"]
            lines.append(axis_label.upper())
            lines.append(f"  Confirmed regime:            {cdata['confirmed_regime']}")
            lines.append(f"  Posterior probability:       {_fmt_pct(cdata['posterior_confirmed'])}")
            lines.append(f"  Probability margin:          {_fmt_pct(cdata['margin'])}")
            lines.append(f"  State entropy (normalized):  {cdata['entropy_norm']:.2f}  ({cdata['entropy_band']})")
            lines.append(f"  GMM/HMM agreement:           {'Yes' if cdata['gmm_hmm_agreement'] else 'No'}")
            chall, chall_p = cdata["most_likely_challenger"]
            if chall is not None:
                lines.append(f"  Most likely challenger:       {chall} ({_fmt_pct(chall_p).strip()})")
            lines.append(f"  Cluster fit (Mahalanobis):    {cdata['cluster_fit_label']}"
                         + (f"  (distance={cdata['cluster_fit_distance']:.2f})"
                            if not np.isnan(cdata['cluster_fit_distance']) else ""))
            lines.append(f"  Out-of-distribution flag:    {'YES' if cdata['ood_flag'] else 'No'}")
            lines.append("")

            td = cdata.get("typical_duration", {})
            med, p75 = td.get("median_duration_months", np.nan), td.get("p75_duration_months", np.nan)
            hist_str = (f"  (historical median {med:.0f}mo, 75th pct {p75:.0f}mo)"
                        if not (np.isnan(med) or np.isnan(p75)) else "")
            lines.append(f"  State age:                   {cdata['state_age']} months{hist_str}")
            st = cdata.get("stability", {})
            if st:
                check_str = ", ".join(k for k, v in st["checks"].items() if v)
                lines.append(f"  Regime stability:             {st['label'].upper()}  "
                             f"({st['n_pass']}/{st['n_total']} checks pass: {check_str})")
            lines.append("")

            state_ci = cdata.get("gmm_membership_ci", {})
            if state_ci:
                cur = cdata["confirmed_regime"]
                lo, hi = state_ci.get(cur, (np.nan, np.nan))
                if not (np.isnan(lo) or np.isnan(hi)):
                    lines.append(f"  GMM membership bootstrap CI (block-bootstrap refit, n={cfg.confidence_n_bootstrap}"
                                 f" -- cluster-membership stability proxy, NOT a CI on the HMM posterior above):")
                    lines.append(f"    P({cur}) 95% CI:            [{lo*100:.1f}%, {hi*100:.1f}%]")
            transmat_ci = cdata.get("transmat_ci", {})
            cur = cdata["confirmed_regime"]
            if transmat_ci and cur in transmat_ci:
                lines.append(f"  Transition-matrix 95% CI (row: {cur} ->):")
                for to_lab in canonical_labels:
                    lo, hi = transmat_ci[cur].get(to_lab, (np.nan, np.nan))
                    if not (np.isnan(lo) or np.isnan(hi)):
                        lines.append(f"    -> {to_lab:<26} [{lo:.2f}, {hi:.2f}]")
            lines.append("")

            lines.append("  FORWARD RISK (from Step 7, kept separate from current-state confidence above)")
            lines.append(f"    Transition entropy (this state's next-step row): "
                         f"{cdata['transition_entropy_current']:.2f}  ({cdata['transition_entropy_band']})")
            if cdata["forward_horizon"] is not None:
                lines.append(f"    {cdata['forward_horizon']}M transition risk:      {_fmt_pct(cdata['forward_change_risk'])}")
                if cdata["forward_destination"]:
                    dest_lab, dest_p = cdata["forward_destination"]
                    lines.append(f"    Most likely destination:     {dest_lab} ({_fmt_pct(dest_p).strip()})")
                lines.append(f"    Forecast confidence ({cdata['forward_horizon']}M):    {_fmt_pct(cdata['forward_confidence'])}")
            lines.append("")

            sc = cdata["state_confidence"]
            check_str = ", ".join(k for k, v in sc["checks"].items() if v)
            fail_str = ", ".join(k for k, v in sc["checks"].items() if not v)
            lines.append(f"  STATE CONFIDENCE: {sc['label']}  ({sc['n_pass']}/{sc['n_total']} checks pass"
                         + (f": {check_str}" if check_str else "")
                         + (f"; failed: {fail_str}" if fail_str else "") + ")")

        lines.append("CONFIDENCE, UNCERTAINTY & REGIME STABILITY (Step 8)")
        _emit_confidence_block("Economic Regime", "quadrant")
        lines.append("")
        _emit_confidence_block("Financial-Stress Regime", "vol_liquidity")
        hr()

        def _emit_evidence(axis_label, axis_key):
            cdata = confidence.get(axis_key)
            if not cdata:
                return
            ev = cdata.get("evidence", {})
            lines.append(f"  {axis_label} (drives this axis's clustering directly):")
            for c, v in sorted(ev.get("driving", {}).items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"    {c:<26} {v:+.2f}")
            lines.append(f"  Broader macro context (NOT inputs to this axis -- corroborating evidence only):")
            for c, v in sorted(ev.get("context", {}).items(), key=lambda kv: -abs(kv[1])):
                lines.append(f"    {c:<26} {v:+.2f}")
            lines.append("")

        lines.append("REGIME EVIDENCE — current factor readings (z-scores, most recent vintage)")
        _emit_evidence("Economic Regime axis", "quadrant")
        _emit_evidence("Financial-Stress Regime axis", "vol_liquidity")
        hr()

        fu = confidence.get("factor_uncertainty", {})
        if fu:
            lines.append("FACTOR (PCA) UNCERTAINTY — current vs. historical explained variance")
            for block, d in fu.items():
                flag = "  ** LOW RELIABILITY **" if d["low_reliability"] else ""
                med_str = f"{d['historical_median_evr']*100:5.1f}%" if not np.isnan(d['historical_median_evr']) else "  n/a"
                stab = d.get("loading_stability", np.nan)
                stab_str = f", loading stability {stab:+.2f}" if not (isinstance(stab, float) and np.isnan(stab)) else ""
                lines.append(f"  {block.capitalize() + ' gauge:':<17} current {d['current_evr']*100:5.1f}%  "
                             f"vs. historical median {med_str}{stab_str}{flag}")
            hr()

    if validation is not None:
        lines.append("VALIDATION & ECONOMIC REALITY CHECK (Step 9)")
        lines.append("")

        lines.append("  DASHBOARD A -- CLASSIFICATION QUALITY")
        for axis_label, axis_key in (("Economic Regime", "quadrant"), ("Financial-Stress Regime", "vol_liquidity")):
            a = validation["A"].get(axis_key, {})
            if not a:
                continue
            lines.append(f"    {axis_label}:")
            se = a["short_episodes"]
            if se["n_transitions"]:
                lines.append(f"      Confirmed transitions: {se['n_transitions']}  "
                             f"(short episodes <{cfg.validation_false_transition_min_months}mo: "
                             f"{se['n_likely_short']}, rate={se['short_episode_rate']*100:.1f}% -- "
                             f"any short post-transition episode, not necessarily a reversion)")
            else:
                lines.append("      Confirmed transitions: 0")
            rv = a["reversions"]
            if rv["n_candidate_transitions"]:
                lines.append(f"      A->B->A reversions within {cfg.validation_false_transition_min_months}mo: "
                             f"{rv['n_reversions']}/{rv['n_candidate_transitions']} "
                             f"(rate={rv['reversion_rate']*100:.1f}% -- the genuine false-transition test)")
            agr = a["model_agreement_by_status"]
            lines.append(f"      GMM/HMM agreement by status: " + ", ".join(f"{k}={v*100:.0f}%" for k, v in agr.items()))
            lines.append("      Persistence P(stay), empirical vs. most recent fitted transition matrix:")
            for lab in a["empirical_persistence"]:
                emp = a["empirical_persistence"][lab]
                fit_p = a["fitted_persistence_most_recent_refit"].get(lab, np.nan)
                emp_str = f"{emp*100:5.1f}%" if not np.isnan(emp) else "  n/a"
                fit_str = f"{fit_p*100:5.1f}%" if not np.isnan(fit_p) else "  n/a"
                lines.append(f"        {lab:<28} empirical={emp_str}   fitted={fit_str}")
            bss = a.get("bootstrap_state_stability", {})
            if bss:
                lines.append(f"      Bootstrap state-assignment stability (illustrative dates, "
                             f"n={cfg.validation_state_stability_n_boot} block-bootstrap refits each):")
                for dt in sorted(bss.keys()):
                    v = bss[dt]
                    lines.append(f"        {dt.date()}  {str(v['assigned_state']):<28} "
                                 f"stability={v['stability_pct']*100:5.1f}%  (n_valid={v['n_valid_draws']})")
        if "recession_detection_lag" in validation["A"]:
            lines.append(f"    NBER recession detection lag (P(recession_3m) >= "
                         f"{cfg.validation_recession_detection_threshold*100:.0f}%, +/-"
                         f"{cfg.validation_detection_lag_window}mo window):")
            for _, row in validation["A"]["recession_detection_lag"].iterrows():
                lag_str = f"{row['lag_months']:+.0f}mo" if not pd.isna(row["lag_months"]) else "not detected in window"
                lines.append(f"      Onset {row['event_date'].date()}: {lag_str}")
        lines.append("")

        lines.append("  DASHBOARD B -- FORECAST QUALITY")
        for axis_label, axis_key in (("Economic Regime", "quadrant"), ("Financial-Stress Regime", "vol_liquidity")):
            b = validation["B"].get(axis_key, {})
            for k, v in b.items():
                if k.startswith("structural_h") or k.startswith("feature_conditioned_h"):
                    lines.append(f"    {axis_label} {k}: Brier={v['brier']:.4f}  LogLoss={v['log_loss']:.4f}  n={v['n']}")
                elif k.startswith("transition_hit_rate_h"):
                    auc_str = f"{v['auc']:.3f}" if not np.isnan(v['auc']) else "n/a"
                    lines.append(f"    {axis_label} {k}: AUC={auc_str}  base_rate={v['base_rate']*100:.1f}%  n={v['n']}")
                elif k.startswith("destination_accuracy_h"):
                    if v["n_total"]:
                        lines.append(f"    {axis_label} {k}: accuracy={v['accuracy']*100:.1f}% ({v['n_correct']}/{v['n_total']})")
                    else:
                        lines.append(f"    {axis_label} {k}: n/a (no confirmed transitions in-sample)")
        rb = validation["B"].get("recession", {})
        if rb:
            lines.append("    NBER recession forecast (Random Forest, walk-forward):")
            for h in (3, 6):
                m = rb.get(f"h{h}")
                if m and not np.isnan(m["precision"]):
                    lines.append(f"      {h}M: precision={m['precision']:.2f} recall={m['recall']:.2f} "
                                 f"F1={m['f1']:.2f} ROC-AUC={m['roc_auc']:.2f} PR-AUC={m['pr_auc']:.2f} "
                                 f"(n={m['n']}, n_positive={m['n_positive']})")
                ci = rb.get(f"h{h}_calibration_slope_intercept")
                if ci and not np.isnan(ci["slope"]):
                    lines.append(f"      {h}M calibration: intercept={ci['intercept']:+.2f} slope={ci['slope']:.2f} (ideal: 0, 1)")
        lines.append("")

        lines.append("  DASHBOARD C -- ROBUSTNESS")
        for label, key in (("Economic Regime", "quadrant"), ("Financial-Stress Regime", "vol_liquidity")):
            ps = validation["C"].get(f"parameter_stability_{key}", {})
            cd, td = ps.get("centroid_drift", []), ps.get("transmat_frobenius_drift", [])
            if cd or td:
                mean_drift = np.mean([np.mean(list(d.values())) for d in cd]) if cd else np.nan
                mean_tm_drift = np.mean(td) if td else np.nan
                lines.append(f"    {label} parameter stability across {len(cd)+1} refits: "
                             f"mean centroid drift={mean_drift:.3f}  mean transmat Frobenius drift={mean_tm_drift:.3f}")
        for label, key in (("Economic Regime", "state_count_sensitivity_quadrant"),
                            ("Financial-Stress Regime", "state_count_sensitivity_vol_liquidity")):
            sc = validation["C"].get(key)
            if sc is not None and not sc.empty:
                lines.append(f"    {label} state-count sensitivity (BIC, smallest-state occupancy):")
                for k, row in sc.iterrows():
                    lines.append(f"      K={k}: BIC={row['bic']:.0f}  smallest_state_occ={row['smallest_state_occupancy']*100:4.1f}%  "
                                 f"n_states<5%={int(row['n_states_below_5pct'])}")
        loo = validation["C"].get("leave_one_gauge_out_stress")
        if loo is not None and not loo.empty:
            lines.append("    Leave-one-gauge-out (Financial-Stress axis, agreement with full model):")
            for gauge, row in loo.iterrows():
                agr_str = f"{row['agreement_with_full_model']*100:5.1f}%" if not pd.isna(row['agreement_with_full_model']) else "  n/a"
                shift_str = f"{row['occupancy_shift_l1']*100:4.1f}pp" if not pd.isna(row['occupancy_shift_l1']) else "n/a"
                lines.append(f"      without {gauge:<20} agreement={agr_str}  occupancy_shift={shift_str}")
        rve = validation["C"].get("rolling_vs_expanding_stress")
        if rve:
            lines.append(f"    Rolling ({rve['window_months']}mo) vs. expanding window (Financial-Stress axis): "
                         f"agreement={rve['agreement_with_expanding']*100:5.1f}%  "
                         f"occupancy_shift={rve['occupancy_shift_l1']*100:4.1f}pp")
        sbb = validation["C"].get("structural_break_breakdown")
        if sbb is not None and not sbb.empty:
            lines.append("    Structural-break-era breakdown (named subperiods, not a fitted break test):")
            for _, row in sbb.iterrows():
                lines.append(f"      {row['period_start']} -> {row['period_end']} (n={row['n_months']}): "
                             f"modal quadrant={row['modal_quadrant']}, modal stress={row['modal_vol_liquidity']}, "
                             f"agreement={row['quadrant_model_agreement']*100:.0f}%/{row['vol_liquidity_model_agreement']*100:.0f}%")
        lines.append("")

        lines.append("  DASHBOARD D -- ECONOMIC REALITY")
        heo = validation["D"].get("held_out_economic_reality", {})
        for col in ("labour_z", "housing_z", "rates_z"):
            if col not in heo:
                continue
            lines.append(f"    Held-out variable: {col} (not used by either axis's clustering inputs)")
            for axis_label, axis_key in ((" by Quadrant", "quadrant"), (" by Vol/Liquidity", "vol_liquidity")):
                tbl = heo[col].get(axis_key)
                if tbl is not None and not tbl.empty:
                    for state, row in tbl.iterrows():
                        sig = "*" if row["hac_pvalue"] < 0.05 else " "
                        lines.append(f"     {axis_label:<18} {state:<28} mean={row['mean']:+.2f}  "
                                     f"HAC p={row['hac_pvalue']:.3f}{sig}  n={int(row['n'])}")
        for k, v in heo.get("_orderings", {}).items():
            lines.append(f"    Ordering check -- {k}: {v}")
        es = validation["D"].get("event_studies", {})
        for name, row in es.items():
            if not row.get("available"):
                lines.append(f"    {name}: not covered by this sample's classified history")
                continue
            lines.append(f"    {name} ({row['n_months']}mo covered): "
                         f"max P(Contraction)={row.get('max_contraction_probability', float('nan'))*100:.0f}%  "
                         f"max P(Crisis)={row.get('max_crisis_probability', float('nan'))*100:.0f}%")
        if "cross_asset_jennrich" in validation["D"]:
            j = validation["D"]["cross_asset_jennrich"]
            lines.append(f"    Cross-asset regime differentiation (Jennrich test, Stable Growth vs. Contraction "
                         f"correlation structure, from Section 5's defensibility suite): "
                         f"chi2={j.get('chi2', float('nan')):.2f}  p={j.get('p_value', float('nan')):.3f}")
        hr()

    if cross_asset:
        lines.append("CROSS-ASSET REGIME MAPPING (Step 10) — Regime -> Cross-Asset Behaviour")
        lines.append("  (strictly downstream of the classifier -- asset returns are never a clustering")
        lines.append("   input; see the module docstring above build_cross_asset_engine() for why)")
        lines.append("")

        lines.append("  SLEEVE-LEVEL REGIME-CONDITIONED PROFILE (contemporaneous, points 2-3)")
        for axis_label, axis_key in (("Economic axis", "quadrant"), ("Financial-Stress axis", "vol_liquidity")):
            prof = cross_asset["asset_profile"][axis_key]
            if prof.empty:
                continue
            agg = prof.groupby(["regime", "sleeve"])[["ann_return", "ann_vol", "sharpe", "max_drawdown"]].mean()
            lines.append(f"    {axis_label}:")
            lines.append("      " + agg.round(3).to_string().replace("\n", "\n      "))
        hr()

        lines.append("  CORRELATION STRUCTURE BY REGIME (point 4 -- avg pairwise corr, Eq-Credit, Eq-Rates)")
        for axis_label, axis_key in (("Economic axis", "quadrant"), ("Financial-Stress axis", "vol_liquidity")):
            cs = cross_asset["correlation_structure"][axis_key]
            lines.append(f"    {axis_label}:")
            for state, d in cs["by_state"].items():
                if d.get("insufficient_sample"):
                    lines.append(f"      {state:<28} insufficient sample (n={d['n_obs']})")
                else:
                    lines.append(f"      {state:<28} avg_pairwise={d['avg_pairwise_corr']:+.2f}  "
                                 f"eq-credit={d['corr_equities_credit']:+.2f}  "
                                 f"eq-rates={d['corr_equities_rates']:+.2f}  n={d['n_obs']}")
            for pair, j in cs["pairwise_jennrich"].items():
                if not np.isnan(j.get("chi2", np.nan)):
                    lines.append(f"        Jennrich {pair}: chi2={j['chi2']:.2f}  p={j['p_value']:.3f}")
        hr()

        lines.append("  CONDITIONAL BETA TO BENCHMARK, sleeve avg by regime (point 5)")
        for axis_label, axis_key in (("Economic axis", "quadrant"), ("Financial-Stress axis", "vol_liquidity")):
            beta = cross_asset["conditional_beta"][axis_key]
            if beta.empty:
                continue
            bagg = beta.groupby(["regime", "sleeve"])["beta"].mean()
            lines.append(f"    {axis_label}:")
            lines.append("      " + bagg.round(3).to_string().replace("\n", "\n      "))
        hr()

        lines.append("  JOINT ECONOMIC x FINANCIAL-STRESS MATRIX (point 6 -- sleeve ann. return, "
                     "insufficient-sample cells blank)")
        jm = cross_asset["joint_axis_matrix"]
        if not jm.empty:
            piv = jm.pivot_table(index=["quadrant", "vol_liquidity"], columns="sleeve", values="ann_return")
            lines.append("    " + piv.round(3).to_string().replace("\n", "\n    "))
        hr()

        lines.append("  FORWARD (POINT-IN-TIME CAUSAL) CONDITIONAL RETURNS -- E[R_t+1:t+h | S_t=k] (point 7, Economic axis)")
        for h, tbl in cross_asset["forward_conditional_returns"]["quadrant"].items():
            if tbl.empty:
                continue
            agg = tbl.groupby(["regime", "sleeve"])["mean_fwd_return"].mean()
            lines.append(f"    h={h}mo:")
            lines.append("      " + agg.round(4).to_string().replace("\n", "\n      "))
        hr()

        lines.append("  PROBABILITY-WEIGHTED / TRANSITION-AWARE EXPECTED RETURN (points 8-9, latest date only)")
        for axis_label, axis_key in (("Economic axis", "quadrant"), ("Financial-Stress axis", "vol_liquidity")):
            out = cross_asset["probability_weighted_outlook"][axis_key]
            if out.empty:
                continue
            lines.append(f"    {axis_label}:")
            lines.append("      " + out.round(4).to_string().replace("\n", "\n      "))
        hr()

        lines.append("  MAIN RESEARCH QUESTION -- does regime information change P(R)? (point 16, "
                     "Holm-Bonferroni-corrected, Economic axis)")
        info, _ = cross_asset["regime_information_content"]["quadrant"]
        if not info.empty:
            show_cols = ["sleeve", "unconditional_ann_return", "widest_regime_pair",
                         "widest_pair_mean_diff_annualized", "widest_pair_hac_pvalue",
                         "significant_after_holm_bonferroni"]
            lines.append("    " + info[show_cols].round(4).to_string().replace("\n", "\n    "))
        hr()

        lines.append("  SUB-PERIOD RANK STABILITY (point 11 -- reused as-is from Section 5's defensibility suite)")
        for regime, rho in cross_asset["rank_stability_reused"].items():
            lines.append(f"    {regime:<28} rho={rho:.3f}" if not np.isnan(rho) else f"    {regime:<28} n/a")
        hr()

        lines.append("  CROSS-ASSET OUTLOOK SUMMARY (point 15, current snapshot, sleeve level)")
        summ = cross_asset["outlook_summary"]
        if not summ.empty:
            lines.append("    " + summ.round(4).to_string().replace("\n", "\n    "))
        hr()

    if truth_df is not None:
        lines.append("SYNTHETIC-DATA SANITY CHECK (ground truth known only in synthetic mode)")
        common = regime_panel.index.intersection(truth_df.index)
        acc_q = (regime_panel.loc[common, "quadrant"] == truth_df.loc[common, "true_quadrant"]).mean()
        acc_v = (regime_panel.loc[common, "vol_liquidity"] == truth_df.loc[common, "true_vol_liq"]).mean()
        lines.append(f"  Growth x Inflation quadrant recovery accuracy: {acc_q*100:5.1f}%")
        lines.append(f"  Volatility x Liquidity state recovery accuracy: {acc_v*100:5.1f}%")
        lines.append("  (Recovery accuracy on real data cannot be computed — nobody observes the")
        lines.append("   'true' regime live. This check only validates the estimation machinery.)")
        hr()

    lines.append("KNOWN LIMITATIONS (stated explicitly, not glossed over)")
    lims = [
        "Current-vintage FRED data only when no FRED_API_KEY is set (Section 3: true "
        "ALFRED vintage history is used per-series when a key is available -- see "
        "audit_snapshot() for which series actually got true vintages vs. the "
        "release-lag fallback on any given run)",
        "Synthetic-data mode (used only when live FRED/Yahoo data is unreachable) "
        "drives GMM/HMM states AND the per-block (Section 2) PCA composite gauges "
        "from just 4 shared latent factors (growth/inflation/vol/liquidity) applied "
        "to a small representative subset of series -- e.g. the Liquidity proxies "
        "(TEDRATE, M2, Fed balance sheet/WALCL) and the credit proxy (BAA10Y) are "
        "all driven by the SAME liquidity latent factor with only a sign/scale "
        "difference, not independently simulated -- so synthetic-mode results "
        "validate the estimation machinery, not realistic cross-series independence",
        "Regime-label permutation test's real-label and shuffled-label distributions "
        "both use the same simplified PROXY allocation (same-regime historical "
        "top-Sharpe selection only, no vol targeting/leverage/transaction costs), "
        "not the full walk-forward tactical allocation -- see the Defensibility "
        "Suite's own note on this in the tearsheet",
        "Release lags are modeled (Section 3: PUBLICATION_LAG_DAYS heuristic, used "
        "only when a series falls back from true ALFRED vintage history). When a "
        "FRED_API_KEY is set, fetch_fred_vintage_series keeps the FULL observation "
        "x vintage table (every revision, indexed by both its own observation date "
        "and the date it was actually published), and reconstruct_asof_history() "
        "reconstructs the exact historical curve as it stood on any past decision "
        "date -- the genuine observation x vintage x decision-date cube, not just a "
        "single collapsed track. The production feature pipeline uses this directly: "
        "level features (rates, spreads, ratios) read the collapsed real-time "
        "'nowcast' series (_collapse_to_nowcast_series), which is exactly correct "
        "for a single current-period reading and immune to batch-revision row-"
        "ordering; every YoY-transformed feature (build_pit_feature_column, called "
        "from pca_composite_gauge_walkforward) instead reconstructs BOTH the "
        "numerator and the denominator from the vintage actually known at each "
        "walk-forward checkpoint t -- X_t^(v_t) / X_t-12^(v_t) - 1, not X_t^(v_t) / "
        "X_t-12^(v_t-12) -- so a later revision to an already-superseded observation "
        "(e.g. Jan-Apr 2020 revised alongside May in one June-2020 release) is "
        "correctly reflected in every subsequent YoY reading that references it, not "
        "just in the audit tool. reconstruct_asof_history() / audit_snapshot() "
        "remain the standing cross-check for this (asof_value vs. value columns, "
        "plus n_history_revised_asof_12m_ago) rather than the sole place the cube "
        "reconstruction is exercised",
        "Section 5: GaussianHMM refits use a small multi-restart (cfg.hmm_n_restarts, "
        "keeping the highest-training-likelihood fit, mirroring the GMM's own n_init) "
        "and cfg.hmm_n_iter=500 -- a live-data warnings audit found the single-restart, "
        "n_iter=200 version this replaced frequently failed to converge (in several "
        "refits the training log-likelihood actually DECREASED between EM iterations, "
        "a bad-local-optimum symptom). The HMM FIT DIAGNOSTICS line in this tearsheet "
        "reports the actual convergence count for THIS run -- check it before trusting "
        "a fitted transition matrix's exact values, especially early in the sample "
        "where the training window is thinnest",
        "Step 7: Financial-Stress forecast horizons are monthly (1M/3M), not the "
        "spec's suggested 1W/1M/3M -- this model's data is monthly throughout; a "
        "true weekly stress-transition forecast needs a separate, higher-frequency "
        "data pipeline",
        "Step 7: Monte Carlo path simulation holds the most-recently-fitted "
        "transition matrix fixed over the whole simulated horizon -- time-varying, "
        "feature-conditioned transition probabilities (A_t = f(X_t)) are flagged "
        "as a future investigation, not implemented",
        "Step 7: the feature-conditioned (logistic regression) forecast is trained "
        "on far fewer historical transition examples than the structural HMM-matrix "
        "forecast, especially for rare states (e.g. Overheating) -- treat its "
        "early-history and rare-state probabilities as lower-confidence",
        "Step 7: 'expected state duration -- analytic' uses the most recent refit's "
        "RAW filtered-posterior transition matrix (1/(1-p_ii)), which can show rapid "
        "flip-flopping between adjacent states (verified live: the Vol/Liquidity "
        "axis's fitted Calm self-persistence was 0.1%) even while Step 6's "
        "hysteresis engine holds the reported confirmed_regime steady through that "
        "noise for many months -- the tearsheet flags this explicitly when the two "
        "diverge, since both numbers are correct but answer different questions",
        "Step 8: multi-model disagreement (the spec's GMM/HMM/DFM-HMM/Markov-"
        "Switching/change-point ladder) is honestly limited to the 2 models this "
        "repo actually fits (GMM, HMM) -- the same limitation already stated for "
        "Step 6's model_agreement; extend the average as Section 5's deferred "
        "model ladder grows",
        "Step 8: bootstrap confidence intervals (state posterior, transition "
        "matrix, NBER probability) refit only the CURRENT training window's model "
        "once per draw, not a full walk-forward re-run per draw -- the latter is "
        "computationally infeasible at cfg.confidence_n_bootstrap draws. These CIs "
        "measure how stable TODAY's read is under resampling, not how stable every "
        "historical month's read was",
        "Step 8: cluster-fit / out-of-distribution bucketing compares the current "
        "Mahalanobis distance against this state's OWN historical distance "
        "distribution, which is itself limited to this model's ~2015-2026 sample "
        "-- a genuinely novel macro environment may not yet have a well-populated "
        "'normal' band to compare against, especially for sparsely-occupied states",
        "Step 8: 'regime evidence' context features (item 13) are shown for "
        "interpretability only -- current z-scores, not a weighted causal "
        "decomposition into 'why this regime'. No such per-quadrant attribution "
        "score is computed, by design (see classify_state_confidence's docstring "
        "on why an opaque blended score was deliberately avoided here)",
        "Step 9: vintage-sensitivity (naive current-vintage vs. point-in-time "
        "classification) is a standalone function (vintage_sensitivity_naive_vs_pit) "
        "not called from main() -- it doubles the entire live data pull and "
        "walk-forward refit, the same 'standing rerunnable tool, not a one-off "
        "claim' pattern already used for compare_state_counts",
        "Step 9: 'structural-break breakdown' reports occupancy/agreement across "
        "NAMED subperiod boundaries, not a statistically fitted break test (e.g. "
        "Chow/Bai-Perron) -- no such test is implemented here",
        "Step 9: leave-one-gauge-out and the rolling-vs-expanding comparison run "
        "only on the Financial-Stress axis -- the Economic axis has just 2 "
        "driving gauges (growth_z/inflation_z), so removing either one leaves no "
        "quadrant left to classify into",
        "Step 9: bootstrap state-assignment stability is computed at a handful of "
        "economically illustrative dates (each state's longest episode's midpoint, "
        "plus the current month), not every historical month -- a full sweep "
        "would mean len(history) x n_boot model refits",
        "Step 9: first-passage (any-time-during-horizon) forecast accuracy is not "
        "validated historically -- Step 7's Monte Carlo hitting-probability "
        "estimate is only computed for the CURRENT state at report time, not "
        "archived at every past walk-forward checkpoint; the end-horizon "
        "transition hit-rate/destination-accuracy metrics above validate the "
        "point-in-time forecast that IS archived (structural_forecast_panel)",
        "Step 10: conditional beta and the correlation/Jennrich structure are "
        "estimated within-regime over the FULL sample, not walk-forward -- they "
        "are descriptive cross-asset findings, not forecast inputs anywhere in "
        "this model, so look-ahead is not the concern it is for Sections 5-9's "
        "own regime/forecast machinery",
        "Step 10: probability-weighted and transition-aware expected returns "
        "(points 8-9) are computed only for the LATEST available date, not "
        "re-run as a full walk-forward historical series -- doing so per-asset, "
        "per-date would mean re-estimating mu_{i,k} at every historical "
        "checkpoint using only point-in-time data, the same class of expensive "
        "re-derivation Step 9 already declined to pay for twice",
        "Step 10: the joint Economic x Financial-Stress matrix and the "
        "cross-asset outlook summary use equal-weight SLEEVE proxies (mean of "
        "sleeve members), not every individual ticker or a cap/liquidity-"
        "weighted index",
        "Step 10: regime-conditional 'max_drawdown' is now computed per "
        "contiguous episode (worst single episode's peak-to-trough, via "
        "_max_drawdown_by_episode) rather than concatenating every in-regime "
        "month into one artificial continuous path -- the old concatenated-path "
        "figure is kept for reference under 'max_drawdown_concat_caveat' but "
        "should not be read as a real historical outcome; episode counts can "
        "still be thin for rare states",
        "Step 10: Holm-Bonferroni correction in the 'main research question' "
        "table is applied across every asset x regime-pair p-value tested, not "
        "just each asset's widest gap -- a pair can show a large mean difference "
        "that still doesn't survive correction once the full multiple-testing "
        "family is counted",
        "Section 6: confirm_economic_p_exit/confirm_stress_p_exit are used only "
        "to derive a required GAP between the candidate's and incumbent's "
        "posteriors (hysteresis_gap = p_enter - p_exit) -- this is a "
        "probability-margin-gap confirmation rule, not textbook two-threshold "
        "hysteresis (which would separately require the incumbent's OWN "
        "posterior to fall below an exit threshold before a challenger could "
        "take over regardless of margin). Kept as-is rather than changed to "
        "true entry/exit hysteresis, since confirm_stress_n_confirm was "
        "calibrated against this exact mechanism's live false-transition rate; "
        "switching mechanisms would need its own fresh walk-forward "
        "recalibration",
        "Section 4: vol-targeting scales exposure off the STRATEGY'S OWN "
        "trailing realized volatility (lagged), the standard approach for a "
        "vol-targeting overlay -- but it is a feedback loop by construction, "
        "since today's target-vol scaling shapes tomorrow's 'realized' vol "
        "input. This is normal for this class of strategy, not a bug, but it "
        "does mean realized vol understates the UNSCALED asset selection's own "
        "volatility",
        "Step 9: short_episode_rate flags any post-transition episode shorter "
        "than validation_false_transition_min_months, including a short "
        "A->B->C run that never reverts -- regime_reversion_rate is the "
        "stricter, genuine false-transition test (A->B->A within that window) "
        "and is reported alongside it; the two will diverge whenever short "
        "episodes are driven by a real subsequent regime change rather than "
        "reverting noise",
        "Step 10: the interactive Plotly dashboard now surfaces the Section "
        "7-10 engines (current regime posteriors, forward transition forecast, "
        "confidence/entropy over time, recession forecast, cross-asset outlook) "
        "in that order ahead of the supporting tearsheet-style charts -- "
        "previously it was built from only the regime panel, allocation, HMM "
        "fits and risk table, so its visual output lagged well behind what "
        "main() actually computes and what the printed tearsheet already "
        "reported",
    ]
    for l in lims:
        lines.append(f"  - {l}" if not l.startswith("  ") else l)
    lines.append("=" * W)

    text = "\n".join(lines)
    print(text)
    with open(f"{OUTPUT_DIR}/tearsheet.txt", "w") as f:
        f.write(text)
    return text

# MAIN ORCHESTRATION
def main(cfg: Config = CFG):
    print(">> Loading macro + asset data ...")
    macro, prices, truth_df = load_all_data(cfg)
    print(f"   {macro.shape[1]} macro series, {prices.shape[1]} assets, "
          f"{len(macro)} months: {macro.index.min().date()} -> {macro.index.max().date()}\n")

    if _LAST_VINTAGE_PANEL is not None:

        demo_date = min(pd.Timestamp("2020-03-31"), macro.index.max()).strftime("%Y-%m-%d")
        snap = audit_snapshot(demo_date, block_map=cfg.indicator_blocks())
        snap.to_csv(f"{OUTPUT_DIR}/audit_snapshot_{demo_date}.csv")
        n_vintage = int((snap["source"] == "alfred_vintage").sum())
        print(f">> Point-in-time audit snapshot as of {demo_date}: {len(snap)} series "
              f"({n_vintage} true ALFRED vintage, {len(snap) - n_vintage} lag-heuristic) "
              f"-> outputs/audit_snapshot_{demo_date}.csv\n")

    print(">> Engineering features (PCA gauges, z-scores, cross-asset stress) ...")
    features, rf_features, gauge_info = build_feature_panel(macro, prices, cfg)

    print(">> Fitting regime axes walk-forward (GMM + HMM, expanding window, "
          f"refit every {cfg.refit_every}mo) ...")
    regime_panel, quad_fit, vol_liq_fit, validator = classify_regimes(
        features, rf_features, macro, cfg)
    print(f"   {len(regime_panel)} months classified.\n")

    print(">> Running walk-forward regime-conditioned tactical allocation ...")
    alloc = walkforward_regime_allocation(prices, regime_panel, cfg)

    print(">> Building regime-conditioned risk model (VaR/CVaR, dynamic lookback/leverage) ...")
    quadrant_risk_table, pit_risk = regime_conditioned_risk_model(
        alloc["net_returns"], regime_panel, cfg)

    print(">> Running defensibility suite (bootstrap, permutation, sub-period, "
          "cost sensitivity, hit rate, Jennrich, Newey-West) ...")
    suite = run_defensibility_suite(prices, regime_panel, alloc, cfg)

    print(">> Building transition & forecasting engine (structural + feature-conditioned "
          "forecasts, transition risk, Monte Carlo simulation, recession forecast) ...")
    forecast = build_forecast_engine(features, regime_panel, quad_fit, vol_liq_fit,
                                      rf_features, macro, cfg)

    print(">> Building confidence, uncertainty & regime stability engine (posterior "
          "diagnostics, bootstrap CIs, factor uncertainty, regime evidence) ...")
    confidence = build_confidence_engine(features, regime_panel, quad_fit, vol_liq_fit,
                                          forecast, rf_features, macro, gauge_info, cfg)

    print(">> Running validation & economic reality check suite (classification quality, "
          "forecast quality, robustness, economic reality) ...")
    validation = build_validation_engine(features, regime_panel, quad_fit, vol_liq_fit,
                                          forecast, rf_features, macro, suite, cfg)

    print(">> Mapping regimes to cross-asset behaviour (regime-conditioned returns/risk, "
          "correlation structure, conditional beta, forward causal returns, probability-"
          "weighted outlook) ...")
    cross_asset = build_cross_asset_engine(prices, regime_panel, forecast, suite, cfg)

    print(">> Rendering charts ...")
    plot_regime_timeline(regime_panel, prices, cfg)
    plot_transition_heatmap(
        quad_fit["last_transition_matrix"], quad_fit["last_label_map"] or {},
        "Growth x Inflation HMM Transition Matrix (most recent refit)",
        "02_quadrant_transition_matrix.png")
    plot_transition_heatmap(
        vol_liq_fit["last_transition_matrix"], vol_liq_fit["last_label_map"] or {},
        "Volatility x Liquidity HMM Transition Matrix (most recent refit)",
        "03_volliq_transition_matrix.png")
    plot_equity_curve(alloc, prices, cfg)
    plot_regime_risk_bars(quadrant_risk_table)

    print(">> Building interactive Plotly dashboard ...")
    dashboard_path = build_plotly_dashboard(
        cfg, regime_panel, prices, alloc, quad_fit, vol_liq_fit,
        quadrant_risk_table, validator, forecast=forecast, cross_asset=cross_asset)

    print(">> Writing tearsheet ...\n")
    print_tearsheet(cfg, regime_panel, quad_fit, vol_liq_fit, validator, alloc,
                     quadrant_risk_table, pit_risk, suite, gauge_info, truth_df,
                     forecast=forecast, confidence=confidence, validation=validation,
                     cross_asset=cross_asset)

    regime_panel.to_csv(f"{OUTPUT_DIR}/regime_panel.csv")
    pit_risk.to_csv(f"{OUTPUT_DIR}/point_in_time_risk.csv")
    alloc["net_returns"].to_csv(f"{OUTPUT_DIR}/strategy_net_returns.csv")
    print(f"\n>> Done. Outputs written to {OUTPUT_DIR}/")
    print(f">> Interactive dashboard opened in your browser (temp file, not saved to outputs/): {dashboard_path}")

    return {
        "macro": macro, "prices": prices, "features": features,
        "regime_panel": regime_panel, "quad_fit": quad_fit, "vol_liq_fit": vol_liq_fit,
        "validator": validator, "alloc": alloc,
        "quadrant_risk_table": quadrant_risk_table, "pit_risk": pit_risk,
        "suite": suite, "gauge_info": gauge_info, "truth_df": truth_df,
        "forecast": forecast, "confidence": confidence, "validation": validation,
        "cross_asset": cross_asset,
    }

if __name__ == "__main__":
    results = main(CFG)