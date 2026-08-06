"""Central experiment configuration: paths, splits, horizons, hyperparameters.

Everything the experiment needs is parameterised here so the comparison is
identical for XGBoost and LightGBM, and for real vs synthetic data.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"

RAW_HOURLY_CSV = DATA_DIR / "hourly_counts.csv"
RAW_LOCATIONS_CSV = DATA_DIR / "sensor_locations.csv"

# Time-respecting split (fixed calendar dates, identical for every model).
# test is the most recent block; the last `max(horizons)` hours of each sensor
# are naturally excluded because targets are shifted into the future.
VAL_START = "2025-11-01"   # train  <  val  <  test
TEST_START = "2026-03-01"  # test = [TEST_START, end of data)

HORIZONS = (1, 6, 24)      # forecast horizons in hours, direct multi-horizon
ALPHAS = (0.1, 0.5, 0.9)   # quantiles; 80% interval = [q0.1, q0.9]
POINT_OBJECTIVE = "point"  # standard L2 regression objective

# Sensors whose first record falls on/after this date are "short history"
# (sparse training data); the rest are "full history".
SHORT_HISTORY_THRESHOLD = "2025-06-01"

MAPE_FLOOR = 5.0  # MAPE denominator guard: max(y, floor); WMAPE is also reported
SEED = 42

N_ESTIMATORS = 1000   # cap; early stopping on validation decides the real size
EARLY_STOPPING = 50   # rounds without val improvement
EVAL_ROUND = 10       # check val metric every N rounds (faster large-data runs)

# Shared, deliberately plain hyperparameters (no per-framework tuning search).
XGB_PARAMS = {
    "objective": None,  # set per model (reg:squarederror / reg:quantileerror)
    "eval_metric": None,
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "nthread": 20,
    "seed": SEED,
}

LGB_PARAMS = {
    "objective": None,  # set per model (regression / quantile)
    "metric": None,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "lambda_l2": 1.0,
    "num_threads": 20,
    "seed": SEED,
    "verbosity": -1,
}

FEATURES = [
    "hour", "dow", "month", "is_weekend", "is_public_holiday",
    "lag_1", "lag_24", "lag_168",
    "roll_24_mean", "roll_168_mean", "roll_672_mean",
    "location_code",
]
CATEGORICAL_FEATURE = "location_code"
TARGET_PREFIX = "target_"

SYNTHETIC = dict(
    n_sensors=8,
    start="2021-01-01",
    end="2026-06-30",
    short_ids=(30, 31, 32),          # deliberately short-history sensors
    short_starts=("2025-07-01", "2025-10-01", "2026-01-01"),
)
