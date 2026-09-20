"""Common constants and paths for the equipment degradation RUL prediction project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# C-MAPSS column names: unit, cycle, 3 operational settings, 26 sensor measurements
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 27)]

# Standard sensor selection (14 informative sensors; the rest are constant or near-constant)
SELECTED_SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                    "s13", "s14", "s15", "s17", "s20", "s21"]

# RUL cap: standard practice in the literature (Heimes 2008; Saxena 2008) — piecewise-linear target
RUL_CAP = 125

# Sliding-window length for sequence models
WINDOW_SIZE = 30

DATASET_INFO = {
    "FD001": dict(train_engines=100, test_engines=100, conditions=1, faults=1),
    "FD002": dict(train_engines=260, test_engines=259, conditions=6, faults=1),
    "FD003": dict(train_engines=100, test_engines=100, conditions=1, faults=2),
    "FD004": dict(train_engines=249, test_engines=248, conditions=6, faults=2),
}

SEED = 42
