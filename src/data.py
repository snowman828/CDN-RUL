"""Data loading, preprocessing, and windowing for C-MAPSS turbofan degradation data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import COLUMNS, SELECTED_SENSORS, RUL_CAP, WINDOW_SIZE, DATA_DIR


def load_dataset(subset: str = "FD001") -> dict:
    """Load train/test/RUL raw data for a C-MAPSS subset."""
    train = pd.read_csv(DATA_DIR / f"train_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    test = pd.read_csv(DATA_DIR / f"test_{subset}.txt", sep=r"\s+", header=None, names=COLUMNS)
    rul = pd.read_csv(DATA_DIR / f"RUL_{subset}.txt", sep=r"\s+", header=None, names=["RUL"])
    return {"train": train, "test": test, "rul": rul}


def add_rul_targets(train: pd.DataFrame, rul_cap: int = RUL_CAP) -> pd.DataFrame:
    """Compute per-cycle RUL for training trajectories (run-to-failure)."""
    df = train.copy()
    df["RUL"] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=rul_cap)  # piecewise-linear target
    return df


def fit_scaler(train: pd.DataFrame, sensors: list = SELECTED_SENSORS) -> StandardScaler:
    """Fit z-score scaler on training sensor columns only (no leakage)."""
    scaler = StandardScaler()
    scaler.fit(train[sensors])
    return scaler


def apply_scaler(df: pd.DataFrame, scaler: StandardScaler, sensors: list = SELECTED_SENSORS) -> pd.DataFrame:
    """Transform sensor columns of a dataframe with a fitted scaler."""
    out = df.copy()
    out[sensors] = scaler.transform(df[sensors])
    return out


def make_windows(df: pd.DataFrame, sensors: list = SELECTED_SENSORS,
                 window: int = WINDOW_SIZE, stride: int = 1):
    """Build (N, window, n_features) sliding-window sequences with aligned RUL targets.

    Returns X (float32 array), y (float32 array, one target per window = last cycle RUL),
    and unit id per window (for leakage-free engine-level splits).
    """
    X, y, units = [], [], []
    for unit, grp in df.groupby("unit"):
        vals = grp[sensors].values
        ruls = grp["RUL"].values
        n = len(grp)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            X.append(vals[start:start + window])
            y.append(ruls[start + window - 1])
            units.append(unit)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
            np.asarray(units, dtype=np.int64))


def make_test_windows(test: pd.DataFrame, scaler: StandardScaler,
                      sensors: list = SELECTED_SENSORS, window: int = WINDOW_SIZE):
    """Build windowed test sequences: last `window` cycles per engine (prediction at truncation point)."""
    test_s = apply_scaler(test, scaler, sensors)
    X = []
    engine_ids = []
    for unit, grp in test_s.groupby("unit"):
        vals = grp[sensors].values
        engine_ids.append(unit)
        if len(vals) >= window:
            X.append(vals[-window:])
        else:
            # pad shorter trajectories by repeating the first row
            pad = np.repeat(vals[:1], window - len(vals), axis=0)
            X.append(np.vstack([pad, vals]))
    return np.asarray(X, dtype=np.float32), np.asarray(engine_ids)


def rolling_features(df: pd.DataFrame, sensors: list = SELECTED_SENSORS,
                     lookback: int = 30):
    """Aggregate rolling statistics per engine for classical ML (XGBoost) baselines."""
    frames = []
    for _, grp in df.groupby("unit"):
        g = grp.sort_values("cycle").copy()
        feats = {}
        for s in sensors:
            win = g[s]
            roll = win.rolling(lookback, min_periods=1)
            feats[f"{s}_mean"] = roll.mean().values
            feats[f"{s}_std"] = roll.std().values
            feats[f"{s}_min"] = roll.min().values
            feats[f"{s}_max"] = roll.max().values
            feats[f"{s}_last"] = win.values
            feats[f"{s}_slope"] = win.diff().fillna(0).values
        feat_df = pd.DataFrame(feats, index=g.index)
        feat_df["unit"] = g["unit"].values
        feat_df["cycle"] = g["cycle"].values
        feat_df["RUL"] = g["RUL"].values if "RUL" in g.columns else np.nan
        frames.append(feat_df)
    return pd.concat(frames, axis=0)


def prepare_tabular(subset: str = "FD001", lookback: int = 30):
    """End-to-end tabular pipeline: load -> scale -> rolling features for train & test."""
    data = load_dataset(subset)
    train = add_rul_targets(data["train"])
    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    train_feat = rolling_features(train_s, lookback=lookback).dropna(subset=["RUL"])

    # Test: scale with train scaler, then take LAST lookback-window stats per engine
    test_s = apply_scaler(data["test"], scaler)
    test_rows = []
    for unit, grp in test_s.groupby("unit"):
        tail = grp.sort_values("cycle").tail(lookback).copy()
        feats = {}
        for s in SELECTED_SENSORS:
            win = tail[s]
            feats[f"{s}_mean"] = win.mean()
            feats[f"{s}_std"] = win.std()
            feats[f"{s}_min"] = win.min()
            feats[f"{s}_max"] = win.max()
            feats[f"{s}_last"] = win.iloc[-1]
            feats[f"{s}_slope"] = win.diff().fillna(0).mean()
        feats["unit"] = unit
        test_rows.append(feats)
    test_feat = pd.DataFrame(test_rows)
    y_test = data["rul"]["RUL"].values.astype(np.float32)

    feature_cols = [c for c in train_feat.columns if c not in ("unit", "cycle", "RUL")]
    return (train_feat[feature_cols].values.astype(np.float32),
            train_feat["RUL"].values.astype(np.float32),
            test_feat[feature_cols].values.astype(np.float32),
            y_test)


def prepare_sequences(subset: str = "FD001", window: int = WINDOW_SIZE):
    """End-to-end sequence pipeline: load -> scale -> sliding windows for train & test."""
    data = load_dataset(subset)
    train = add_rul_targets(data["train"])
    scaler = fit_scaler(train)
    train_s = apply_scaler(train, scaler)
    X_train, y_train, train_units = make_windows(train_s, window=window)
    X_test, engine_ids = make_test_windows(data["test"], scaler, window=window)
    y_test = data["rul"]["RUL"].values.astype(np.float32)
    return X_train, y_train, train_units, X_test, y_test, engine_ids
