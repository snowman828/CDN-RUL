"""N-CMAPSS DS02 data loading and windowing (memory-friendly).

Format (N-CMAPSS_DS02-006.h5):
  W:    operative conditions  ['alt','Mach','TRA','T2']                (4 cols)
  X_s:  measured signals      [T24,T30,T48,T50,P15,P2,P21,P24,
                               Ps30,P40,P50,Nf,Nc,Wf]                  (14 cols)
  X_v:  virtual sensors       [T40,P30,P45,W21,...] (use first 2)      (2 cols)
  Y:    RUL [in cycles]
  A:    auxiliary             ['unit','cycle','Fc','hs'] (keep 'unit')
  T:    engine health parameters (theta, not used as input)

Benchmark split (Chao et al. 2021): train units {2,5,10,16,18,20},
test units {11,14,15}. Standard protocol: 0.1Hz subsampling (sampling=10),
window length 50.
"""
import numpy as np
import pandas as pd
import h5py
from sklearn.preprocessing import StandardScaler

from common import DATA_DIR, RUL_CAP

NCMAPSS_FILE = DATA_DIR / "N-CMAPSS_DS02-006.h5"
TRAIN_UNITS = [2.0, 5.0, 10.0, 16.0, 18.0, 20.0]
TEST_UNITS = [11.0, 14.0, 15.0]
W_COLS = ["alt", "Mach", "TRA", "T2"]
XS_COLS = ["T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
           "Ps30", "P40", "P50", "Nf", "Nc", "Wf"]
XV2_COLS = ["T40", "P30"]  # first 2 virtual sensors (reference convention)
FEATURE_COLS = W_COLS + XS_COLS + XV2_COLS  # 20 features


def load_arrays():
    """Stream arrays from h5 with per-unit subsampling (memory-friendly).

    Returns dev/test dataframes with FEATURE_COLS + unit + RUL, subsampled
    per unit at rate `sampling` (default 10 -> 0.1 Hz).
    """
    def read_subsampled(hdf, suffix, sampling):
        A = hdf[f"A{suffix}"][:]
        unit_ids = A[:, 0]  # 'unit' is first aux column (A: unit, cycle, Fc, hs)
        parts = []
        for u in np.unique(unit_ids):
            idx = np.where(unit_ids == u)[0][::sampling]
            if len(idx) == 0:
                continue
            W = hdf[f"W{suffix}"][idx]
            X_s = hdf[f"X_s{suffix}"][idx]
            X_v = hdf[f"X_v{suffix}"][idx][:, :2]
            Y = hdf[f"Y{suffix}"][idx]
            df = pd.DataFrame(np.column_stack([W, X_s, X_v]),
                              columns=FEATURE_COLS)
            df["unit"] = u
            df["RUL"] = Y
            parts.append(df)
        return pd.concat(parts, axis=0).reset_index(drop=True)

    with h5py.File(NCMAPSS_FILE, "r") as hdf:
        dev = read_subsampled(hdf, "_dev", 10)
        test = read_subsampled(hdf, "_test", 10)
    return dev, test


def prepare_ncmapss(window: int = 50, stride: int = 1, rul_cap: int | None = None,
                    cache: str | None = None):
    """Full pipeline -> sliding windows for train (dev) & test.

    Returns (X_train, y_train, train_units, X_test, y_test, engine_ids).
    RUL targets are clipped to rul_cap (same piecewise-linear convention).
    """
    if cache:
        import os
        if os.path.exists(cache):
            d = np.load(cache)
            return (d["X_train"], d["y_train"], d["train_units"],
                    d["X_test"], d["y_test"], d["engine_ids"])

    dev, test = load_arrays()

    # scale on train only (no leakage)
    scaler = StandardScaler().fit(dev[FEATURE_COLS])
    dev_s = dev.copy()
    dev_s[FEATURE_COLS] = scaler.transform(dev[FEATURE_COLS])
    test_s = test.copy()
    test_s[FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

    # RUL cap on train targets
    dev_s["RUL"] = dev_s["RUL"].clip(upper=rul_cap)

    # --- train windows (per unit) ---
    X_train, y_train, units = [], [], []
    for u, grp in dev_s.groupby("unit"):
        vals = grp[FEATURE_COLS].values
        ruls = grp["RUL"].values
        n = len(grp)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            X_train.append(vals[start:start + window])
            y_train.append(ruls[start + window - 1])
            units.append(u)
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    train_units = np.asarray(units, dtype=np.float64)

    # --- test: ALL windows per engine (N-CMAPSS protocol: evaluate every window) ---
    X_test, y_test, test_units = [], [], []
    for u, grp in test_s.groupby("unit"):
        vals = grp[FEATURE_COLS].values
        ruls = grp["RUL"].values
        n = len(grp)
        if n < window:
            continue
        for start in range(0, n - window + 1, stride):
            X_test.append(vals[start:start + window])
            y_test.append(ruls[start + window - 1])
            test_units.append(u)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    test_units = np.asarray(test_units, dtype=np.float64)

    if cache:
        np.savez_compressed(cache, X_train=X_train, y_train=y_train,
                            train_units=train_units, X_test=X_test,
                            y_test=y_test, engine_ids=test_units)
    return X_train, y_train, train_units, X_test, y_test, test_units


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    X_train, y_train, units, X_test, y_test, eids = prepare_ncmapss()
    print(f"train windows: {X_train.shape} | test windows: {X_test.shape}")
    print(f"train engines: {np.unique(units)} | test engines: {np.unique(eids)}")
    print(f"y_train: mean={y_train.mean():.1f} std={y_train.std():.1f} "
          f"[{y_train.min()}, {y_train.max()}]")
    print(f"y_test : mean={y_test.mean():.1f} std={y_test.std():.1f} "
          f"[{y_test.min()}, {y_test.max()}]")
