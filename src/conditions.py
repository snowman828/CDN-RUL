"""Condition-aware preprocessing for multi-condition C-MAPSS subsets (FD002/FD004).

Multi-condition subsets have 6 distinct operational regimes; sensor distributions
differ across regimes, so global z-scoring mixes them. Fix: cluster the operational
settings with KMeans, scale sensors per-cluster (condition-wise normalization), and
feed the standardized op-settings themselves as extra input channels.

Leakage-safe design: ALL scalers (op-scaler, KMeans, per-cluster sensor scalers)
are fitted ONCE on the training set and stored; test data is transformed with them.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from common import COLUMNS, SELECTED_SENSORS, RUL_CAP, WINDOW_SIZE, DATA_DIR, SEED
from data import load_dataset, add_rul_targets


OP_COLS = ["op1", "op2", "op3"]


class ConditionEncoder:
    """Fits op-scaler + KMeans + per-cluster sensor scalers on train; transforms train/test."""

    def __init__(self, n_clusters: int = 6, seed: int = SEED):
        self.n_clusters = n_clusters
        self.seed = seed
        self.op_scaler = None
        self.km = None
        self.cluster_scalers = None  # list[StandardScaler] per cluster
        self.sensors = SELECTED_SENSORS

    def fit(self, df: pd.DataFrame):
        """Fit everything on the TRAINING dataframe only."""
        self.op_scaler = StandardScaler().fit(df[OP_COLS])
        op_z = self.op_scaler.transform(df[OP_COLS])
        self.km = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        self.km.fit(op_z)
        labels = self.km.predict(op_z)
        self.cluster_scalers = []
        for c in range(self.n_clusters):
            mask = labels == c
            scaler = StandardScaler()
            if mask.sum() > 1:
                scaler.fit(df.loc[mask, self.sensors])
            else:
                scaler.fit(df[self.sensors])  # degenerate cluster fallback
            self.cluster_scalers.append(scaler)
        return self

    def _encode(self, df: pd.DataFrame):
        """Encode one dataframe: (scaled_sensors, op_scaled, cluster_onehot)."""
        op_z = self.op_scaler.transform(df[OP_COLS])
        labels = self.km.predict(op_z)
        scaled = np.empty((len(df), len(self.sensors)), dtype=np.float32)
        for c in range(self.n_clusters):
            mask = labels == c
            if mask.sum() == 0:
                continue
            scaled[mask] = self.cluster_scalers[c].transform(df.loc[mask, self.sensors])
        onehot = np.zeros((len(df), self.n_clusters), dtype=np.float32)
        onehot[np.arange(len(df)), labels] = 1.0
        return scaled, op_z.astype(np.float32), onehot

    def feature_dim(self, include_op: bool, include_cluster: bool) -> int:
        d = len(self.sensors)
        if include_op:
            d += len(OP_COLS)
        if include_cluster:
            d += self.n_clusters
        return d

    def window_features(self, df: pd.DataFrame, window: int = WINDOW_SIZE,
                        stride: int = 1, include_op: bool = True,
                        include_cluster: bool = True):
        """Sliding windows with per-step features; returns X, y, unit ids."""
        scaled, op_z, onehot = self._encode(df)
        parts = [scaled]
        if include_op:
            parts.append(op_z)
        if include_cluster:
            parts.append(onehot)
        X_all = np.concatenate(parts, axis=1).astype(np.float32)  # (N, D)

        X, y, units = [], [], []
        ruls = df["RUL"].values
        ids = df["unit"].values
        for unit_id in pd.unique(ids):
            idx = np.where(ids == unit_id)[0]
            n = len(idx)
            if n < window:
                continue
            for start in range(0, n - window + 1, stride):
                X.append(X_all[idx[start:start + window]])
                y.append(ruls[idx[start + window - 1]])
                units.append(unit_id)
        return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32),
                np.asarray(units, dtype=np.int64))

    def test_windows(self, test: pd.DataFrame, window: int = WINDOW_SIZE,
                     include_op: bool = True, include_cluster: bool = True):
        """Last-window per test engine with train-fitted scalers."""
        scaled, op_z, onehot = self._encode(test)
        parts = [scaled]
        if include_op:
            parts.append(op_z)
        if include_cluster:
            parts.append(onehot)
        X_all = np.concatenate(parts, axis=1).astype(np.float32)

        X, engine_ids = [], []
        ids = test["unit"].values
        for unit_id in pd.unique(ids):
            idx = np.where(ids == unit_id)[0]
            engine_ids.append(unit_id)
            tail = X_all[idx[-window:]]
            if len(tail) < window:
                pad = np.repeat(tail[:1], window - len(tail), axis=0)
                tail = np.vstack([pad, tail])
            X.append(tail)
        return np.asarray(X, dtype=np.float32), np.asarray(engine_ids)

    def cluster_sizes(self, df: pd.DataFrame) -> dict:
        labels = self.km.predict(self.op_scaler.transform(df[OP_COLS]))
        counts = pd.Series(labels).value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.items()}


def prepare_cond(subset: str = "FD001", window: int = WINDOW_SIZE,
                 n_clusters: int = 6, include_op: bool = True,
                 include_cluster: bool = True):
    """End-to-end condition-aware pipeline: fit on train, encode train+test."""
    data = load_dataset(subset)
    train = add_rul_targets(data["train"])
    enc = ConditionEncoder(n_clusters=n_clusters).fit(train)
    X_train, y_train, units = enc.window_features(
        train, window=window, include_op=include_op, include_cluster=include_cluster)
    X_test, engine_ids = enc.test_windows(
        data["test"], window=window, include_op=include_op, include_cluster=include_cluster)
    y_test = data["rul"]["RUL"].values.astype(np.float32)
    return X_train, y_train, units, X_test, y_test, engine_ids, enc
