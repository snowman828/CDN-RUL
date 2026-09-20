"""Recalibrate MC-Dropout uncertainty: scale std so 90% CI hits target coverage on a
held-out calibration set (validation engines, same seed split as training), then
apply the calibrated width to the test set."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR, SEED, WINDOW_SIZE
from conditions import prepare_cond
from data import load_dataset, add_rul_targets
from train_lstm_cond import LSTMRULCond, mc_dropout_predict


def engine_split(train_df: pd.DataFrame, frac: float = 0.2):
    engine_pool = train_df["unit"].unique()
    rng = np.random.default_rng(SEED)
    val_engines = set(rng.choice(engine_pool, size=int(frac * len(engine_pool)), replace=False))
    return val_engines


def calibrate(subset: str, n_clusters: int = 6, mc_samples: int = 100,
              target_coverage: float = 0.9, z: float = 1.645):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, y_train, units, X_test, y_test, engine_ids, enc = prepare_cond(
        subset, n_clusters=n_clusters)
    n_feat = X_train.shape[2]

    model = LSTMRULCond(n_features=n_feat)
    model.load_state_dict(torch.load(str(MODELS_DIR / f"lstm_cond_{subset}.pt"),
                                     map_location="cpu"))
    model.to(device)

    # --- calibration set: validation engines (last-window per engine) ---
    data = load_dataset(subset)
    train_full = add_rul_targets(data["train"])
    val_engines = engine_split(train_full)
    cal_df = train_full[train_full["unit"].isin(val_engines)].copy()
    X_cal, cal_eng_ids = enc.test_windows(cal_df, window=WINDOW_SIZE)
    # true RUL at last cycle of each calibration engine, aligned to cal_eng_ids order
    rul_by_unit = cal_df.groupby("unit")["RUL"].last().to_dict()
    cal_y = np.array([rul_by_unit[u] for u in cal_eng_ids], dtype=np.float32)

    mean_c, std_c, _ = mc_dropout_predict(model, X_cal, n_samples=mc_samples,
                                          device=device)
    mean_c, std_c = mean_c.clip(min=0), np.maximum(std_c, 1e-6)

    # --- find std multiplier k achieving target coverage on calibration set ---
    def coverage_for(k):
        lo = mean_c - z * k * std_c
        hi = mean_c + z * k * std_c
        return float(np.mean((cal_y >= lo) & (cal_y <= hi)))

    lo_k, hi_k = 0.5, 8.0
    for _ in range(60):  # bisection
        mid = 0.5 * (lo_k + hi_k)
        if coverage_for(mid) < target_coverage:
            lo_k = mid
        else:
            hi_k = mid
    k = 0.5 * (lo_k + hi_k)
    print(f"[{subset}] calibration: k={k:.3f} -> cal coverage {coverage_for(k):.3f} "
          f"(target {target_coverage})")

    # --- apply calibrated width to test set ---
    mean_t, std_t, _ = mc_dropout_predict(model, X_test, n_samples=mc_samples,
                                          device=device)
    mean_t, std_t = mean_t.clip(min=0), np.maximum(std_t, 1e-6)
    lo, hi = mean_t - z * k * std_t, mean_t + z * k * std_t
    coverage_test = float(np.mean((y_test >= lo) & (y_test <= hi)))
    width_test = float(np.mean(2 * z * k * std_t))
    rmse_t = float(np.sqrt(np.mean((y_test - mean_t) ** 2)))
    print(f"[{subset}] calibrated test: coverage={coverage_test:.3f} | "
          f"mean 90% CI width={width_test:.2f} cycles | RMSE(mean)={rmse_t:.3f}")

    metrics_path = RESULTS_DIR / f"metrics_lstm_cond_{subset}.json"
    out = json.loads(metrics_path.read_text(encoding="utf-8"))
    out["mc_calib_k"] = round(k, 3)
    out["mc_calib_coverage"] = round(coverage_test, 3)
    out["mc_calib_mean_ci_width"] = round(width_test, 2)
    out["mc_calib_target"] = target_coverage
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    np.savez(RESULTS_DIR / f"mc_calib_{subset}.npz",
             y_test=y_test, mean=mean_t, std=std_t, k=k)
    print(f"updated -> {metrics_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--mc-samples", type=int, default=100)
    args = ap.parse_args()
    calibrate(args.subset, n_clusters=args.n_clusters, mc_samples=args.mc_samples)
