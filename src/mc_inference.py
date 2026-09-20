"""MC-Dropout uncertainty inference on saved condition-aware models.

Loads lstm_cond_<subset>.pt, re-encodes test data, runs N stochastic forward
passes, and appends coverage/CI-width metrics to the existing metrics json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR
from conditions import prepare_cond
from train_lstm_cond import LSTMRULCond, mc_dropout_predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True)
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--mc-samples", type=int, default=100)
    args = ap.parse_args()
    subset = args.subset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train, y_train, units, X_test, y_test, engine_ids, enc = prepare_cond(
        subset, n_clusters=args.n_clusters)
    n_feat = X_train.shape[2]

    model = LSTMRULCond(n_features=n_feat)
    model.load_state_dict(torch.load(str(MODELS_DIR / f"lstm_cond_{subset}.pt"),
                                     map_location="cpu"))
    model.to(device)

    mean, std, _ = mc_dropout_predict(model, X_test, n_samples=args.mc_samples,
                                      device=device)
    z = 1.645
    lo, hi = mean - z * std, mean + z * std
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    mean_ci_width = float(np.mean(2 * z * std))
    rmse = float(np.sqrt(np.mean((y_test - mean) ** 2)))
    print(f"[{subset}] MC-Dropout | RMSE(mean)={rmse:.3f} | 90% CI coverage={coverage:.3f} "
          f"| mean CI width={mean_ci_width:.2f} cycles")

    metrics_path = RESULTS_DIR / f"metrics_lstm_cond_{subset}.json"
    out = json.loads(metrics_path.read_text(encoding="utf-8"))
    out["mc90_coverage"] = round(coverage, 3)
    out["mc_mean_ci_width"] = round(mean_ci_width, 2)
    out["mc_rmse_mean"] = round(rmse, 3)
    out["mc_samples"] = args.mc_samples
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    np.savez(RESULTS_DIR / f"mc_pred_{subset}.npz",
             y_test=y_test, mean=mean, std=std)
    print(f"updated -> {metrics_path}")


if __name__ == "__main__":
    main()
