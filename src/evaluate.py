"""Evaluation & visualization: aggregate metrics across models/subsets, plot prediction curves."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, FIGURES_DIR
from data import prepare_tabular, prepare_sequences
from metrics import rmse, nasa_score
import xgboost as xgb
import torch

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def load_model_predictions(subset: str, model: str):
    """Re-run inference to get per-engine predictions + true RUL."""
    if model == "xgb":
        X_train, y_train, X_test, y_test = prepare_tabular(subset, lookback=30)
        m = xgb.XGBRegressor()
        m.load_model(str(RESULTS_DIR / "models" / f"xgb_{subset}.json"))
        y_pred = m.predict(X_test).clip(min=0)
    else:
        from train_lstm import LSTMRUL
        from common import SELECTED_SENSORS
        X_train, y_train, train_units, X_test, y_test, engine_ids = prepare_sequences(subset)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = LSTMRUL(n_features=len(SELECTED_SENSORS))
        m.load_state_dict(torch.load(str(RESULTS_DIR / "models" / f"lstm_{subset}.pt"),
                                     map_location="cpu"))
        m.to(device).eval()
        with torch.no_grad():
            y_pred = m(torch.tensor(X_test).to(device)).cpu().numpy()
        y_pred = y_pred.clip(min=0)
    return y_test, y_pred


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, subset: str,
                     model: str, save_path: Path):
    """Scatter: predicted vs true RUL, with y=x line."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, alpha=0.65, c="#1f77b4", edgecolors="none")
    lims = [0, max(y_true.max(), y_pred.max()) + 10]
    ax.plot(lims, lims, "r--", lw=1.2, label="perfect prediction")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"{subset} · {model.upper()}\nRMSE={rmse(y_true, y_pred):.2f}  "
                 f"NASA Score={nasa_score(y_true, y_pred):.0f}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray, subset: str,
                            model: str, save_path: Path):
    """Histogram of prediction errors (pred - true)."""
    err = np.asarray(y_pred) - np.asarray(y_true)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(err, bins=30, color="#2ca02c", alpha=0.8, edgecolor="white")
    ax.axvline(0, color="red", ls="--", lw=1.2)
    ax.set_xlabel("Prediction error (pred − true, cycles)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{subset} · {model.upper()} error distribution  (mean={err.mean():+.2f}, "
                 f"std={err.std():.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main():
    subsets = sys.argv[1:] or ["FD001", "FD002", "FD003", "FD004"]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for ds in subsets:
        for model in ["xgb", "lstm"]:
            try:
                y_true, y_pred = load_model_predictions(ds, model)
            except FileNotFoundError as e:
                print(f"skip {ds}/{model}: {e}")
                continue
            r, s = rmse(y_true, y_pred), nasa_score(y_true, y_pred)
            summary.append({"subset": ds, "model": model, "rmse": round(r, 3),
                            "nasa_score": round(s, 1)})
            plot_predictions(y_true, y_pred, ds, model,
                             FIGURES_DIR / f"scatter_{ds}_{model}.png")
            plot_error_distribution(y_true, y_pred, ds, model,
                                    FIGURES_DIR / f"errhist_{ds}_{model}.png")
            print(f"  {ds} {model}: RMSE={r:.3f} NASA={s:.1f}")

    summary_path = RESULTS_DIR / "summary_all.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary_path}")


if __name__ == "__main__":
    main()
