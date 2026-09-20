"""Comparison visualization: baseline LSTM vs condition-aware LSTM, plus calibrated
MC-Dropout confidence intervals for multi-condition subsets."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, FIGURES_DIR
from metrics import rmse, nasa_score
import xgboost as xgb
import torch

plt.rcParams.update({"font.size": 11, "figure.dpi": 130})


def load_cond_predictions(subset: str):
    """Point predictions of the condition-aware LSTM on the test set."""
    from train_lstm_cond import LSTMRULCond
    from conditions import prepare_cond
    # read the n_clusters the model was actually trained with
    meta = json.loads((RESULTS_DIR / f"metrics_lstm_cond_{subset}.json").read_text(encoding="utf-8"))
    n_clusters = meta.get("n_clusters", 6)
    X_train, y_train, units, X_test, y_test, engine_ids, enc = prepare_cond(
        subset, n_clusters=n_clusters)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = LSTMRULCond(n_features=X_train.shape[2])
    m.load_state_dict(torch.load(str(RESULTS_DIR / "models" / f"lstm_cond_{subset}.pt"),
                                 map_location="cpu"))
    m.to(device).eval()
    with torch.no_grad():
        y_pred = m(torch.tensor(X_test).to(device)).cpu().numpy()
    return y_test, y_pred.clip(min=0)


def comparison_plots(subset: str):
    """Scatter comparison: XGBoost vs plain LSTM vs condition-aware LSTM."""
    y_test, y_cond = load_cond_predictions(subset)
    from data import prepare_sequences
    from train_lstm import LSTMRUL
    from common import SELECTED_SENSORS
    X_train, y_train, u, X_test, y_test_p, eids = prepare_sequences(subset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = LSTMRUL(n_features=len(SELECTED_SENSORS))
    m.load_state_dict(torch.load(str(RESULTS_DIR / "models" / f"lstm_{subset}.pt"),
                                 map_location="cpu"))
    m.to(device).eval()
    with torch.no_grad():
        y_base = m(torch.tensor(X_test).to(device)).cpu().numpy().clip(min=0)
    y_test = np.asarray(y_test_p)

    # XGBoost predictions
    from data import prepare_tabular
    X_t, y_t, X_te, y_te = prepare_tabular(subset, lookback=30)
    xb = xgb.XGBRegressor()
    xb.load_model(str(RESULTS_DIR / "models" / f"xgb_{subset}.json"))
    y_xgb = xb.predict(X_te).clip(min=0)

    for name, pred in [("XGBoost", y_xgb), ("LSTM", y_base), ("LSTM-Cond", y_cond)]:
        r, s = rmse(y_test, pred), nasa_score(y_test, pred)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.scatter(y_test, pred, s=14, alpha=0.65, c="#1f77b4", edgecolors="none")
        lims = [0, max(y_test.max(), pred.max()) + 10]
        ax.plot(lims, lims, "r--", lw=1.2, label="perfect prediction")
        ax.set_xlabel("True RUL (cycles)")
        ax.set_ylabel("Predicted RUL (cycles)")
        ax.set_title(f"{subset} · {name}\nRMSE={r:.2f}  NASA Score={s:.0f}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"scatter_{subset}_{name}.png")
        plt.close(fig)
        print(f"  {subset} {name}: RMSE={r:.3f} NASA={s:.0f}")


def _xgb_preds(subset: str):
    from data import prepare_tabular
    X_t, y_t, X_te, y_te = prepare_tabular(subset, lookback=30)
    m = xgb.XGBRegressor()
    m.load_model(str(RESULTS_DIR / "models" / f"xgb_{subset}.json"))
    return m.predict(X_te).clip(min=0)


def uncertainty_plot(subset: str, n_engines: int = 30):
    """Predicted RUL with calibrated 90% CI error bars vs true RUL."""
    data = np.load(RESULTS_DIR / f"mc_calib_{subset}.npz")
    y_test, mean, std, k = data["y_test"], data["mean"], data["std"], float(data["k"])
    z = 1.645
    order = np.argsort(y_test)
    y_test, mean, std = y_test[order][:n_engines], mean[order][:n_engines], std[order][:n_engines]
    lo, hi = mean - z * k * std, mean + z * k * std

    fig, ax = plt.subplots(figsize=(11, 5.5))
    xs = np.arange(n_engines)
    ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt="o", ms=4,
                capsize=2.5, elinewidth=1, color="#1f77b4", ecolor="#7f7f7f",
                label="predicted RUL ± 90% CI")
    ax.plot(xs, y_test, "r--", lw=1.4, label="true RUL")
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    cov = float(np.mean((y_test >= lo) & (y_test <= hi)))
    ax.set_title(f"{subset} · LSTM-Cond with calibrated 90% CI "
                 f"(coverage on shown={cov:.2f}, k={k:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"uncertainty_{subset}.png")
    plt.close(fig)
    print(f"  uncertainty plot -> uncertainty_{subset}.png")


def summary_table():
    rows = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        base = json.load(open(RESULTS_DIR / f"metrics_lstm_{ds}.json"))
        cond = json.load(open(RESULTS_DIR / f"metrics_lstm_cond_{ds}.json"))
        xgb = json.load(open(RESULTS_DIR / f"metrics_xgb_{ds}.json"))
        rows.append({
            "subset": ds,
            "xgb_rmse": xgb["rmse"], "xgb_score": xgb["nasa_score"],
            "lstm_rmse": base["rmse"], "lstm_score": base["nasa_score"],
            "cond_rmse": cond["rmse"], "cond_score": cond["nasa_score"],
            "cond_cov": cond.get("mc_calib_coverage", cond.get("mc90_coverage", "-")),
            "cond_ciw": cond.get("mc_calib_mean_ci_width", cond.get("mc_mean_ci_width", "-")),
        })
    out = RESULTS_DIR / "summary_cond_all.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n{'subset':6s} {'XGB RMSE':>9s} {'LSTM RMSE':>10s} {'Cond RMSE':>10s} "
          f"{'XGB Score':>10s} {'LSTM Score':>11s} {'Cond Score':>11s} {'CI cov':>7s}")
    for r in rows:
        print(f"{r['subset']:6s} {r['xgb_rmse']:9.2f} {r['lstm_rmse']:10.2f} {r['cond_rmse']:10.2f} "
              f"{r['xgb_score']:10.1f} {r['lstm_score']:11.1f} {r['cond_score']:11.1f} {str(r['cond_cov']):>7s}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        comparison_plots(ds)
    for ds in ["FD002", "FD004"]:
        uncertainty_plot(ds)
    summary_table()
