"""PhysDec-RUL ablation analysis: compare full / nophys / noorth / baseline against
the existing C-MAPSS baselines (XGBoost, LSTM, LSTM-Cond)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, FIGURES_DIR

PHYSDEC_DIR = RESULTS_DIR / "physdec"


def load_metrics(subset: str):
    rows = {}
    for tag in ["full", "nophys", "noorth", "baseline"]:
        p = PHYSDEC_DIR / f"metrics_{subset}_{tag}.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            rows[tag] = {"rmse": d["rmse"], "nasa_score": d["nasa_score"],
                         "mc90": d.get("mc90_coverage", None),
                         "ciw": d.get("mc_mean_ci_width", None),
                         "beta": d.get("beta_learned", None)}
    return rows


def main():
    subset = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    ab = load_metrics(subset)
    if not ab:
        print(f"no physdec metrics for {subset}")
        return

    # existing baselines
    base = {}
    p = RESULTS_DIR / f"metrics_lstm_{subset}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        base["LSTM"] = (d["rmse"], d["nasa_score"])
    p = RESULTS_DIR / f"metrics_lstm_cond_{subset}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        base["LSTM-Cond"] = (d["rmse"], d["nasa_score"])
    p = RESULTS_DIR / f"metrics_xgb_{subset}.json"
    if p.exists():
        d = json.loads(p.read_text(encoding="utf-8"))
        base["XGBoost"] = (d["rmse"], d["nasa_score"])

    print(f"=== {subset} ablation (PhysDec-RUL) ===")
    print(f"{'config':10s} {'RMSE':>8s} {'NASA':>10s} {'MC90':>7s} {'CIw':>7s} {'beta':>7s}")
    for tag, v in ab.items():
        print(f"{tag:10s} {v['rmse']:8.3f} {v['nasa_score']:10.1f} "
              f"{str(v['mc90']):>7s} {str(v['ciw']):>7s} {str(v['beta']):>7s}")
    print("\n=== existing baselines ===")
    for name, (r, s) in base.items():
        print(f"{name:10s} RMSE={r:.3f} NASA={s:.1f}")

    # ---- ablation bar chart ----
    tags = ["full", "nophys", "noorth", "baseline"]
    rmses = [ab[t]["rmse"] for t in tags if t in ab]
    labels = [t for t in tags if t in ab]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, rmses, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    for b, r in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.15, f"{r:.2f}",
                ha="center", fontsize=10)
    ax.set_ylabel("RMSE (cycles)")
    ax.set_title(f"{subset} · PhysDec-RUL ablation")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"physdec_ablation_{subset}.png", dpi=130)
    plt.close(fig)
    print(f"\nsaved -> {FIGURES_DIR / f'physdec_ablation_{subset}.png'}")

    # ---- save combined json ----
    out = {"subset": subset, "ablation": ab,
           "baselines": {k: {"rmse": v[0], "nasa_score": v[1]}
                         for k, v in base.items()}}
    p = PHYSDEC_DIR / f"ablation_{subset}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"combined -> {p}")


if __name__ == "__main__":
    main()
