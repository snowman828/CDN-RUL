"""Uncertainty evaluation protocol for the paper (§4.5).

Computes, for each subset and both models (PhysDec-RUL vs LSTM-Cond baseline):
  - PICP: prediction interval coverage probability at nominal levels 0.5..0.95
  - PINAW: prediction interval normalized average width
  - Reliability curve: empirical coverage vs nominal coverage
  - Aggregates PhysDec over the 5 seeds (mean±std); baseline is 1 run.

Reads calibrated MC-Dropout predictions from results/physdec/mc_*.npz
(PhysDec, calibrated std) and results/mc_calib_FD00{1..4}.npz (LSTM-Cond).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "physdec" / "uncertainty_stats.json"

PHYSDEC_FILES = {
    "FD001": "mc_FD001_seed{n}.npz", "FD002": "mc_FD002_seed{n}.npz",
    "FD003": "mc_FD003_seed{n}.npz", "FD004": "mc_FD004_seed{n}.npz",
    "DS02": "mc_DS02_seed{n}.npz",
}
BASELINE_FILES = {
    "FD001": "mc_calib_FD001.npz", "FD002": "mc_calib_FD002.npz",
    "FD003": "mc_calib_FD003.npz", "FD004": "mc_calib_FD004.npz",
}
SEEDS = [42, 2024, 7, 123, 99]
NOMINAL = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]


def picp_pinaw(y, mean, std, z):
    lo, hi = mean - z * std, mean + z * std
    picp = float(np.mean((y >= lo) & (y <= hi)))
    pinaw = float(np.mean(2 * z * std) / (np.ptp(y) + 1e-9))
    return picp, pinaw


def load_npz(path: Path):
    """Load MC predictions with the CORRECT std convention per file type.

    - PhysDec files (train_physdec.py): std ALREADY calibration-scaled (k applied
      before np.savez) and have NO 'k' key → use std as-is.
    - Baseline LSTM-Cond files (mc_calibrate.py): std stored RAW with a 'k' key
      → must apply std = std * k (that file's coverage was computed as z·k·std).
    """
    d = np.load(path)
    y = d["y_test"]
    mean = d["mean"]
    std = d["std"]
    if "k" in d and abs(float(d["k"]) - 1.0) > 1e-6:
        std = std * float(d["k"])   # baseline convention: raw std + k
    return y, mean, std


def evaluate(paths: list[Path]):
    """Aggregate over seeds: per-nominal PICP/PINAW with mean±std."""
    picps = np.zeros((len(paths), len(NOMINAL)))
    pinaws = np.zeros((len(paths), len(NOMINAL)))
    for i, p in enumerate(paths):
        y, m, s = load_npz(p)
        zmap = {0.5: 0.6745, 0.6: 0.8416, 0.7: 1.0364, 0.8: 1.2816,
                0.9: 1.6449, 0.95: 1.96}
        for j, nom in enumerate(NOMINAL):
            picps[i, j], pinaws[i, j] = picp_pinaw(y, m, s, zmap[nom])
    return {
        "picp_mean": picps.mean(0).round(3).tolist(),
        "picp_std": picps.std(0, ddof=1).round(3).tolist(),
        "pinaw_mean": pinaws.mean(0).round(4).tolist(),
        "pinaw_std": pinaws.std(0, ddof=1).round(4).tolist(),
    }


def main():
    out = {}
    for ds in ["FD001", "FD002", "FD003", "FD004", "DS02"]:
        phys_paths = [ROOT / "results" / "physdec" /
                      PHYSDEC_FILES[ds].format(n=s) for s in SEEDS]
        phys_paths = [p for p in phys_paths if p.exists()]
        if not phys_paths:
            print(f"[{ds}] no physdec mc files, skip")
            continue
        phys = evaluate(phys_paths)
        out[ds] = {"physdec": phys, "n_seeds": len(phys_paths)}
        if ds in BASELINE_FILES:
            bp = ROOT / "results" / BASELINE_FILES[ds]
            if bp.exists():
                y, m, s = load_npz(bp)
                zmap = {0.5: 0.6745, 0.6: 0.8416, 0.7: 1.0364, 0.8: 1.2816,
                        0.9: 1.6449, 0.95: 1.96}
                base_picp, base_pinaw = [], []
                for nom in NOMINAL:
                    p, w = picp_pinaw(y, m, s, zmap[nom])
                    base_picp.append(round(p, 3))
                    base_pinaw.append(round(w, 4))
                out[ds]["baseline_lstmcond"] = {
                    "picp": base_picp, "pinaw": base_pinaw}

        # summary line
        p = phys
        print(f"[{ds}] PhysDec PICP @0.9: {p['picp_mean'][4]}±{p['picp_std'][4]} "
              f"| PINAW @0.9: {p['pinaw_mean'][4]}±{p['pinaw_std'][4]}")
        if "baseline_lstmcond" in out[ds]:
            b = out[ds]["baseline_lstmcond"]
            print(f"     LSTM-Cond PICP @0.9: {b['picp'][4]} | PINAW @0.9: {b['pinaw'][4]}")

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
