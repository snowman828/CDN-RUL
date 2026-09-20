"""Multi-seed baselines, run on the same seeds as the proposed model.

Runs each baseline 5 seeds (same seeds as PhysDec), aggregates mean±std,
and performs PAIRED t-tests (baseline vs PhysDec per seed) so the
"no difference" claims are statistically valid.

Baselines:
  FD001/FD003 -> LSTM       (train_lstm.py)
  FD002/FD004 -> LSTM-Cond  (train_lstm_cond.py, n_clusters=6)
  DS02        -> LSTM       (train_ncmapss.py)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
SEEDS = [42, 2024, 7, 123, 99]

BASELINES = {
    "FD001": ("lstm", "train_lstm.py", {}),
    "FD002": ("lstm_cond", "train_lstm_cond.py", {"--n-clusters": "6", "--mc-samples": "30"}),
    "FD003": ("lstm", "train_lstm.py", {}),
    "FD004": ("lstm_cond", "train_lstm_cond.py", {"--n-clusters": "6", "--mc-samples": "30"}),
    "DS02": ("lstm", "train_ncmapss.py", {"--model": "lstm"}),
}


def run_baseline_seed(subset: str, seed: int) -> dict:
    _, script, extra = BASELINES[subset]
    if subset == "DS02":
        # train_ncmapss.py takes --model/--seed (no --subset)
        cmd = [PY, str(ROOT / "src" / script), "--model", "lstm",
               "--seed", str(seed), "--batch-size", "128"]
    else:
        cmd = [PY, str(ROOT / "src" / script), "--subset", subset,
               "--seed", str(seed)]
        for k, v in extra.items():
            cmd += [k, v]

    # metrics file paths differ by baseline family
    family = BASELINES[subset][0]
    if subset == "DS02":
        out = ROOT / "results" / "ncmapss" / f"metrics_{family}_{subset}_seed{seed}.json"
    else:
        out = ROOT / "results" / f"metrics_{family}_{subset}_seed{seed}.json"

    # Same rule as run_multiseed.run_seed: the exit code is not the signal, the
    # artifact is. This machine intermittently returns 0xC0000409 during
    # interpreter teardown, after the metrics are on disk, so a non-zero code
    # alone proves nothing -- but reading the previous run's file when this run
    # wrote nothing would silently inject an old number into the baseline table.
    before = out.stat().st_mtime_ns if out.exists() else None
    r = subprocess.run(cmd, capture_output=True, text=True)

    if not out.exists() or out.stat().st_mtime_ns == before:
        raise RuntimeError(
            f"{family} seed {seed} of {subset} produced no fresh metrics file "
            f"(rc={r.returncode}). Refusing to read a stale result. "
            f"stderr tail: {r.stderr[-400:]!r}")
    if r.returncode not in (0, 127, 3221226505):
        print(f"  [seed {seed}] note: rc={r.returncode} but {out.name} is fresh")
    return json.loads(out.read_text(encoding="utf-8"))


def load_physdec(subset: str) -> list[float]:
    """PhysDec per-seed RMSEs from multiseed_stats.json."""
    ms = json.loads((ROOT / "results" / "physdec" / "multiseed_stats.json")
                    .read_text(encoding="utf-8"))
    return [s["rmse"] for s in ms[subset]["per_seed"]]


def main():
    subsets = sys.argv[1:] or list(BASELINES.keys())
    out = {}
    for ds in subsets:
        rows = [run_baseline_seed(ds, s) for s in SEEDS]
        b_rmses = [r["rmse"] for r in rows]
        b_scores = [r["nasa_score"] for r in rows]
        p_rmses = load_physdec(ds)

        # paired t-test on per-seed RMSE differences (baseline - physdec)
        diffs = np.array(b_rmses) - np.array(p_rmses)
        t, p = stats.ttest_rel(b_rmses, p_rmses)
        better = "PhysDec better" if np.mean(diffs) > 0 else "baseline better"

        res = {
            "subset": ds, "n_seeds": len(SEEDS),
            "baseline_rmse_mean": round(float(np.mean(b_rmses)), 3),
            "baseline_rmse_std": round(float(np.std(b_rmses, ddof=1)), 3),
            "baseline_score_mean": round(float(np.mean(b_scores)), 1),
            "baseline_score_std": round(float(np.std(b_scores, ddof=1)), 1),
            "physdec_rmse_mean": round(float(np.mean(p_rmses)), 3),
            "paired_t": round(float(t), 3), "paired_p": round(float(p), 4),
            "mean_diff": round(float(np.mean(diffs)), 3),
            "verdict": better,
            "per_seed": [{"seed": s, "baseline_rmse": br, "physdec_rmse": pr}
                         for s, br, pr in zip(SEEDS, b_rmses, p_rmses)],
        }
        out[ds] = res
        print(f"[{ds}] baseline {res['baseline_rmse_mean']}±{res['baseline_rmse_std']} "
              f"vs PhysDec {res['physdec_rmse_mean']} | "
              f"t={res['paired_t']} p={res['paired_p']} ({res['verdict']})")

    p = ROOT / "results" / "physdec" / "baseline_multiseed_stats.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
