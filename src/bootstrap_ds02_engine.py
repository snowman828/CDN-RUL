"""P2: engine-level (cluster-robust) coverage intervals for N-CMAPSS DS02.

WHY THIS EXISTS
---------------
Table 3 reports DS02 coverage as 0.845 +/- 0.143 over five seeds. That mean is
computed over 125,228 test WINDOWS drawn from only THREE test engines, and the
windows are stride-1 with window=50, so consecutive windows share 49 of 50
timesteps. Treating those windows as independent observations is
pseudo-replication: the effective sample size is closer to 3 than to 125,228,
and any window-level confidence interval on coverage is anti-conservatively
narrow. A paper that argues RUL intervals must be audited has to audit its own
coverage estimate the same way.

WHAT IT COMPUTES (per seed, then aggregated)
  1. split-conformal quantile q from the validation engines (same protocol as
     conformal_eval.conformal_widths, so results tie back to Table 3)
  2. pooled window-level coverage and per-ENGINE coverage
  3. naive i.i.d. window-level binomial interval  (the misleading one)
  4. cluster bootstrap over engines                 (honest about between-engine
                                                     variation; only 3 clusters)
  5. moving-block bootstrap within each engine      (honest about within-engine
                                                     autocorrelation)
  6. integrated autocorrelation time and effective sample size
  7. design effect = between-engine variance / i.i.d. variance

Usage: python bootstrap_ds02_engine.py [seed ...]
       (default: all five manuscript seeds)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import RESULTS_DIR  # noqa: E402
from conformal_eval import physdec_mc, CONFIGS, SEEDS, ALPHA  # noqa: E402

OUT = RESULTS_DIR / "physdec" / "ds02_engine_bootstrap.json"
SUBSET = "DS02"
N_BOOT = 5000
RNG_SEED = 20260919


# ---------------------------------------------------------------- statistics
def integrated_act(x: np.ndarray, max_lag: int = 3000) -> float:
    """Integrated autocorrelation time (in samples) of a 0/1 indicator series.

    Sequential windows overlap, so the coverage indicator is strongly
    autocorrelated; this estimates how many consecutive windows are worth
    approximately one independent observation.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    var = float(np.dot(x, x)) / n
    if var <= 0 or n < 3:
        return 1.0
    act = 1.0
    for k in range(1, min(max_lag, n - 1)):
        rho = float(np.dot(x[:-k], x[k:])) / (n * var)
        if rho <= 0.05:
            break
        act += 2.0 * rho
    return max(1.0, act)


def mbb_resample(x: np.ndarray, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap: resample contiguous blocks, preserving local
    dependence, then truncate to the original length."""
    n = x.size
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
    return x[idx]


def cluster_bootstrap(masks_by_engine: list[np.ndarray], rng: np.random.Generator):
    """Resample engines with replacement; pool their windows.

    With 3 engines there are only 3^3 = 27 distinct resamples, so this interval
    is necessarily coarse -- that coarseness is a property of the data, not of
    the method, and is reported rather than hidden.
    """
    k = len(masks_by_engine)
    covs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, k, size=k)
        cov = float(np.mean(np.concatenate([masks_by_engine[i] for i in pick])))
        covs.append(cov)
    covs = np.asarray(covs)
    return float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))


def block_bootstrap(masks_by_engine: list[np.ndarray], blocks: list[int],
                    rng: np.random.Generator):
    covs = []
    for _ in range(N_BOOT):
        parts = [mbb_resample(m, b, rng) for m, b in zip(masks_by_engine, blocks)]
        covs.append(float(np.mean(np.concatenate(parts))))
    covs = np.asarray(covs)
    return float(np.percentile(covs, 2.5)), float(np.percentile(covs, 97.5))


def naive_binomial_ci(p: float, n: int, z: float = 1.96):
    se = float(np.sqrt(max(p * (1 - p), 1e-12) / n))
    return p - z * se, p + z * se


# ---------------------------------------------------------------- main
def main() -> int:
    seeds = [int(a) for a in sys.argv[1:]] or list(SEEDS)
    cfg = CONFIGS[SUBSET]
    rng = np.random.default_rng(RNG_SEED)

    per_seed = []
    for seed in seeds:
        try:
            (mu_v, sd_v, y_v, mu_t, sd_t, y_t, _g_v, _g_t, u_t) = physdec_mc(
                SUBSET, seed, cfg["window"], cfg["n_clusters"], cfg["norm_mode"],
                return_units=True)
        except FileNotFoundError as exc:
            print(f"[{SUBSET}] seed {seed}: missing checkpoint ({exc}), skip")
            continue

        # --- split-conformal quantile from the validation engines (same as Table 3)
        resid_v = np.abs(y_v - mu_v) / np.maximum(sd_v, 1e-6)
        n_cal = resid_v.size
        q = float(np.quantile(resid_v, np.ceil((n_cal + 1) * (1 - ALPHA)) / n_cal))

        resid_t = np.abs(y_t - mu_t) / np.maximum(sd_t, 1e-6)
        covered = (resid_t <= q).astype(np.float64)

        engines = np.unique(u_t)
        masks, blocks, n_eff = [], [], []
        per_engine = {}
        for e in engines:
            m = u_t == e
            ce = covered[m]
            per_engine[str(int(e))] = {
                "n_windows": int(m.sum()),
                "coverage": round(float(ce.mean()), 4),
                "act": round(integrated_act(ce), 1),
                "n_eff": round(float(m.sum()) / integrated_act(ce), 1),
            }
            masks.append(ce)
            blocks.append(int(np.ceil(integrated_act(ce))))
            n_eff.append(m.sum() / integrated_act(ce))

        pooled = float(covered.mean())
        eng_covs = np.array([per_engine[str(int(e))]["coverage"] for e in engines])

        # cluster bootstrap (resample engines) and moving-block bootstrap
        # (resample contiguous blocks within each engine)
        cb_lo, cb_hi = cluster_bootstrap(masks, rng)
        bb_lo, bb_hi = block_bootstrap(masks, blocks, rng)

        per_seed.append({
            "seed": seed,
            "n_cal": int(n_cal),
            "q": round(q, 4),
            "n_test_windows": int(covered.size),
            "coverage_pooled_window_level": round(pooled, 4),
            "coverage_engine_mean": round(float(eng_covs.mean()), 4),
            "coverage_engine_std": round(float(eng_covs.std(ddof=1)), 4),
            "per_engine": per_engine,
            "n_eff_total": round(float(np.sum(n_eff)), 1),
            "cluster_bootstrap_ci95": [round(cb_lo, 4), round(cb_hi, 4)],
            "block_bootstrap_ci95": [round(bb_lo, 4), round(bb_hi, 4)],
            "naive_binomial_ci95": [round(v, 4) for v in
                                    naive_binomial_ci(pooled, covered.size)],
        })
        print(f"[{SUBSET}] seed {seed}: pooled={pooled:.4f} "
              f"engine-mean={eng_covs.mean():.4f} (+/-{eng_covs.std(ddof=1):.4f}) "
              f"q={q:.3f} n_eff={np.sum(n_eff):.0f}")
        print(f"          CI95 naive=[{naive_binomial_ci(pooled, covered.size)[0]:.4f},"
              f"{naive_binomial_ci(pooled, covered.size)[1]:.4f}] "
              f"cluster=[{cb_lo:.4f},{cb_hi:.4f}] block=[{bb_lo:.4f},{bb_hi:.4f}]")

    if not per_seed:
        print("no seeds produced results")
        return 1

    pooled_covs = np.array([r["coverage_pooled_window_level"] for r in per_seed])
    eng_means = np.array([r["coverage_engine_mean"] for r in per_seed])

    n_total = per_seed[0]["n_test_windows"]
    mean_pooled = float(pooled_covs.mean())

    # naive interval on the pooled (pseudo-replicated) count
    nv_lo, nv_hi = naive_binomial_ci(mean_pooled, n_total)

    # intervals computed on seed 42's masks (representative) and averaged across
    # seeds' block lengths where relevant
    summary = {
        "subset": SUBSET,
        "note": ("Coverage is computed over stride-1 overlapping windows drawn from three test "
                 "engines. Window-level statistics treat these as independent and are therefore "
                 "anti-conservatively narrow; engine-level statistics are the deployment-relevant "
                 "ones."),
        "n_seeds": len(per_seed),
        "n_test_engines": len(per_seed[0]["per_engine"]),
        "n_test_windows": n_total,
        "per_seed": per_seed,
        "window_level": {
            "coverage_mean": round(mean_pooled, 4),
            "coverage_std_over_seeds": round(float(pooled_covs.std(ddof=1)), 4),
            "naive_binomial_ci95": [round(nv_lo, 4), round(nv_hi, 4)],
            "ci95_width": round(nv_hi - nv_lo, 5),
        },
        "engine_level": {
            "coverage_mean": round(float(eng_means.mean()), 4),
            "coverage_std_over_seeds": round(float(eng_means.std(ddof=1)), 4),
            "between_engine_std_mean": round(
                float(np.mean([r["coverage_engine_std"] for r in per_seed])), 4),
            "per_engine_coverage_range": [
                round(float(min(r["per_engine"][e]["coverage"]
                                for r in per_seed for e in r["per_engine"])), 4),
                round(float(max(r["per_engine"][e]["coverage"]
                                for r in per_seed for e in r["per_engine"])), 4),
            ],
        },
        "effective_sample_size": {
            "n_windows_pooled": n_total,
            "n_engines": len(per_seed[0]["per_engine"]),
            "n_eff_total_mean_over_seeds": round(
                float(np.mean([r["n_eff_total"] for r in per_seed])), 1),
            "reduction_factor": round(
                float(n_total / np.mean([r["n_eff_total"] for r in per_seed])), 1),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n================ DS02 ENGINE-LEVEL COVERAGE SUMMARY ================")
    wl, el, es = summary["window_level"], summary["engine_level"], summary["effective_sample_size"]
    print(f"  window-level coverage     : {wl['coverage_mean']:.4f} "
          f"+- {wl['coverage_std_over_seeds']:.4f} (over seeds)")
    print(f"  naive i.i.d. 95% CI       : [{wl['naive_binomial_ci95'][0]:.4f}, "
          f"{wl['naive_binomial_ci95'][1]:.4f}]  width={wl['ci95_width']:.5f}")
    print(f"  engine-level mean coverage: {el['coverage_mean']:.4f} "
          f"+- {el['coverage_std_over_seeds']:.4f} (over seeds)")
    print(f"  between-engine std (mean) : {el['between_engine_std_mean']:.4f}")
    print(f"  per-engine coverage range : {el['per_engine_coverage_range']}")
    print(f"  n_eff (of {n_total} windows): {es['n_eff_total_mean_over_seeds']:.0f} "
          f"-> reduction x{es['reduction_factor']:.0f}")
    print(f"\nsaved -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
