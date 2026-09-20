"""Is the shift-aware remedy real, or is it just widening the intervals?

THE QUESTION THIS ANSWERS
test_section46_remedy.py showed, on the worst seed, coverage rising from 0.690 to
0.830 under density-ratio reweighting. That number alone proves nothing, because
any intervention that inflates the conformity quantile raises coverage. Section
4.6 already claims the principled route beats "a wider ad-hoc scaling factor",
and that claim has never been tested either.

So the control is a PLACEBO: inflate the global quantile by whatever constant
factor is needed to reach the same coverage as the shift-weighted quantile, and
compare the two intervals at matched coverage. If reweighting is doing something
real, it should reach that coverage with NARROWER intervals -- it is spending its
widening on the points that need it. If the widths are the same, the remedy is a
renaming of the thing it was supposed to replace, and the honest write-up says so.

Also reported, because a weak classifier producing a large effect deserves an
explanation rather than a shrug:
  - the weight distribution (odds ratios can span a wide range even when the
    thresholded accuracy is near chance)
  - the correlation between the test-likeness score and the validation residual,
    which is the mechanism the reweighting is supposed to exploit
  - the effective sample size of the reweighted calibration set,
    n_eff = (sum w)^2 / sum(w^2), since reweighting always costs calibration data
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from test_section46_remedy import weighted_quantile, CFG_DS02, ALPHA  # noqa: E402


def feats(X):
    X = np.asarray(X, dtype=np.float64)
    return np.concatenate([X.mean(axis=1), X.std(axis=1), X[:, -1, :], X[:, 0, :]], axis=1)


def run_seed(seed: int) -> dict:
    from conformal_eval import physdec_mc, conformal_widths
    from train_physdec import load_data

    mu_v, sd_v, y_v, mu_t, sd_t, y_t, g_v, g_t, u_t = physdec_mc(
        "DS02", seed, CFG_DS02["window"], CFG_DS02["n_clusters"], CFG_DS02["norm_mode"],
        return_units=True)
    r_v = np.abs(y_v - mu_v) / np.maximum(sd_v, 1e-8)
    r_t = np.abs(y_t - mu_t) / np.maximum(sd_t, 1e-8)
    n = r_v.size
    lvl = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)

    cov_a, width_a, q_global = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
    span = float(np.ptp(y_t)) + 1e-9
    width_of = lambda q: float(np.mean(2 * q * sd_t) / span)

    # ---- shift-aware weights
    d = load_data("DS02", window=CFG_DS02["window"], seed=seed)
    Fv, Ft = feats(d["X_val"]), feats(d["X_test"])
    mean, std = Fv.mean(0), Fv.std(0) + 1e-8
    Fv, Ft = (Fv - mean) / std, (Ft - mean) / std
    X = np.vstack([Fv, Ft]); y = np.r_[np.zeros(len(Fv)), np.ones(len(Ft))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); half = len(y) // 2
    tr, te = idx[:half], idx[half:]
    w = np.zeros(Fv.shape[1] + 1)
    Xtr = np.c_[np.ones(len(tr)), X[tr]]; ytr = y[tr]
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-Xtr @ w))
        w -= 1.5 * (Xtr.T @ (p - ytr) / len(ytr) + 1e-4 * w)
    acc = float(np.mean((1 / (1 + np.exp(-np.c_[np.ones(len(te)), X[te]] @ w)) > 0.5) == y[te]))
    p_v = np.clip(1 / (1 + np.exp(-np.c_[np.ones(len(Fv)), Fv] @ w)), 1e-6, 1 - 1e-6)
    wts = (p_v / (1 - p_v)) * (len(Ft) / len(Fv)); wts = wts / wts.mean()

    q_w = weighted_quantile(r_v, wts, lvl)
    cov_c = float(np.mean(r_t <= q_w))

    # ---- placebo: inflate the global quantile to the same coverage
    lo, hi = 0.1, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if float(np.mean(r_t <= mid)) < cov_c:
            lo = mid
        else:
            hi = mid
    q_placebo = (lo + hi) / 2

    n_eff = float(wts.sum() ** 2 / np.sum(wts ** 2))
    return {
        "seed": seed,
        "current": {"cov": float(cov_a), "width": float(width_a), "q": float(q_global)},
        "shift": {"cov": cov_c, "width": width_of(q_w), "q": float(q_w)},
        "placebo": {"cov": float(np.mean(r_t <= q_placebo)), "width": width_of(q_placebo),
                    "q": float(q_placebo)},
        "domain_clf_acc": acc,
        "weight_stats": {"min": float(wts.min()), "p50": float(np.median(wts)),
                         "p99": float(np.percentile(wts, 99)), "max": float(wts.max())},
        "corr_score_residual": float(np.corrcoef(p_v, r_v)[0, 1]),
        "calib_n_eff": n_eff,
        "calib_n": int(n),
    }


def main() -> int:
    seeds = [int(s) for s in sys.argv[1:]] or [42, 2024, 7, 123, 99]
    rows = []
    for s in seeds:
        try:
            r = run_seed(s)
        except Exception as e:
            print(f"  seed {s}: FAILED {type(e).__name__}: {e}")
            continue
        rows.append(r)
        print(f"  seed {s:>4} | current {r['current']['cov']:.4f} (w {r['current']['width']:.4f}) "
              f"| shift {r['shift']['cov']:.4f} (w {r['shift']['width']:.4f}) "
              f"| placebo {r['placebo']['cov']:.4f} (w {r['placebo']['width']:.4f}) "
              f"| clf {r['domain_clf_acc']:.3f} corr {r['corr_score_residual']:+.3f} "
              f"n_eff {r['calib_n_eff']:.0f}/{r['calib_n']}")

    if rows:
        print()
        sw = np.mean([r["shift"]["width"] for r in rows])
        pw = np.mean([r["placebo"]["width"] for r in rows])
        print(f"  MEAN width at matched coverage: shift {sw:.4f}  vs  placebo {pw:.4f}  "
              f"-> shift is {(1 - sw / pw) * 100:+.1f}% relative width")
        print(f"  MEAN shift coverage {np.mean([r['shift']['cov'] for r in rows]):.4f} "
              f"(std {np.std([r['shift']['cov'] for r in rows]):.4f})  vs  "
              f"current {np.mean([r['current']['cov'] for r in rows]):.4f} "
              f"(std {np.std([r['current']['cov'] for r in rows]):.4f})")

    out = ROOT / "results" / "physdec" / "remedy_mechanism.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
