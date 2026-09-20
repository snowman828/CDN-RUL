# -*- coding: utf-8 -*-
"""Conformal under-coverage diagnosis: why the intervals are too narrow.

WHAT THIS PRODUCES
Section 4.6 of the manuscript reports a two-layer diagnosis of the coverage shortfall.
Layer one: the conformity quantile is estimated on a calibration split whose residuals
are systematically smaller than those met at test time -- the ratio of test to
calibration mean absolute studentized residual is 1.42 on FD002 (1.32-1.55 across five
seeds) and 1.67 on FD004 (1.57-1.87), every seed on both subsets. Layer two: the
predictive standard deviation tracks residual magnitude only imperfectly, |r| vs sigma
correlating 0.49-0.56 on FD002 and 0.49-0.53 on FD004.

Every number in that paragraph comes from this script's output,
`results/physdec/undercov_diagnosis.json`. The file is shipped alongside the code
precisely so the paragraph can be checked without re-training: the ratios are
`test_E_r / cal_E_r` per seed, which the JSON stores directly.

WHY IT WAS NOT SHIPPED UNTIL NOW
It lived in the project's internal tooling directory with the results landing in a
scratch outputs/ folder, so the manuscript's central mechanistic claim -- the one that
turns the coverage table from a negative result into an actionable one -- had no
released artifact behind it, while the cover letter stated that all results are fully
reproducible. The numbers were correct; the chain from them to a reader was not.

USAGE
    python src/conformal_undercov_diagnosis.py              # step 1, test-side, numpy only
    python src/conformal_undercov_diagnosis.py --with-val   # step 2, adds calibration
                                                            # comparison; needs the model
                                                            # checkpoints and a GPU
Step 1 runs in seconds. Step 2 is the one that produces the ratios quoted in the paper.

PATHS
Resolved relative to this file, never absolute: the earlier version hardcoded
D:/hermes/projects/... in two places, so it could only ever have run on one machine.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "results" / "physdec"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 2024, 7, 123, 99]
ALPHA = 0.10


def load_npz(subset, seed=None):
    if subset == "DS02":
        if seed is None:
            return None
        p = R / "physdec" / f"mc_DS02_seed{seed}.npz"
    else:
        p = (R / f"mc_pred_{subset}.npz" if seed is None
             else R / f"mc_pred_{subset}_seed{seed}.npz")
    return np.load(p) if p.exists() else None


def diagnose(z, tag):
    y, mu, sd = z["y_test"], z["mean"], z["std"]
    r = np.abs(y - mu) / np.maximum(sd, 1e-6)
    out = {
        "tag": tag, "n": int(len(y)),
        "studentized_resid": {
            "mean": float(r.mean()), "median": float(np.median(r)),
            "q90": float(np.quantile(r, 0.9)), "q95": float(np.quantile(r, 0.95)),
            "frac_gt1": float(np.mean(r > 1.0)),
            "frac_gt1_645": float(np.mean(r > 1.645)),
            "max": float(r.max()),
        },
        "sigma": {"mean": float(sd.mean()),
                  "cv": float(sd.std() / max(sd.mean(), 1e-9)),
                  "q10_q90_ratio": float(np.quantile(sd, 0.9)
                                         / max(np.quantile(sd, 0.1), 1e-9))},
        "corr_absresid_sigma": float(np.corrcoef(np.abs(y - mu), sd)[0, 1]),
        "coverage_z1_645": float(np.mean((y >= mu - 1.645 * sd)
                                         & (y <= mu + 1.645 * sd))),
    }
    q33, q67 = np.quantile(y, 1 / 3), np.quantile(y, 2 / 3)
    for nm, m in (("low_RUL", y <= q33), ("mid_RUL", (y > q33) & (y <= q67)),
                  ("high_RUL", y > q67)):
        if m.sum() == 0:
            continue
        inband = (y >= mu - 1.645 * sd) & (y <= mu + 1.645 * sd)
        out[f"cov_z1645_{nm}"] = float(inband[m].mean())
    return out


def main():
    with_val = "--with-val" in sys.argv
    results = {"step": 2 if with_val else 1, "subsets": {}}
    for subset in ["FD001", "FD002", "FD003", "FD004", "DS02"]:
        seeds = [None] if subset in ("FD001", "FD003") else SEEDS
        res = []
        for s in seeds:
            z = load_npz(subset, s)
            if z is None:
                continue
            res.append(diagnose(z, f"{subset}" + (f"_s{s}" if s else "_legacy")))
        if not res:
            continue
        agg = {"n_runs": len(res)}
        for k in res[0]:
            if isinstance(res[0][k], dict):
                agg[k] = {kk: [r[k][kk] for r in res] for kk in res[0][k]}
            else:
                agg[k] = [r[k] for r in res]
        results["subsets"][subset] = agg

        e = np.mean([r["studentized_resid"]["mean"] for r in res])
        covs = [r["coverage_z1_645"] for r in res]
        print(f"[{subset}] E[r] mean={e:.2f} (ideal ~0.80) | z=1.645 coverage "
              f"{np.mean(covs):.3f}+-{np.std(covs):.3f} | corr(|res|,sigma) mean="
              f"{np.mean([r['corr_absresid_sigma'] for r in res]):.3f} | sigma CV mean="
              f"{np.mean([r['sigma']['cv'] for r in res]):.3f}")

    if with_val:
        # The ratios quoted in Section 4.6 come from here: for each seed,
        # test_E_r / cal_E_r. Both components are stored, so the division is checkable.
        sys.path.insert(0, str(ROOT / "src"))
        from conformal_eval import physdec_mc, conformal_widths, CONFIGS  # noqa: E402
        print("\n[step 2] calibration vs test residual comparison")
        val_test = {}
        for ds in ["FD002", "FD004", "DS02"]:
            cfg = CONFIGS[ds]
            for seed in SEEDS:
                try:
                    mu_v, sd_v, y_v, mu_t, sd_t, y_t, _, _ = physdec_mc(
                        ds, seed, cfg["window"], cfg["n_clusters"], cfg["norm_mode"],
                        window_level=(ds == "DS02"))
                except FileNotFoundError:
                    continue
                rv = np.abs(y_v - mu_v) / np.maximum(sd_v, 1e-6)
                rt = np.abs(y_t - mu_t) / np.maximum(sd_t, 1e-6)
                n = len(rv)
                q = np.quantile(rv, np.ceil((n + 1) * (1 - ALPHA)) / n)
                c, w, _ = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
                val_test.setdefault(ds, []).append({
                    "seed": seed, "n_cal": int(n), "q": float(q),
                    "cal_E_r": float(rv.mean()), "test_E_r": float(rt.mean()),
                    "ratio_test_over_cal": float(rt.mean() / rv.mean()),
                    "cal_q90": float(np.quantile(rv, 0.9)),
                    "test_q90": float(np.quantile(rt, 0.9)),
                    "test_frac_r_gt_q": float(np.mean(rt > q)),
                    "coverage": c, "width": w})
                print(f"[{ds}] s{seed}: n_cal={n} q={q:.3f} | E[r]: "
                      f"cal={rv.mean():.2f} test={rt.mean():.2f} "
                      f"ratio={rt.mean()/rv.mean():.3f} | coverage={c:.3f}")
        results["val_test"] = val_test

    outp = OUT / "undercov_diagnosis.json"
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\nsaved -> {outp}")
    print("\nverdict rules:"
          "\n  H1 sigma under-estimated : test E[r] >> cal E[r] and test r>q >> 0.10"
          "\n  H2 distribution shift    : cal/test residual location or scale differ"
          "\n  H3 sigma uninformative   : corr(|res|,sigma) < 0.2 and sigma CV < 0.5")


if __name__ == "__main__":
    main()
