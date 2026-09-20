"""Group-conditional split conformal prediction for RUL (§4.x, A1 contribution).

Global split conformal (conformal_eval.py) guarantees only MARGINAL coverage.
Group-conditional conformal computes a per-condition quantile q_k on the
validation engines belonging to each operating-regime group (GMM/KMeans
clusters over op-settings), then applies q_k to test windows of that group.

This yields coverage guarantees conditional on the operating regime — the
natural requirement for condition-aware RUL, and a methodological gap in the
RUL literature. Comparison reported: global vs group-conditional coverage
per subset (overall and per-group).

Usage: python conformal_group.py [FD001 FD002 ...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR
from train_physdec import load_data
from conformal_eval import physdec_mc, CONFIGS, SEEDS, ALPHA


def condition_labels(subset: str, seed: int, n_clusters: int, norm_mode: str,
                     window: int):
    """Assign each window (val & test) a condition-group label via the same
    GMM/KMeans used by the model. Returns (val_labels, test_labels)."""
    data = load_data(subset, window=window, seed=seed)
    n_conds = data["n_conds"]
    cond_tr = data["X_train"][..., :n_conds].reshape(-1, n_conds)
    if norm_mode == "soft" and n_clusters > 0:
        from sklearn.mixture import GaussianMixture
        gm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                             random_state=seed, max_iter=200).fit(cond_tr)
        return gm.predict(data["X_val"][..., :n_conds].mean(axis=1)), \
               gm.predict(data["X_test"][..., :n_conds].mean(axis=1))
    if n_clusters > 0:
        from sklearn.cluster import KMeans as SKMeans
        km = SKMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(cond_tr)
        return km.predict(data["X_val"][..., :n_conds].mean(axis=1)), \
               km.predict(data["X_test"][..., :n_conds].mean(axis=1))
    # no clustering -> single global group
    n_val, n_test = len(data["X_val"]), len(data["X_test"])
    return np.zeros(n_val, dtype=int), np.zeros(n_test, dtype=int)


def group_conformal_widths(mu_cal, sd_cal, y_cal, g_cal,
                           mu_test, sd_test, y_true, g_test,
                           alpha=ALPHA, min_group=30):
    """Group-conditional conformal: per-group studentized quantile q_k.

    Returns (overall_cov, overall_width, per_group dict, q_by_group, q_global).
    per_group reports, for each test group:
      - n, coverage_group (using group-specific q_k)
      - coverage_marginal (using the GLOBAL q on this group only) — the key
        comparison showing group-conditional converges per-group coverage.
    Groups with < min_group calibration samples fall back to the global q."""
    resid = np.abs(y_cal - mu_cal) / np.maximum(sd_cal, 1e-6)
    # global quantile as fallback
    n = len(resid)
    q_global = np.quantile(resid, np.ceil((n + 1) * (1 - alpha)) / n)

    # initialize q for every group that appears in EITHER cal or test
    groups = np.unique(np.concatenate([g_cal, g_test]))
    q_by_group = {int(k): float(q_global) for k in groups}
    per_group = {}
    for k in groups:
        idx = g_cal == k
        nk = idx.sum()
        if nk >= min_group:
            qk = np.quantile(resid[idx],
                             np.ceil((nk + 1) * (1 - alpha)) / nk)
            q_by_group[int(k)] = float(qk)
        # per-group coverage on test (use q from q_by_group, incl. fallback)
        t_idx = g_test == k
        if t_idx.sum() > 0:
            qk = q_by_group[int(k)]
            lo = mu_test[t_idx] - qk * sd_test[t_idx]
            hi = mu_test[t_idx] + qk * sd_test[t_idx]
            # marginal counterpart: same group, GLOBAL q
            lo_m = mu_test[t_idx] - q_global * sd_test[t_idx]
            hi_m = mu_test[t_idx] + q_global * sd_test[t_idx]
            per_group[int(k)] = {
                "n": int(t_idx.sum()),
                "coverage": float(np.mean((y_true[t_idx] >= lo)
                                          & (y_true[t_idx] <= hi))),
                "coverage_marginal": float(np.mean((y_true[t_idx] >= lo_m)
                                                   & (y_true[t_idx] <= hi_m))),
                "q": float(qk),
            }

    q_test = np.array([q_by_group[int(k)] for k in g_test])
    lo = mu_test - q_test * sd_test
    hi = mu_test + q_test * sd_test
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    width = float(np.mean(hi - lo) / (np.ptp(y_true) + 1e-9))
    return cov, width, per_group, q_global


def main():
    subsets = sys.argv[1:] or list(CONFIGS.keys())
    out = {}
    for ds in subsets:
        cfg = CONFIGS[ds]
        rows = []
        for seed in SEEDS:
            try:
                # physdec_mc now returns aligned condition-group labels (8-tuple)
                # DS02: continuous conditions -> window-level grouping
                mu_v, sd_v, y_v, mu_t, sd_t, y_t, g_v, g_t = physdec_mc(
                    ds, seed, cfg["window"], cfg["n_clusters"], cfg["norm_mode"],
                    window_level=(ds == "DS02"))
            except FileNotFoundError:
                print(f"[{ds}] missing checkpoint seed {seed}, skip")
                continue
            cov_g, width_g, per_g, q_glob = group_conformal_widths(
                mu_v, sd_v, y_v, g_v, mu_t, sd_t, y_t, g_t)
            # global (marginal) for comparison
            from conformal_eval import conformal_widths
            cov_m, width_m, _ = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
            rows.append({"seed": seed, "group_cov": cov_g, "group_width": width_g,
                         "marginal_cov": cov_m, "marginal_width": width_m,
                         "per_group": per_g, "q_global": q_glob})

        n = len(rows)
        out[ds] = {
            "n_seeds": n,
            "group_conditional": {
                "coverage_mean": round(float(np.mean([r["group_cov"] for r in rows])), 3),
                "coverage_std": round(float(np.std([r["group_cov"] for r in rows], ddof=1)), 3),
                "width_mean": round(float(np.mean([r["group_width"] for r in rows])), 4),
                "width_std": round(float(np.std([r["group_width"] for r in rows], ddof=1)), 4),
            },
            "marginal": {
                "coverage_mean": round(float(np.mean([r["marginal_cov"] for r in rows])), 3),
                "coverage_std": round(float(np.std([r["marginal_cov"] for r in rows], ddof=1)), 3),
                "width_mean": round(float(np.mean([r["marginal_width"] for r in rows])), 4),
                "width_std": round(float(np.std([r["marginal_width"] for r in rows], ddof=1)), 4),
            },
            "per_group_avg": {},
        }
        # aggregate per-group coverage across seeds (group vs marginal q)
        gmap, gmmap = {}, {}
        for r in rows:
            for k, v in r["per_group"].items():
                gmap.setdefault(k, []).append(v["coverage"])
                gmmap.setdefault(k, []).append(v["coverage_marginal"])
        for k in gmap:
            out[ds]["per_group_avg"][str(k)] = {
                "coverage_mean": round(float(np.mean(gmap[k])), 3),
                "coverage_std": round(float(np.std(gmap[k], ddof=1)), 3),
                "coverage_marginal_mean": round(float(np.mean(gmmap[k])), 3),
            }
        p, m = out[ds]["group_conditional"], out[ds]["marginal"]
        print(f"[{ds}] group-cond 90%: cov {p['coverage_mean']}±{p['coverage_std']} "
              f"w {p['width_mean']} | marginal: cov {m['coverage_mean']}±{m['coverage_std']} "
              f"w {m['width_mean']}")
        if out[ds]["per_group_avg"]:
            print(f"       per-group cov: "
                  + ", ".join(f"g{k}={v['coverage_mean']}"
                              for k, v in out[ds]["per_group_avg"].items()))

    out_path = RESULTS_DIR / "physdec" / "conformal_group_stats.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
