"""Group-conditional conformal comparison: LSTM (no condition norm) vs PhysDec-RUL.

Purpose (paper A1):
1. Demonstrate group-conditional conformal — per-group quantiles q_k give
   condition-aware coverage guarantees (vs a single marginal q).
2. Show the honest finding: because CDFE already condition-normalizes the
   degradation stream, PhysDec-RUL residuals are near-homogeneous across
   condition groups, so group-conditional and marginal coverage nearly
   coincide. For a model WITHOUT condition normalization (plain LSTM), the
   groups are more heterogeneous and group-conditional matters more.

This contrast is the empirical justification of both the protocol AND the
condition-decoupling mechanism.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR
from train_physdec import DEVICE
from conformal_eval import conformal_widths
from conformal_group import group_conformal_widths


def lstm_mc_ds02(seed: int, window: int = 50, mc_samples: int = 30):
    """MC-Dropout predictions for the DS02 plain-LSTM baseline (val = 20% units)."""
    from train_ncmapss import LSTMRUL
    from ncmapss_data import prepare_ncmapss
    X_train, y_train, units, X_test, y_test, eids = prepare_ncmapss(window=window)

    # same engine-level 80/20 split as train_ncmapss.train_lstm (seed)
    rng = np.random.default_rng(seed)
    pool = np.unique(units)
    val_units = set(rng.choice(pool, size=max(1, int(0.2 * len(pool))), replace=False))
    vm = np.isin(units, list(val_units))
    X_va, y_va = X_train[vm], y_train[vm]

    model = LSTMRUL(n_features=X_test.shape[2]).to(DEVICE)
    state = torch.load(str(MODELS_DIR / f"lstm_DS02_seed{seed}.pt"),
                       map_location="cpu")
    model.load_state_dict(state)

    def mc(X):
        model.train()
        means, stds = [], []
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                xb = torch.tensor(X[i:i + 1024]).to(DEVICE)
                sm = []
                for _ in range(mc_samples):
                    sm.append(model(xb).cpu().numpy())
                sm = np.stack(sm)
                means.append(sm.mean(0)); stds.append(sm.std(0))
        return np.concatenate(means).ravel(), np.concatenate(stds).ravel()

    mu_v, sd_v = mc(X_va)
    mu_t, sd_t = mc(X_test)
    return mu_v, sd_v, y_va, mu_t, sd_t, y_test


def gmm_labels_ds02(seed: int, window: int = 50, val_mask=None):
    """Window-level GMM condition labels for DS02 val & test (aligned with split)."""
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from ncmapss_data import prepare_ncmapss
    X_train, y_train, units, X_test, y_test, eids = prepare_ncmapss(window=window)
    if val_mask is None:
        rng = np.random.default_rng(seed)
        pool = np.unique(units)
        val_units = set(rng.choice(pool, size=max(1, int(0.2 * len(pool))), replace=False))
        val_mask = np.isin(units, list(val_units))
    cond_tr = X_train[..., :4].reshape(-1, 4)
    sc = StandardScaler().fit(cond_tr)
    gm = GaussianMixture(n_components=6, covariance_type="full", random_state=42,
                         max_iter=100, reg_covar=1e-3).fit(sc.transform(cond_tr))
    g_v = gm.predict(sc.transform(X_train[val_mask][..., :4].mean(axis=1)))
    g_t = gm.predict(sc.transform(X_test[..., :4].mean(axis=1)))
    return g_v.astype(int), g_t.astype(int), y_train[val_mask], y_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 7, 123, 99])
    ap.add_argument("--subsample", type=int, default=0,
                    help="random subsample of test windows for speed (0=all)")
    args = ap.parse_args()

    print("=== DS02 group-conditional conformal: LSTM vs PhysDec-RUL ===")
    out = {"mc_samples": args.mc_samples}

    # PhysDec-RUL: reuse already-computed results from conformal_group.py
    try:
        phys_stats = json.loads((RESULTS_DIR / "physdec" /
                                 "conformal_group_stats.json")
                                .read_text(encoding="utf-8"))["DS02"]
        out["physdec"] = [{
            "seed": -1, "group_cov": phys_stats["group_conditional"]["coverage_mean"],
            "marginal_cov": phys_stats["marginal"]["coverage_mean"],
            "group_width": phys_stats["group_conditional"]["width_mean"],
            "marginal_width": phys_stats["marginal"]["width_mean"],
            "per_group": {k: {"cov": v["coverage_mean"],
                              "marg": v.get("coverage_marginal_mean", v["coverage_mean"]),
                              "n": -1} for k, v in phys_stats["per_group_avg"].items()},
        }]
        print("[physdec ] (reused conformal_group_stats.json)")
    except (FileNotFoundError, KeyError) as e:
        print(f"[physdec ] missing: {e}")

    # LSTM baseline (only this is newly computed)
    for seed in args.seeds:
        try:
            mu_v, sd_v, y_v, mu_t, sd_t, y_t = lstm_mc_ds02(seed, mc_samples=args.mc_samples)
            g_v, g_t, _, _ = gmm_labels_ds02(seed)
        except FileNotFoundError:
            print(f"[lstm    ] missing checkpoint seed {seed}, skip")
            continue
        if args.subsample > 0 and len(y_t) > args.subsample:
            rng = np.random.default_rng(0)
            pick = rng.choice(len(y_t), args.subsample, replace=False)
            mu_t, sd_t, y_t, g_t = mu_t[pick], sd_t[pick], y_t[pick], g_t[pick]
        cov_g, w_g, per_g, q_glob = group_conformal_widths(
            mu_v, sd_v, y_v, g_v, mu_t, sd_t, y_t, g_t)
        cov_m, w_m, _ = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
        out.setdefault("lstm", []).append({
            "seed": seed, "group_cov": cov_g, "marginal_cov": cov_m,
            "group_width": w_g, "marginal_width": w_m,
            "per_group": {k: {"cov": v["coverage"], "marg": v["coverage_marginal"],
                              "n": v["n"]} for k, v in per_g.items()},
        })

    for model in ["physdec", "lstm"]:
        rows = out[model]
        if not rows:
            continue
        gc = np.mean([r["group_cov"] for r in rows])
        mc = np.mean([r["marginal_cov"] for r in rows])
        gw = np.mean([r["group_width"] for r in rows])
        # per-group spread (range of per-group coverage, averaged over seeds)
        spreads = []
        for r in rows:
            covs = [v["cov"] for v in r["per_group"].values()]
            spreads.append(max(covs) - min(covs))
        spread = np.mean(spreads)
        print(f"[{model:8s}] group-cov {gc:.3f} vs marginal {mc:.3f} | "
              f"per-group spread {spread:.3f} | group-width {gw:.4f}")

    out_path = RESULTS_DIR / "physdec" / "conformal_group_compare.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
