"""Split-conformal uncertainty comparison.

Both models receive the SAME conformal treatment: studentized residuals from
the validation engines define a conformity quantile, which is then applied to
the test predictions. Because split conformal carries a finite-sample marginal
coverage guarantee at any alpha, the informative comparison metric is interval
EFFICIENCY (mean width relative to the range), where the sharper model wins.

The protocol also removes a calibration artefact. The two pipelines archive
their spread under different conventions -- the CDN-RUL archives store a
k-scaled standard deviation, the baseline archives the raw standard deviation
plus k. Under conformal prediction both conventions become irrelevant, so the
comparison reflects the models rather than their storage formats.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR
from train_physdec import load_data, cond_of, DEVICE
from physdec_model import PhysDecRUL

SEEDS = [42, 2024, 7, 123, 99]
ALPHA = 0.1  # 90% conformal coverage


def physdec_mc(subset: str, seed: int, window: int, n_clusters: int,
               norm_mode: str, mc_samples: int = 50, window_level: bool = False,
               return_units: bool = False):
    """Load a trained PhysDec checkpoint and produce MC mean/std for val & test."""
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import KMeans as SKMeans

    data = load_data(subset, window=window, seed=seed)
    n_sensors, n_conds = data["n_sensors"], data["n_conds"]
    cond_tr = data["X_train"][..., :n_conds].reshape(-1, n_conds)

    cluster_centers = cluster_stats = gmm_params = None
    label_fit = None
    if n_clusters > 0 and n_conds > 0:
        cond_f = data["X_train"][..., :n_conds].reshape(-1, n_conds)
        sens_f = data["X_train"][..., n_conds:].reshape(-1, n_sensors)
        if norm_mode == "soft":
            gm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                                 random_state=seed, max_iter=200).fit(cond_f)
            cluster_centers = torch.tensor(gm.means_, dtype=torch.float32)
            gmm_params = {"means": torch.tensor(gm.means_, dtype=torch.float32),
                          "covs": torch.tensor(gm.covariances_, dtype=torch.float32),
                          "logpi": torch.tensor(np.log(gm.weights_), dtype=torch.float32)}
            labels = gm.predict(cond_f)
            label_fit = gm
        else:
            km = SKMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(cond_f)
            cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            labels = km.predict(cond_f)
            label_fit = km
        means, stds = [], []
        for k in range(n_clusters):
            m = labels == k
            means.append(sens_f[m].mean(0) if m.sum() >= 2 else np.zeros(n_sensors, np.float32))
            stds.append(sens_f[m].std(0) if m.sum() >= 2 else np.ones(n_sensors, np.float32))
        cluster_stats = (torch.tensor(np.stack(means), dtype=torch.float32),
                         torch.tensor(np.stack(stds), dtype=torch.float32))

    model = PhysDecRUL(n_sensors=n_sensors, n_conditions=n_conds,
                       window=window, cluster_centers=cluster_centers,
                       cluster_stats=cluster_stats, norm_mode=norm_mode,
                       gmm_params=gmm_params).to(DEVICE)
    ckpt_path = MODELS_DIR / "physdec" / f"physdec_{subset}_seed{seed}.pt"
    if not ckpt_path.exists():
        # fallback: legacy multiseed runs saved as physdec_{subset}_{tag}.pt with tag=seed{seed}
        ckpt_path = MODELS_DIR / "physdec" / f"physdec_{subset}_seed{seed}.pt"
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(ckpt["state"])
    model.to(DEVICE)

    def mc(X):
        means, stds = [], []
        model.train()
        with torch.no_grad():
            for i in range(0, len(X), 1024):
                cond, sens = cond_of(X[i:i + 1024].to(DEVICE), n_conds)
                m, s = model.mc_predict(sens, cond, n_samples=mc_samples)
                means.append(m.cpu().numpy()); stds.append(s.cpu().numpy())
        return np.concatenate(means), np.concatenate(stds)

    X_va = torch.tensor(data["X_val"])
    X_te = torch.tensor(data["X_test"])
    mu_v, sd_v = mc(X_va)
    mu_t, sd_t = mc(X_te)

    # ---- engine-level condition-group labels (dominant condition over the
    # engine's FULL trajectory, not just the last window) ----
    g_v, g_t = _engine_condition_labels(data, label_fit, n_conds, n_clusters,
                                        window_level=window_level)
    if return_units:
        # u_test lets callers attribute each test window to its engine, which is
        # required for cluster-robust coverage intervals (windows within one
        # engine are highly dependent: stride-1 windows overlap by window-1 steps).
        return (mu_v, sd_v, data["y_val"], mu_t, sd_t, data["y_test"], g_v, g_t,
                np.asarray(data["u_test"], dtype=np.float64))
    return (mu_v, sd_v, data["y_val"], mu_t, sd_t, data["y_test"], g_v, g_t)


def _engine_condition_labels(data, label_fit, n_conds, n_clusters,
                             window_level: bool = False):
    """Return WINDOW-level condition-group labels for val and test.

    Each window's label = dominant condition (mode) of its engine's trajectory
    windows, so len(g_v) == len(X_val) and len(g_t) == len(X_test).

    window_level=True (N-CMAPSS DS02, continuous conditions): group each
    window by its OWN condition cluster — engine-mode would collapse all
    windows of a unit into one label and destroy the condition grouping."""
    u_val = data["u_val"]
    u_test = data["u_test"]
    if label_fit is None and n_clusters <= 0:
        # no model clustering: fit fresh GMM over SCALED conditions for grouping
        from sklearn.mixture import GaussianMixture as _GM
        from sklearn.preprocessing import StandardScaler as _SS
        cond_tr = data["X_train"][..., :n_conds].reshape(-1, n_conds)
        _sc = _SS().fit(cond_tr)
        # subsample for speed (DS02 has ~22M window rows)
        rng = np.random.default_rng(42)
        pick = rng.choice(len(cond_tr), min(500_000, len(cond_tr)), replace=False)
        gm = _GM(n_components=6, covariance_type="full", random_state=42,
                 max_iter=100, reg_covar=1e-3).fit(_sc.transform(cond_tr[pick]))
        cond_v = _sc.transform(data["X_val"][..., :n_conds].mean(axis=1))
        cond_t = _sc.transform(data["X_test"][..., :n_conds].mean(axis=1))
        lab_v = gm.predict(cond_v)
        lab_t = gm.predict(cond_t)
        if window_level:
            return lab_v.astype(int), lab_t.astype(int)
        return _mode_by_engine(lab_v, u_val, lab_t, u_test)
    if label_fit is None:
        return np.zeros(len(u_val), dtype=int), np.zeros(len(u_test), dtype=int)

    cond_v = data["X_val"][..., :n_conds].mean(axis=1)
    cond_t = data["X_test"][..., :n_conds].mean(axis=1)
    lab_v = label_fit.predict(cond_v)
    lab_t = label_fit.predict(cond_t)
    if window_level:
        return lab_v.astype(int), lab_t.astype(int)
    return _mode_by_engine(lab_v, u_val, lab_t, u_test)


def _mode_by_engine(lab_v, u_val, lab_t, u_test):
    """Per-engine dominant label (mode over that engine's windows)."""
    def mode_map(ids, labs):
        out = {}
        for u, l in zip(ids, labs):
            out.setdefault(u, []).append(l)
        return {u: int(np.bincount(ls).argmax()) for u, ls in out.items()}

    map_v = mode_map(u_val, lab_v)
    map_t = mode_map(u_test, lab_t)
    return (np.array([map_v[u] for u in u_val], dtype=int),
            np.array([map_t[u] for u in u_test], dtype=int))


def _gmm_labels(cond, gmm_params):
    """Assign GMM hard labels to condition vectors (full posterior incl. prior)."""
    import torch as _t
    c = _t.tensor(cond)
    means = _t.tensor(gmm_params["means"])
    covs = _t.tensor(gmm_params["covs"])
    logpi = _t.tensor(gmm_params["logpi"])
    K = means.shape[0]
    logp = _t.empty((len(c), K))
    for k in range(K):
        d = c - means[k]
        prec = _t.linalg.inv(covs[k])
        maha = (d @ prec * d).sum(dim=1)
        logdet = _t.logdet(covs[k])
        logp[:, k] = logpi[k] - 0.5 * (maha + logdet)
    return logp.argmax(dim=1).numpy()


def lstmcond_mc(subset: str, window: int = 30, mc_samples: int = 50):
    """Load the LSTM-Cond baseline checkpoint and produce MC mean/std."""
    from train_lstm_cond import LSTMRULCond, mc_dropout_predict
    from conditions import prepare_cond

    # read n_clusters the checkpoint was actually trained with (FD001/FD003=1)
    meta = json.loads((RESULTS_DIR / f"metrics_lstm_cond_{subset}.json")
                      .read_text(encoding="utf-8"))
    n_clusters = meta.get("n_clusters", 6)

    X_train, y_train, u, X_test, y_test, eids, enc = prepare_cond(
        subset, n_clusters=n_clusters, window=window)
    n_feat = X_train.shape[2]
    model = LSTMRULCond(n_features=n_feat).to(DEVICE)
    state = torch.load(str(MODELS_DIR / f"lstm_cond_{subset}.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    X_te = torch.tensor(X_test).to(DEVICE)
    mu_t, sd_t, _ = mc_dropout_predict(model, X_te, n_samples=mc_samples, device=DEVICE)
    # validation set: re-split engines to match training split (seed 42)
    rng = np.random.default_rng(42)
    pool = np.unique(u)
    val_engines = set(rng.choice(pool, size=int(0.2 * len(pool)), replace=False))
    vm = np.isin(u, list(val_engines))
    X_va = torch.tensor(X_train[vm]).to(DEVICE)
    mu_v, sd_v, _ = mc_dropout_predict(model, X_va, n_samples=mc_samples, device=DEVICE)
    return (mu_v, sd_v, y_train[vm], mu_t, sd_t, y_test)


def conformal_widths(mu_cal, sd_cal, y_cal, mu_test, sd_test, y_true, alpha=ALPHA):
    """Studentized split conformal: interval = mu ± q·sd, q from |resid|/sd quantile."""
    resid = np.abs(y_cal - mu_cal) / np.maximum(sd_cal, 1e-6)
    n = len(resid)
    q = np.quantile(resid, np.ceil((n + 1) * (1 - alpha)) / n)
    lo = mu_test - q * sd_test
    hi = mu_test + q * sd_test
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    width = float(np.mean(hi - lo) / (np.ptp(y_true) + 1e-9))
    return cov, width, q


CONFIGS = {
    "FD001": dict(n_clusters=1, norm_mode="soft", window=30),
    "FD002": dict(n_clusters=6, norm_mode="soft", window=30),
    "FD003": dict(n_clusters=1, norm_mode="soft", window=30),
    "FD004": dict(n_clusters=6, norm_mode="soft", window=30),
    "DS02": dict(n_clusters=0, norm_mode="hard", window=50),
}


def main():
    subsets = sys.argv[1:] or list(CONFIGS.keys())
    out = {}
    for ds in subsets:
        cfg = CONFIGS[ds]
        # PhysDec: aggregate over seeds
        phys_covs, phys_widths = [], []
        for seed in SEEDS:
            try:
                mu_v, sd_v, y_v, mu_t, sd_t, y_t, _, _ = physdec_mc(
                    ds, seed, cfg["window"], cfg["n_clusters"], cfg["norm_mode"])
            except FileNotFoundError:
                print(f"[{ds}] missing checkpoint seed {seed}, skip")
                continue
            c, w, q = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
            phys_covs.append(c); phys_widths.append(w)
        # baseline (LSTM-Cond, seed 42) — C-MAPSS only
        base = None
        if ds.startswith("FD"):
            try:
                mu_v, sd_v, y_v, mu_t, sd_t, y_t = lstmcond_mc(ds, window=cfg["window"])
                base = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
            except FileNotFoundError as e:
                print(f"[{ds}] baseline missing: {e}")
        out[ds] = {
            "physdec": {"n_seeds": len(phys_covs),
                        "coverage_mean": round(float(np.mean(phys_covs)), 3),
                        "coverage_std": round(float(np.std(phys_covs, ddof=1)), 3),
                        "width_mean": round(float(np.mean(phys_widths)), 4),
                        "width_std": round(float(np.std(phys_widths, ddof=1)), 4)},
            "baseline_lstmcond": None if base is None else
            {"coverage": round(base[0], 3), "width": round(base[1], 4)},
        }
        p = out[ds]["physdec"]; b = out[ds]["baseline_lstmcond"]
        print(f"[{ds}] PhysDec conformal 90%: cov {p['coverage_mean']}±{p['coverage_std']} "
              f"| width {p['width_mean']}±{p['width_std']}")
        if b:
            print(f"     LSTM-Cond conformal 90%: cov {b['coverage']} | width {b['width']}")

    out_path = RESULTS_DIR / "physdec" / "conformal_stats.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
