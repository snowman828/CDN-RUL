"""Empirical decoupling validation — does CDFE actually separate
degradation (z_d) from operating-condition (z_c) information?

Correlation check on the FD002 validation engines: for each representation
dimension, compute the max |Pearson r| against any op-setting. If CDFE
decouples properly, z_c should correlate strongly with conditions while
z_d should be (near-)orthogonal to them.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from common import MODELS_DIR
from train_physdec import load_data, cond_of
from physdec_model import PhysDecRUL


def build_model(subset: str, seed: int, n_clusters: int, norm_mode: str, window: int):
    from sklearn.cluster import KMeans as SKMeans
    from sklearn.mixture import GaussianMixture
    data = load_data(subset, window=window, seed=seed)
    n_sensors, n_conds = data["n_sensors"], data["n_conds"]
    cond_f = data["X_train"][..., :n_conds].reshape(-1, n_conds)
    sens_f = data["X_train"][..., n_conds:].reshape(-1, n_sensors)

    cluster_centers = cluster_stats = gmm_params = None
    if n_clusters > 0 and n_conds > 0:
        if norm_mode == "soft":
            gm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                                 random_state=seed, max_iter=200).fit(cond_f)
            cluster_centers = torch.tensor(gm.means_, dtype=torch.float32)
            gmm_params = {"means": torch.tensor(gm.means_, dtype=torch.float32),
                          "covs": torch.tensor(gm.covariances_, dtype=torch.float32),
                          "logpi": torch.tensor(np.log(gm.weights_), dtype=torch.float32)}
            labels = gm.predict(cond_f)
        else:
            km = SKMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(cond_f)
            cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            labels = km.predict(cond_f)
        means, stds = [], []
        for k in range(n_clusters):
            m = labels == k
            means.append(sens_f[m].mean(0) if m.sum() >= 2 else np.zeros(n_sensors, np.float32))
            stds.append(sens_f[m].std(0) if m.sum() >= 2 else np.ones(n_sensors, np.float32))
        cluster_stats = (torch.tensor(np.stack(means), dtype=torch.float32),
                         torch.tensor(np.stack(stds), dtype=torch.float32))
    model = PhysDecRUL(n_sensors=n_sensors, n_conditions=n_conds, window=window,
                       cluster_centers=cluster_centers, cluster_stats=cluster_stats,
                       norm_mode=norm_mode, gmm_params=gmm_params)
    return model, data


def cond_corr(Z, C, label):
    rs = np.array([max(abs(pearsonr(Z[:, j], C[:, k])[0]) for k in range(C.shape[1]))
                   for j in range(Z.shape[1])])
    print(f"{label}: |r| mean={rs.mean():.3f} max={rs.max():.3f} "
          f"dims>0.5={np.sum(rs > 0.5)}/{len(rs)}")
    return rs


def main():
    subset = "FD002"
    model, data = build_model(subset, seed=42, n_clusters=6, norm_mode="soft", window=30)
    ckpt = torch.load(str(MODELS_DIR / "physdec" / "physdec_FD002_full_v16.pt"),
                      map_location="cpu")
    model.load_state_dict(ckpt["state"])
    model.to("cuda")
    model.eval()

    X_va = torch.tensor(data["X_val"])
    cond_va = data["X_val"][..., :3].mean(axis=1)  # (N, 3) mean over window
    zds, zcs = [], []
    with torch.no_grad():
        for i in range(0, len(X_va), 512):
            cond, sens = cond_of(X_va[i:i + 512].to("cuda"), data["n_conds"])
            zd, zc, g = model.cdfe(sens, cond)
            zds.append(zd.mean(1).cpu().numpy())
            zcs.append(zc.mean(1).cpu().numpy())
    zd = np.concatenate(zds)
    zc = np.concatenate(zcs)
    c = cond_va

    print(f"[{subset}] validation engines {len(zd)} | z_d {zd.shape}, z_c {zc.shape}")
    r_zd = cond_corr(zd, c, "z_d (degradation repr.)")
    r_zc = cond_corr(zc, c, "z_c (condition repr.)")
    ratio = r_zc.mean() / max(r_zd.mean(), 1e-6)
    print(f"\nDecoupling ratio (z_c/z_d condition correlation) = {ratio:.1f}x")
    decoupled = r_zd.mean() < 0.3 and r_zc.mean() > 0.3
    print(f"z_d weakly correlated with conditions (<0.3): {r_zd.mean() < 0.3}")
    print(f"z_c strongly correlated with conditions (>0.3): {r_zc.mean() > 0.3}")
    print(f"VERDICT: empirical decoupling {'SUPPORTED ✅' if decoupled else 'NOT SUPPORTED ❌'}")


if __name__ == "__main__":
    main()
