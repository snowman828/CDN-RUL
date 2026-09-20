"""PhysDec-RUL trainer: unified C-MAPSS / N-CMAPSS pipeline with ablation support.

Usage:
  python train_physdec.py --subset FD001 --tag full
  python train_physdec.py --subset FD001 --tag nophys  --no-physics
  python train_physdec.py --subset FD001 --tag noorth  --no-orthogonal
  python train_physdec.py --subset DS02 --tag full --epochs 40
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from common import RESULTS_DIR, MODELS_DIR, SEED
from metrics import report_metrics
from physdec_model import PhysDecRUL

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Unified data interface: returns train/valid/test tensors + engine ids
# --------------------------------------------------------------------------
def load_data(subset: str, window: int, seed: int = SEED, val_frac: float = 0.2):
    """Returns dict with keys X_train, y_train, u_train, X_val, y_val, u_val,
    X_test, y_test, u_test, n_sensors, n_conds, cmapss (bool)."""
    if subset.startswith("FD"):
        from data import load_dataset, add_rul_targets, fit_scaler, apply_scaler, make_windows
        from common import SELECTED_SENSORS
        d = load_dataset(subset)
        train = add_rul_targets(d["train"])
        scaler = fit_scaler(train)
        train_s = apply_scaler(train, scaler)
        # feature window = [op1-3] + [14 selected sensors]  (cond cols first)
        FEATS = ["op1", "op2", "op3"] + SELECTED_SENSORS
        X, y, u = make_windows(train_s, sensors=FEATS, window=window)
        # test: last window per engine
        test_s = apply_scaler(d["test"], scaler)
        Xt, et = [], []
        for unit, grp in test_s.groupby("unit"):
            vals = grp[FEATS].values
            tail = vals[-window:]
            if len(tail) < window:
                tail = np.vstack([np.repeat(tail[:1], window - len(tail), axis=0), tail])
            Xt.append(tail); et.append(unit)
        X_test = np.asarray(Xt, dtype=np.float32)
        y_test = d["rul"]["RUL"].values.astype(np.float32)
        u_test = np.asarray(et, dtype=np.float64)
        n_feat, n_conds, n_sensors = len(FEATS), 3, len(SELECTED_SENSORS)
    else:  # N-CMAPSS
        from ncmapss_data import prepare_ncmapss, FEATURE_COLS
        X, y, u, X_test, y_test, u_test = prepare_ncmapss(window=window)
        n_feat, n_conds = len(FEATURE_COLS), 4
        n_sensors = n_feat - n_conds  # sensors = total minus op-conditions
    # engine-level split (no leakage)
    rng = np.random.default_rng(seed)
    pool = np.unique(u)
    val_engines = set(rng.choice(pool, size=max(1, int(val_frac * len(pool))),
                                 replace=False))
    vm = np.isin(u, list(val_engines))
    return {"X_train": X[~vm], "y_train": y[~vm], "u_train": u[~vm],
            "X_val": X[vm], "y_val": y[vm], "u_val": u[vm],
            "X_test": X_test, "y_test": y_test, "u_test": u_test,
            "n_sensors": n_sensors, "n_conds": n_conds, "cmapss": subset.startswith("FD")}


def cond_of(x_sensors: torch.Tensor, n_first: int):
    """Split window into (conditions, sensors): conditions are the first n_first cols."""
    return x_sensors[..., :n_first], x_sensors[..., n_first:]


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(subset: str, tag: str = "full", window: int = 50, epochs: int = 50,
          batch_size: int = 256, lr: float = 1e-3, patience: int = 12,
          use_physics: bool = True, use_orthogonal: bool = True,
          lambda_phys: float = 1.0, lambda_orth: float = 0.5,
          phys_warmup: int = 8, orth_delay: int = 10,
          mc_samples: int = 50, calibrate: bool = True,
          per_step: bool | None = None, n_clusters: int = 0,
          norm_mode: str = "hard", seed: int = SEED):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = load_data(subset, window=window, seed=seed)
    # conditions are the FIRST columns for both datasets (op1-3 / alt,Mach,TRA,T2)
    n_sensors = data["n_sensors"]
    n_conds = data["n_conds"]
    # per-step time-varying targets only valid when cycle decrements 1 per window
    # step (C-MAPSS); N-CMAPSS at 0.1Hz breaks that → last-step supervision only.
    # Default: C-MAPSS True / N-CMAPSS False; CLI can force for ablations.
    if per_step is None:
        per_step = data["cmapss"]

    X_tr = torch.tensor(data["X_train"]).to(DEVICE)
    y_tr = torch.tensor(data["y_train"]).to(DEVICE)
    X_va = torch.tensor(data["X_val"]).to(DEVICE)
    y_va = torch.tensor(data["y_val"]).to(DEVICE)
    X_te = torch.tensor(data["X_test"]).to(DEVICE)
    y_te = data["y_test"]

    # ---- per-condition normalization prior (KMeans hard | GMM soft) ----
    # (fitted ONLY on train windows; leakage-free. Sensors are z-scored per
    # operating-regime cluster before the degradation stream.)
    cluster_centers = None
    cluster_stats = None
    gmm_params = None
    if n_clusters > 0 and n_conds > 0:
        cond_flat = data["X_train"][..., :n_conds].reshape(-1, n_conds)
        sens_flat = data["X_train"][..., n_conds:].reshape(-1, n_sensors)
        # cap sample count for fitting on huge sets (DS02: 22M rows)
        rng_s = np.random.default_rng(SEED)
        if len(cond_flat) > 1_000_000:
            pick = rng_s.choice(len(cond_flat), 1_000_000, replace=False)
            cond_fit, sens_fit = cond_flat[pick], sens_flat[pick]
        else:
            cond_fit, sens_fit = cond_flat, sens_flat

        if norm_mode == "soft":
            from sklearn.mixture import GaussianMixture
            gm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                                 random_state=seed, max_iter=200).fit(cond_fit)
            cluster_centers = torch.tensor(gm.means_, dtype=torch.float32)
            gmm_params = {
                "means": torch.tensor(gm.means_, dtype=torch.float32),
                "covs": torch.tensor(gm.covariances_, dtype=torch.float32),
                "logpi": torch.tensor(np.log(gm.weights_), dtype=torch.float32),
            }
            labels = gm.predict(cond_fit)
            kind = "GMM soft"
        else:
            from sklearn.cluster import KMeans as SKMeans
            km = SKMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(cond_fit)
            cluster_centers = torch.tensor(km.cluster_centers_, dtype=torch.float32)
            labels = km.predict(cond_fit)
            kind = "KMeans hard"

        means, stds = [], []
        for k in range(n_clusters):
            m = labels == k
            if m.sum() < 2:
                means.append(np.zeros(n_sensors, dtype=np.float32))
                stds.append(np.ones(n_sensors, dtype=np.float32))
                continue
            means.append(sens_fit[m].mean(0))
            stds.append(sens_fit[m].std(0))
        cluster_stats = (torch.tensor(np.stack(means), dtype=torch.float32),
                         torch.tensor(np.stack(stds), dtype=torch.float32))
        print(f"  {kind} prior: {n_clusters} clusters on {n_conds} op-settings "
              f"(sizes {np.bincount(labels).tolist()}) — per-condition sensor norm")

    print(f"[{subset}] tag={tag} | phys={use_physics} orth={use_orthogonal} | "
          f"device={DEVICE} | train {X_tr.shape} val {X_va.shape} test {X_te.shape}")

    model = PhysDecRUL(n_sensors=n_sensors, n_conditions=n_conds,
                       use_physics=use_physics, use_orthogonal=use_orthogonal,
                       window=window, cluster_centers=cluster_centers,
                       cluster_stats=cluster_stats, norm_mode=norm_mode,
                       gmm_params=gmm_params).to(DEVICE)
    if model.receptive_field is not None:
        print(f"  TCN receptive field = {model.receptive_field} (window {window})")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)

    n = len(X_tr)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        eloss = {"total": 0.0, "mse": 0.0, "phys": 0.0, "orth": 0.0}
        nb = 0
        lam_phys = lambda_phys * min(1.0, epoch / phys_warmup) if use_physics else 0.0
        lam_orth = lambda_orth if (use_orthogonal and epoch > orth_delay) else 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx]
            cond, sens = cond_of(xb, n_conds)
            opt.zero_grad()
            step_mu, mu, aux = model(sens, cond)
            losses = model.total_loss(y_tr[idx], step_mu, mu, aux["z_d"], aux["z_c"],
                                      lambda_phys=lam_phys, lambda_orth=lam_orth)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in eloss:
                eloss[k] += losses[k].item()
            nb += 1
        # validation (batched — DS02 val = 85k windows would OOM in one pass)
        model.eval()
        with torch.no_grad():
            val_preds = []
            for j in range(0, len(X_va), 2048):
                cond_v, sens_v = cond_of(X_va[j:j + 2048], n_conds)
                _, mu_v, _ = model(sens_v, cond_v)
                val_preds.append(mu_v)
            v_nll = F.mse_loss(torch.cat(val_preds), y_va).item()
        sched.step(v_nll)
        if v_nll < best_val:
            best_val, best_state = v_nll, {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ epoch {epoch}")
                break
        if epoch % 5 == 0 or epoch == 1:
            print(f"  ep {epoch:3d} | total {eloss['total']/nb:.2f} | "
                  f"mse {eloss['mse']/nb:.2f} | phys {eloss['phys']/nb:.3f} | "
                  f"orth {eloss['orth']/nb:.3f} | valMSE {v_nll:.1f}")

    model.load_state_dict(best_state)
    model_dir = MODELS_DIR / "physdec"
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt = model_dir / f"physdec_{subset}_{tag}.pt"
    torch.save({"state": best_state,
                "config": {"n_sensors": n_sensors, "n_conds": n_conds,
                           "use_physics": use_physics,
                           "use_orthogonal": use_orthogonal}},
               ckpt)

    # ---- test point prediction ----
    model.eval()
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X_te), 2048):
            xb = X_te[i:i + 2048]
            cond, sens = cond_of(xb, n_conds)
            y_pred.append(model.predict(sens, cond).cpu().numpy())
    y_pred = np.concatenate(y_pred).clip(min=0)
    metrics = report_metrics(y_te, y_pred, tag=f"{subset} PhysDec-{tag}")

    out = {"model": "physdec", "subset": subset, "tag": tag, "seed": seed,
           "use_physics": use_physics, "use_orthogonal": use_orthogonal,
           **metrics,
           "device": DEVICE, "epochs_used": epoch,
           "train_time_s": round(time.time() - t0, 1)}
    out_dir = RESULTS_DIR / "physdec"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Filename convention: metrics_{subset}_{tag}.json. The tag already carries
    # the seed for seed runs (run_multiseed passes tag=f"seed{seed}"), so this
    # reproduces the committed names exactly -- metrics_FD001_seed42.json,
    # metrics_FD001_full.json, metrics_FD002_nophys_v16_ab_103.json. The old
    # form appended _seed{seed} as well, so a re-run produced
    # metrics_FD001_seed42_seed42.json and never overwrote the committed file,
    # which made a reader's reproduction impossible to compare against the
    # published results.
    p = out_dir / f"metrics_{subset}_{tag}.json"
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8")).get("seed")
        except Exception:
            prev = None
        if prev is not None and prev != seed:
            raise SystemExit(
                f"refusing to overwrite {p.name}: it holds seed {prev}, this run "
                f"is seed {seed}. Use a tag that includes the seed so runs do not "
                f"collide (run_multiseed passes tag=seed<seed>).")
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {p}")

    # ---- MC-Dropout uncertainty ----
    if mc_samples:
        mean, std = mc_eval(model, X_te, n_conds, mc_samples)
        if calibrate:
            k = calibrate_std(model, data, n_conds, window, mc_samples)
            std = std * k
        z = 1.645
        lo, hi = mean - z * std, mean + z * std
        cov = float(np.mean((y_te >= lo) & (y_te <= hi)))
        out["mc90_coverage"] = round(cov, 3)
        out["mc_mean_ci_width"] = round(float(np.mean(2 * z * std)), 2)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        np.savez(out_dir / f"mc_{subset}_{tag}.npz",
                 y_test=y_te, mean=mean, std=std)
        print(f"[{subset}] MC 90% CI coverage={cov:.3f} "
              f"mean width={out['mc_mean_ci_width']} cycles")
    return out


def _beta_of(model):
    return None


@torch.no_grad()
def mc_eval(model, X_te, n_conds, n_samples: int = 50) -> tuple:
    model.train()
    means, stds = [], []
    for i in range(0, len(X_te), 1024):
        xb = X_te[i:i + 1024]
        cond, sens = cond_of(xb, n_conds)
        m, s = model.mc_predict(sens, cond, n_samples=n_samples)
        means.append(m.cpu().numpy())
        stds.append(s.cpu().numpy())
    return np.concatenate(means).clip(min=0), np.concatenate(stds)


def calibrate_std(model, data, n_conds, window, mc_samples: int, target: float = 0.9,
                  z: float = 1.645):
    """Bisection search for std multiplier k on the validation engines."""
    X_va = torch.tensor(data["X_val"]).to(DEVICE)
    y_va = data["y_val"]
    mean, std = mc_eval(model, X_va, n_conds, mc_samples)
    mean, std = mean.clip(min=0), np.maximum(std, 1e-6)

    def cov_for(k):
        lo = mean - z * k * std
        hi = mean + z * k * std
        return float(np.mean((y_va >= lo) & (y_va <= hi)))

    lo_k, hi_k = 0.1, 10.0
    for _ in range(60):
        mid = 0.5 * (lo_k + hi_k)
        if cov_for(mid) < target:
            lo_k = mid
        else:
            hi_k = mid
    k = 0.5 * (lo_k + hi_k)
    print(f"  calibration k={k:.3f} (val coverage {cov_for(k):.3f}, target {target})")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--no-physics", action="store_true")
    ap.add_argument("--no-orthogonal", action="store_true")
    ap.add_argument("--mc-samples", type=int, default=50)
    ap.add_argument("--no-per-step", action="store_true",
                    help="force last-step supervision (isolation ablation)")
    ap.add_argument("--n-clusters", type=int, default=0,
                    help="KMeans operating-regime prior clusters (0=disabled, "
                         "per-condition norm)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--norm-mode", choices=["hard", "soft"], default="hard",
                    help="per-condition normalization: KMeans hard ("
                         "discrete regimes) or GMM soft (continuous)")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="random seed for engine split / GMM / training")
    args = ap.parse_args()
    train(args.subset, tag=args.tag, window=args.window, epochs=args.epochs,
          batch_size=args.batch_size,
          use_physics=not args.no_physics, use_orthogonal=not args.no_orthogonal,
          mc_samples=args.mc_samples,
          per_step=False if args.no_per_step else None,
          n_clusters=args.n_clusters, norm_mode=args.norm_mode,
          seed=args.seed)


if __name__ == "__main__":
    main()
