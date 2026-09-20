"""Condition-aware LSTM for multi-condition subsets + MC-Dropout uncertainty.

Features per time step: cluster-scaled sensors (14) + standardized op-settings (3)
+ condition one-hot (k). MC-Dropout at inference gives prediction mean + CI.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import RESULTS_DIR, MODELS_DIR, SEED, WINDOW_SIZE
from conditions import prepare_cond
from metrics import report_metrics


class LSTMRULCond(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),   # enables MC-Dropout at inference
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train(subset: str = "FD002", n_clusters: int = 6, epochs: int = 80,
          batch_size: int = 256, lr: float = 1e-3, patience: int = 10,
          device: str = "auto", seed: int = SEED):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, units, X_test, y_test, engine_ids, enc = prepare_cond(
        subset, n_clusters=n_clusters)
    n_feat = X_train.shape[2]

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tr = load_train_only(subset)
    print(f"[{subset}] device={device} | n_clusters={enc.n_clusters} "
          f"train-cluster sizes={enc.cluster_sizes(tr)}")
    print(f"  windows train {X_train.shape}, test {X_test.shape}, n_feat={n_feat}")

    # Leakage-free engine-level split
    engine_pool = np.unique(units)
    rng = np.random.default_rng(seed)
    val_engines = set(rng.choice(engine_pool, size=int(0.2 * len(engine_pool)), replace=False))
    val_mask = np.isin(units, list(val_engines))
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
    X_va, y_va = X_train[val_mask], y_train[val_mask]
    print(f"  engines {len(engine_pool)} -> train {X_tr.shape[0]} / valid {X_va.shape[0]} windows")

    X_tr = torch.tensor(X_tr).to(device)
    y_tr = torch.tensor(y_tr).to(device)
    X_va = torch.tensor(X_va).to(device)
    y_va = torch.tensor(y_va).to(device)

    model = LSTMRULCond(n_features=n_feat).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    n = len(X_tr)
    best_val, best_state, bad_epochs = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va), y_va).item()
        sched.step(val_loss)
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.detach().cpu().clone()
                                              for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  early stop @ epoch {epoch}")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | train {epoch_loss/n_batches:.4f} | val {val_loss:.4f}")

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(MODELS_DIR / f"lstm_cond_{subset}_seed{seed}.pt"))
    if seed == SEED:  # backward-compat name for legacy evaluators
        torch.save(best_state, str(MODELS_DIR / f"lstm_cond_{subset}.pt"))

    # Point prediction
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test).to(device)).cpu().numpy()
    y_pred = y_pred.clip(min=0)
    metrics = report_metrics(y_test, y_pred, tag=f"{subset} LSTM-Cond")

    out = {"model": "lstm_cond", "subset": subset, "n_clusters": enc.n_clusters,
           "seed": seed, **metrics, "device": device, "epochs_used": epoch,
           "train_time_s": round(time.time() - t0, 1)}
    out_path = RESULTS_DIR / f"metrics_lstm_cond_{subset}_seed{seed}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")
    return out, model, enc, X_test, y_test


def load_train_only(subset):
    from data import load_dataset, add_rul_targets
    return add_rul_targets(load_dataset(subset)["train"])


def mc_dropout_predict(model: nn.Module, X: np.ndarray, n_samples: int = 100,
                       device: str = "auto"):
    """MC-Dropout inference: N stochastic forward passes -> mean & std per engine."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.train()  # keep dropout active
    Xt = torch.tensor(X).to(device)
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            samples.append(model(Xt).cpu().numpy())
    samples = np.stack(samples)  # (N, n_engines)
    mean = samples.mean(axis=0).clip(min=0)
    std = samples.std(axis=0)
    return mean, std, samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD002")
    ap.add_argument("--n-clusters", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mc-samples", type=int, default=100)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out, model, enc, X_test, y_test = train(
        args.subset, n_clusters=args.n_clusters, epochs=args.epochs,
        device=args.device, seed=args.seed)

    # MC-Dropout uncertainty on test set
    mean, std, _ = mc_dropout_predict(model, X_test, n_samples=args.mc_samples,
                                      device=args.device)
    # 90% CI coverage check
    z = 1.645
    lo, hi = mean - z * std, mean + z * std
    coverage = float(np.mean((y_test >= lo) & (y_test <= hi)))
    mean_ci_width = float(np.mean(2 * z * std))
    print(f"[{args.subset}] MC-Dropout: 90% CI coverage={coverage:.3f} "
          f"mean CI width={mean_ci_width:.2f} cycles")

    out["mc90_coverage"] = round(coverage, 3)
    out["mc_mean_ci_width"] = round(mean_ci_width, 2)
    out["mc_samples"] = args.mc_samples
    out_path = RESULTS_DIR / f"metrics_lstm_cond_{args.subset}_seed{args.seed}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    np.savez(RESULTS_DIR / f"mc_pred_{args.subset}_seed{args.seed}.npz",
             y_test=y_test, mean=mean, std=std)
    print(f"mc results saved -> {out_path}")


if __name__ == "__main__":
    main()
