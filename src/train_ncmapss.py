"""Baseline training on N-CMAPSS DS02: XGBoost (rolling features) + LSTM.

Both models evaluated with RMSE + NASA Score on the 3 test units (11,14,15).
Leakage-free: scalers fit on train only; LSTM validation split by unit.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb

from common import RESULTS_DIR, MODELS_DIR, SEED
from metrics import report_metrics
from ncmapss_data import prepare_ncmapss, FEATURE_COLS, TRAIN_UNITS, TEST_UNITS

NCMAPSS_RESULTS = RESULTS_DIR / "ncmapss"
CACHE = RESULTS_DIR / "ncmapss_windows.npz"


class LSTMRUL(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def rolling_last_stats(X_windows: np.ndarray):
    """Aggregate per-window rolling stats for XGBoost (N, window, F) -> (N, F*5)."""
    feats = []
    for i in range(X_windows.shape[0]):
        w = X_windows[i]
        feats.append(np.concatenate([w.mean(0), w.std(0), w.min(0),
                                     w.max(0), w[-1]]))
    return np.asarray(feats, dtype=np.float32)


def train_xgb(X_train, y_train, X_test, y_test, subset="DS02"):
    t0 = time.time()
    X_tr_f = rolling_last_stats(X_train)   # (N, 20*5=100)
    X_te_f = rolling_last_stats(X_test)
    model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, objective="reg:squarederror",
                             random_state=SEED, n_jobs=-1,
                             early_stopping_rounds=50)
    model.fit(X_tr_f, y_train, eval_set=[(X_tr_f, y_train)], verbose=False)
    y_pred = model.predict(X_te_f).clip(min=0)
    metrics = report_metrics(y_test, y_pred, tag=f"{subset} XGBoost")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / f"xgb_{subset}.json"))
    out = {"model": "xgboost", "subset": subset, **metrics,
           "train_time_s": round(time.time() - t0, 1)}
    NCMAPSS_RESULTS.mkdir(parents=True, exist_ok=True)
    p = NCMAPSS_RESULTS / f"metrics_xgb_{subset}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {p}")
    return out


def train_lstm(X_train, y_train, train_units, X_test, y_test, subset="DS02",
               epochs=60, batch_size=256, lr=1e-3, patience=10, seed=SEED):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{subset}] device={device} | windows {X_train.shape}")

    # unit-level split (no leakage)
    pool = np.unique(train_units)
    rng = np.random.default_rng(seed)
    val_units = set(rng.choice(pool, size=max(1, int(0.2 * len(pool))), replace=False))
    val_mask = np.isin(train_units, list(val_units))
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
    X_va, y_va = X_train[val_mask], y_train[val_mask]
    print(f"  units {len(pool)} -> train {X_tr.shape[0]} / valid {X_va.shape[0]}")

    X_tr = torch.tensor(X_tr).to(device)
    y_tr = torch.tensor(y_tr).to(device)
    X_va = torch.tensor(X_va).to(device)
    y_va = torch.tensor(y_va).to(device)

    model = LSTMRUL(n_features=X_train.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    n = len(X_tr)
    best_val, best_state, bad = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        eloss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            eloss += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            vloss = loss_fn(model(X_va), y_va).item()
        sched.step(vloss)
        if vloss < best_val:
            best_val, best_state = vloss, {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop @ epoch {epoch}")
                break
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | train {eloss/nb:.4f} | val {vloss:.4f}")

    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, str(MODELS_DIR / f"lstm_{subset}_seed{seed}.pt"))
    if seed == SEED:  # backward-compat name for legacy evaluators
        torch.save(best_state, str(MODELS_DIR / f"lstm_{subset}.pt"))

    # batched inference to avoid OOM on large test sets (125k+ windows)
    model.eval()
    y_pred = []
    Xt = torch.tensor(X_test)
    with torch.no_grad():
        for i in range(0, len(Xt), 4096):
            y_pred.append(model(Xt[i:i + 4096].to(device)).cpu().numpy())
    y_pred = np.concatenate(y_pred).clip(min=0)
    metrics = report_metrics(y_test, y_pred, tag=f"{subset} LSTM")

    out = {"model": "lstm", "subset": subset, "seed": seed, **metrics, "device": device,
           "epochs_used": epoch, "train_time_s": round(time.time() - t0, 1)}
    NCMAPSS_RESULTS.mkdir(parents=True, exist_ok=True)
    p = NCMAPSS_RESULTS / f"metrics_lstm_{subset}_seed{seed}.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["xgb", "lstm", "both"])
    ap.add_argument("--window", type=int, default=50)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    X_train, y_train, units, X_test, y_test, eids = prepare_ncmapss(
        window=args.window, cache=None if args.no_cache else str(CACHE))
    print(f"N-CMAPSS DS02 | train windows {X_train.shape} ({len(np.unique(units))} units)"
          f" | test {X_test.shape} ({len(np.unique(eids))} units)")
    print(f"  test units: {np.unique(eids)} | y_test [{y_test.min()}, {y_test.max()}]")

    results = []
    if args.model in ("xgb", "both"):
        results.append(train_xgb(X_train, y_train, X_test, y_test))
    if args.model in ("lstm", "both"):
        results.append(train_lstm(X_train, y_train, units, X_test, y_test,
                                  seed=args.seed, batch_size=args.batch_size))

    summary = NCMAPSS_RESULTS / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsummary -> {summary}")


if __name__ == "__main__":
    main()
