"""LSTM deep-learning model for RUL prediction (PyTorch, CUDA-capable)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR, SEED, SELECTED_SENSORS, WINDOW_SIZE
from data import prepare_sequences
from metrics import report_metrics


class LSTMRUL(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def train_lstm(subset: str = "FD001", epochs: int = 80, batch_size: int = 256,
               lr: float = 1e-3, patience: int = 10, device: str = "auto",
               seed: int = SEED):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_train, y_train, train_units, X_test, y_test, engine_ids = prepare_sequences(subset)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{subset}] device={device} | windows train {X_train.shape}, test {X_test.shape}")

    # Leakage-free split: partition ENGINES (not windows) 80/20
    engine_pool = np.unique(train_units)
    rng = np.random.default_rng(seed)
    val_engines = set(rng.choice(engine_pool, size=int(0.2 * len(engine_pool)), replace=False))
    val_mask = np.isin(train_units, list(val_engines))
    X_tr, y_tr = X_train[~val_mask], y_train[~val_mask]
    X_va, y_va = X_train[val_mask], y_train[val_mask]
    print(f"  split by engine: {len(engine_pool)} engines -> "
          f"train {X_tr.shape[0]} windows / valid {X_va.shape[0]} windows")

    X_tr = torch.tensor(X_tr).to(device)
    y_tr = torch.tensor(y_tr).to(device)
    X_va = torch.tensor(X_va).to(device)
    y_va = torch.tensor(y_va).to(device)

    model = LSTMRUL(n_features=len(SELECTED_SENSORS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
    loss_fn = nn.MSELoss()

    n = len(X_tr)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred = model(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va)
            val_loss = loss_fn(val_pred, y_va).item()
        sched.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
    torch.save(best_state, str(MODELS_DIR / f"lstm_{subset}_seed{seed}.pt"))
    if seed == SEED:  # backward-compat name for legacy evaluators
        torch.save(best_state, str(MODELS_DIR / f"lstm_{subset}.pt"))

    # Evaluate: last-window per engine -> one prediction per engine
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test).to(device)).cpu().numpy()
    y_pred = y_pred.clip(min=0)
    metrics = report_metrics(y_test, y_pred, tag=f"{subset} LSTM")

    out = {"model": "lstm", "subset": subset, "seed": seed, **metrics,
           "device": device, "epochs_used": epoch,
           "train_time_s": round(time.time() - t0, 1)}
    out_path = RESULTS_DIR / f"metrics_lstm_{subset}_seed{seed}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    train_lstm(args.subset, seed=args.seed)
