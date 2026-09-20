"""XGBoost baseline for RUL prediction using rolling-window aggregated features."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR, MODELS_DIR, SEED
from data import prepare_tabular
from metrics import report_metrics


def main(subset: str = "FD001"):
    t0 = time.time()
    X_train, y_train, X_test, y_test = prepare_tabular(subset, lookback=30)
    print(f"[{subset}] tabular shapes: train {X_train.shape}, test {X_test.shape}")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=SEED,
        n_jobs=-1,
        early_stopping_rounds=50,
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)

    y_pred = model.predict(X_test).clip(min=0)
    metrics = report_metrics(y_test, y_pred, tag=f"{subset} XGBoost")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / f"xgb_{subset}.json"))

    out = {"model": "xgboost", "subset": subset, **metrics,
           "train_time_s": round(time.time() - t0, 1)}
    out_path = RESULTS_DIR / f"metrics_xgb_{subset}.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"saved -> {out_path}")
    return out


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "FD001"
    main(ds)
