"""Evaluation metrics for RUL prediction: RMSE + NASA asymmetric scoring function."""
import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """NASA PHM08 asymmetric scoring function (Saxena et al. 2008).

    Over-estimation (predicting later than actual failure) is penalized less
    than under-estimation (predicting failure too early).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(score))


def report_metrics(y_true: np.ndarray, y_pred: np.ndarray, tag: str = "") -> dict:
    """Compute and print both metrics; return as dict."""
    r = rmse(y_true, y_pred)
    s = nasa_score(y_true, y_pred)
    label = f"[{tag}] " if tag else ""
    print(f"{label}RMSE      = {r:.3f}")
    print(f"{label}NASA Score = {s:.1f}")
    return {"rmse": round(r, 3), "nasa_score": round(s, 1)}
