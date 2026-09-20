"""P2-3: Maintenance decision demonstration — CBM economic value of calibrated uncertainty.

Framework: expected-profit maximization (standard CBM economics).
For each test engine with calibrated predictive distribution RUL ~ N(mu, sigma)
(truncated at 0), choose the maintenance time t that maximizes

    Profit(t) = V * E[min(RUL, t)]  -  Cp * P(RUL > t)  -  Cf * P(RUL <= t)

where V = operating value per cycle, Cp = preventive cost, Cf = failure cost.
The V term makes the problem non-degenerate (maintaining at t=0 would forgo
V·E[RUL] of operating value).

Policies compared (realized value given true RUL y_i):
  - run-to-failure: value = V·y_i − Cf
  - point policy:   maintain at t = mu_i   (uncertainty ignored)
  - interval-aware: maintain at t*_i = argmax Profit(t) using calibrated sigma
Value of calibrated uncertainty = mean(Profit_cbm − Profit_point), reported in
cost units and as a % of the RTF loss that is recovered.

Truncated-normal analytics: with a = (0−mu)/sigma, b = (t−mu)/sigma,
  P(RUL<=t) = (Phi(b)−Phi(a)) / (1−Phi(a))
  E[min(RUL,t)] = t − ∫_0^t F(s) ds  (computed numerically via scipy.quad)
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR

CONFIGS = [
    {"cf": 100.0, "cp": 20.0, "v": 1.0, "label": "Cf=100, Cp=20, V=1"},
    {"cf": 100.0, "cp": 20.0, "v": 0.5, "label": "Cf=100, Cp=20, V=0.5 (low value)"},
    {"cf": 100.0, "cp": 50.0, "v": 1.0, "label": "Cf=100, Cp=50, V=1 (costly maint.)"},
]


def p_le(t: float, mu: float, sigma: float) -> float:
    """P(RUL <= t) with RUL truncated normal on [0, inf)."""
    if sigma <= 1e-9:
        return 1.0 if t >= mu else 0.0
    a, b = -mu / sigma, (t - mu) / sigma
    denom = 1 - stats.norm.cdf(a)
    if denom < 1e-9:
        return 0.0
    return (stats.norm.cdf(b) - stats.norm.cdf(a)) / denom


def e_min(t: float, mu: float, sigma: float) -> float:
    """E[min(RUL, t)] for truncated-normal RUL on [0, inf) — closed form.

    For X~N(mu,sigma²) truncated to [0,inf):
      E[min(X,t)] = [mu·(Φ(β)−Φ(α)) + sigma·(φ(α)−φ(β)) + t·(1−Φ(β))] / (1−Φ(α))
    with α = −mu/sigma, β = (t−mu)/sigma.
    """
    if t <= 0:
        return 0.0
    if sigma <= 1e-9:
        return min(t, max(mu, 0.0))
    a, b = -mu / sigma, (t - mu) / sigma
    phi_a, phi_b = stats.norm.pdf(a), stats.norm.pdf(b)
    Phi_a, Phi_b = stats.norm.cdf(a), stats.norm.cdf(b)
    denom = 1 - Phi_a
    if denom < 1e-9:
        return t
    num = mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + t * (1 - Phi_b)
    return float(num / denom)


def profit(t: float, mu: float, sigma: float, cf: float, cp: float, v: float) -> float:
    return v * e_min(t, mu, sigma) - cp * (1 - p_le(t, mu, sigma)) - cf * p_le(t, mu, sigma)


def optimal_t(mu: float, sigma: float, cf: float, cp: float, v: float) -> float:
    """Maximize Profit(t) over t via coarse-fine grid."""
    tmax = max(10.0, 5 * (mu + 2 * sigma))
    ts = np.linspace(0.0, min(400.0, tmax), 500)
    vals = np.array([profit(t, mu, sigma, cf, cp, v) for t in ts])
    i0 = int(np.argmax(vals))
    lo = ts[max(0, i0 - 3)]
    hi = ts[min(len(ts) - 1, i0 + 3)]
    ts2 = np.linspace(lo, hi, 300)
    vals2 = np.array([profit(t, mu, sigma, cf, cp, v) for t in ts2])
    return float(ts2[np.argmax(vals2)])


def load_predictions(subset: str, tag: str):
    p = RESULTS_DIR / "physdec" / f"mc_{subset}_{tag}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_test"], d["mean"], d["std"]


def main():
    subset, tag = "FD002", "full_v16_ab_seed42"
    data = load_predictions(subset, tag)
    if data is None:
        data = load_predictions(subset, "full_v16")
    if data is None:
        subset, tag = "DS02", "full_v13"
        data = load_predictions(subset, tag)
    y_true, mu, sigma = data
    n = len(y_true)
    print(f"[{subset}/{tag}] {n} engines | RUL [{y_true.min():.0f}, {y_true.max():.0f}]")

    results = {}
    for cfg in CONFIGS:
        cf, cp, v = cfg["cf"], cfg["cp"], cfg["v"]
        # run-to-failure realized value
        val_rtf = float(np.mean(v * y_true - cf))
        # point policy: maintain at mu
        val_point = float(np.mean([
            v * y - cf if m >= y else v * m - cp for m, y in zip(mu, y_true)]))
        # interval-aware: optimal t* per engine
        tstar = np.array([optimal_t(m, s, cf, cp, v) for m, s in zip(mu, sigma)])
        val_cbm = float(np.mean([
            v * y - cf if t >= y else v * t - cp for t, y in zip(tstar, y_true)]))
        # value of calibrated uncertainty: extra profit over point policy
        val_unc = val_cbm - val_point
        rtf_loss = max(0.0, -val_rtf)  # RTF expected loss magnitude
        recover = 100 * val_unc / rtf_loss if rtf_loss > 0 else float("nan")
        frac_early = float(np.mean(tstar < 5))
        print(f"\n  {cfg['label']}")
        print(f"    mean value  RTF {val_rtf:8.2f} | point {val_point:8.2f} | "
              f"CBM {val_cbm:8.2f}")
        print(f"    calibrated-uncertainty value: {val_unc:+.2f} cost units "
              f"({recover:+.1f}% of RTF loss recovered)")
        print(f"    engines with t*<5 cycles: {frac_early:.1%} (sanity)")
        results[cfg["label"]] = {
            "val_rtf": round(val_rtf, 2), "val_point": round(val_point, 2),
            "val_cbm": round(val_cbm, 2), "value_uncertainty": round(val_unc, 2),
            "rtf_loss_recovered_pct": round(recover, 2),
            "frac_tstar_lt5": round(frac_early, 4),
        }

    outp = RESULTS_DIR / "physdec" / "maintenance_decision.json"
    outp.write_text(json.dumps(
        {"subset": subset, "tag": tag, "configs": results}, indent=2), encoding="utf-8")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
