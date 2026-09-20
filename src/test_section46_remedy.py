"""Is the prescription in Section 4.6 worth anything? Test it.

THE GAP
Section 4.6 ends: "the remedy is calibration on data drawn from the deployment
distribution -- engine-level or shift-aware calibration [56] -- rather than a
wider ad-hoc scaling factor." The paper diagnoses the under-coverage, prescribes
a fix, and never tests the fix. A referee is entitled to call that an opinion.

WHAT THIS COMPUTES, PER SEED
  (a) current protocol    global split-conformal quantile from the validation
                          engines, applied to the test windows. Must reproduce
                          Table 3, or the harness is not trustworthy and none of
                          the numbers below mean anything.
  (b) oracle per-engine   the quantile each test engine would need if its own
                          labels were available. NOT deployable -- it is the
                          ceiling that bounds what any remedy could achieve, and
                          therefore the test of whether the question is even
                          worth asking.
  (c) shift-aware         weighted split conformal: a logistic classifier is
                          fitted on window features to separate validation from
                          test (labels not used), density-ratio weights
                          w = p/(1-p) reweight the validation residuals, and the
                          quantile is taken on the weighted empirical distribution.
                          This is deployable: it needs unlabelled test features.

Pre-registered reading of the outcome:
  - If (b) is near nominal, the diagnosis is right and a remedy is possible.
  - If (c) closes most of the gap to (b), the prescription is validated.
  - If (c) does not, the honest result is that the prescription is harder than
    Section 4.6 implies, and that gets written instead -- a tested negative is
    still worth more than an untested recommendation.

Order of operations is deliberate: reproduce first, extend second.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

WINDOW = 50
ALPHA = 0.10


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Quantile of the weighted empirical distribution (standard interpolation)."""
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.quantile(values, q))
    cw = cw / cw[-1]
    return float(np.interp(q, cw, v))


CFG_DS02 = dict(n_clusters=0, norm_mode="hard", window=50)


def run_seed(seed: int) -> dict:
    from conformal_eval import physdec_mc, conformal_widths  # noqa: E402

    mu_v, sd_v, y_v, mu_t, sd_t, y_t, g_v, g_t, u_t = physdec_mc(
        "DS02", seed, CFG_DS02["window"], CFG_DS02["n_clusters"], CFG_DS02["norm_mode"],
        return_units=True)

    r_v = np.abs(y_v - mu_v) / np.maximum(sd_v, 1e-8)
    r_t = np.abs(y_t - mu_t) / np.maximum(sd_t, 1e-8)
    n = r_v.size
    lvl = min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n)

    # (a) current protocol -- the paper's own function, so this line reproduces
    #     Table 3 by construction rather than by reimplementation
    cov_a, width_a, q_global = conformal_widths(mu_v, sd_v, y_v, mu_t, sd_t, y_t)
    cov_a, q_global = float(cov_a), float(q_global)

    # (b) oracle per-engine ceiling
    per = {}
    for e in np.unique(u_t):
        m = u_t == e
        qe = float(np.quantile(r_t[m], lvl, method="higher"))
        per[str(e)] = {"n": int(m.sum()), "q": qe, "cov": float(np.mean(r_t[m] <= qe))}
    cov_b = float(np.mean([v["cov"] for v in per.values()]))

    # (c) deployable shift-aware reweighting on window features
    from train_physdec import load_data
    d = load_data("DS02", window=WINDOW, seed=seed)
    Xv, Xt = d["X_val"], d["X_test"]

    def feats(X):
        X = np.asarray(X, dtype=np.float64)
        return np.concatenate([X.mean(axis=1), X.std(axis=1),
                               X[:, -1, :], X[:, 0, :]], axis=1)

    Fv, Ft = feats(Xv), feats(Xt)
    mean, std = Fv.mean(0), Fv.std(0) + 1e-8
    Fv, Ft = (Fv - mean) / std, (Ft - mean) / std
    X = np.vstack([Fv, Ft])
    y = np.r_[np.zeros(len(Fv)), np.ones(len(Ft))]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    tr, te = idx[:half], idx[half:]

    w = np.zeros(Fv.shape[1] + 1)
    Xtr = np.c_[np.ones(len(tr)), X[tr]]
    ytr = y[tr]
    for _ in range(400):  # plain logistic regression, full batch
        p = 1.0 / (1.0 + np.exp(-Xtr @ w))
        g = Xtr.T @ (p - ytr) / len(ytr) + 1e-4 * w
        w -= 1.5 * g
    acc = float(np.mean((1.0 / (1.0 + np.exp(-np.c_[np.ones(len(te)), X[te]] @ w)) > 0.5) == y[te]))

    p_v = 1.0 / (1.0 + np.exp(-np.c_[np.ones(len(Fv)), Fv] @ w))
    p_v = np.clip(p_v, 1e-6, 1 - 1e-6)
    wts = (p_v / (1 - p_v)) * (len(Ft) / len(Fv))
    wts = wts / wts.mean()
    cov_c = float(np.mean(r_t <= weighted_quantile(r_v, wts, lvl)))

    return {
        "seed": seed, "n_val": int(n), "n_test": int(r_t.size),
        "domain_clf_acc": acc,
        "current": cov_a, "q_current": q_global,
        "oracle_per_engine": cov_b, "per_engine": per,
        "shift_weighted": cov_c,
        "ratio_test_val_residual": float(np.mean(r_t) / np.mean(r_v)),
    }


def main() -> int:
    seeds = [int(s) for s in sys.argv[1:]] or [42, 2024, 7, 123, 99]
    rows = []
    for s in seeds:
        try:
            r = run_seed(s)
        except Exception as e:
            print(f"  seed {s}: FAILED {type(e).__name__}: {e}")
            continue
        rows.append(r)
        print(f"  seed {s:>4}  current {r['current']:.4f}  "
              f"shift-weighted {r['shift_weighted']:.4f}  "
              f"oracle {r['oracle_per_engine']:.4f}  "
              f"(domain clf acc {r['domain_clf_acc']:.3f}, "
              f"test/val residual ratio {r['ratio_test_val_residual']:.2f})")
    if not rows:
        return 1
    for k, lab in (("current", "current"), ("shift_weighted", "shift-weighted"),
                   ("oracle_per_engine", "oracle")):
        v = [r[k] for r in rows]
        print(f"  MEAN {lab:15}: {np.mean(v):.4f} +/- {np.std(v):.4f}")
    out = ROOT / "results" / "physdec" / "remedy_test.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
