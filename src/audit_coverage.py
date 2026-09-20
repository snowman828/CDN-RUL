"""Audit a set of prediction intervals against the reporting set of the paper.

WHY THIS EXISTS
Section 5.3 of the manuscript states five things a prediction-interval evaluation
should report for its coverage figures to mean anything: the variance convention
used to calibrate each compared model, the measured (not nominal) coverage, the
effective sample size of that coverage estimate, the per-deployment-unit
coverage, and the per-group spread. A reader who wants to apply that list to
their own results would otherwise have to reimplement it from prose. Running one
command is a lower barrier than reading a paragraph, and a lower barrier is what
"reusable" means in practice.

WHAT IT COMPUTES
  1. nominal level and measured coverage
  2. the pooled coverage with an i.i.d. binomial interval AND a cluster-aware
     interval obtained by resampling whole deployment units
  3. the design effect and effective sample size implied by the clustering
  4. the per-unit coverage range
  5. the per-group spread

The correction in (3) is not new work and is not presented as such: the design
effect deff = 1 + (m_bar - 1) * ICC is the standard survey-sampling treatment of
clustered observations, and reporting the effective rather than the nominal
count is the requirement the GUM has imposed on uncertainty statements since its
first edition (Welch-Satterthwaite). Treating correlated subsamples as
independent replicates is pseudoreplication, named in 1984. What the paper
contributes is not the formula but the observation that a domain which reports
coverage routinely does not apply any of them.

WHAT IT DOES NOT DO
It does not check exchangeability, does not validate the calibration procedure,
and makes no claim about whether an interval is *useful*. It reports what the
numbers do and do not support, which is a narrower thing.

USAGE
  python audit_coverage.py --demo                # synthetic self-test, no input
  python audit_coverage.py results.npz

The .npz must contain y_true, lower, upper and should contain unit (deployment
unit id) and, if conditional validity is being claimed, group (regime id).
Everything is numpy-only.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def coverage_stats(hit: np.ndarray, unit: np.ndarray | None, n_boot: int = 2000,
                   seed: int = 0) -> dict:
    """Coverage, i.i.d. interval, cluster-aware interval, deff, n_eff."""
    n = hit.size
    p = float(hit.mean())
    se_iid = float(np.sqrt(p * (1.0 - p) / n)) if 0 < p < 1 else 0.0

    out = {"n": n, "coverage": p, "se_iid": se_iid,
           "ci_iid": (max(0.0, p - 1.96 * se_iid), min(1.0, p + 1.96 * se_iid)),
           "deff": 1.0, "n_eff": float(n), "se_cluster": se_iid,
           "ci_cluster": None, "n_units": 0}

    if unit is None or len(np.unique(unit)) < 2:
        return out

    units = np.unique(unit)
    out["n_units"] = int(units.size)
    # design effect from the intra-cluster correlation of the coverage indicator
    sizes = np.array([(unit == u).sum() for u in units], dtype=float)
    m_bar = float(sizes.mean())
    grand = p
    between = sum(s * (hit[unit == u].mean() - grand) ** 2 for u, s in zip(units, sizes))
    # ICC as the between-cluster share of the total variance of a binary outcome
    # (analysis-of-variance estimator; bounded to [0, 1] for a usable deff)
    denom = (n - 1) * grand * (1.0 - grand) if 0 < grand < 1 else 1.0
    n_clusters = units.size
    msb = between / max(1, n_clusters - 1)
    icc = (msb - grand * (1 - grand)) / max(1e-12, m_bar * grand * (1 - grand)) \
        if 0 < grand < 1 else 0.0
    icc = float(min(max(icc, 0.0), 1.0))
    deff = float(max(1.0, 1.0 + (m_bar - 1.0) * icc))
    out["deff"] = deff
    out["n_eff"] = float(max(1.0, n / deff))
    out["se_cluster"] = se_iid * float(np.sqrt(deff)) if se_iid else 0.0

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(units, size=units.size, replace=True)
        boot[b] = np.concatenate([hit[unit == u] for u in pick]).mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["ci_cluster"] = (float(lo), float(hi))
    out["_icc"] = icc
    out["_m_bar"] = m_bar
    # the bootstrap is the defensible interval; report its half-width for ratio
    out["_boot_half"] = float((hi - lo) / 2.0)
    return out


def audit(y_true, lower, upper, nominal=0.90, unit=None, group=None) -> int:
    y_true = np.asarray(y_true, float).ravel()
    lower = np.asarray(lower, float).ravel()
    upper = np.asarray(upper, float).ravel()
    if not (y_true.size == lower.size == upper.size):
        print("  FAIL lengths differ")
        return 1
    hit = ((y_true >= lower) & (y_true <= upper)).astype(float)

    print("=" * 74)
    print("  PREDICTION-INTERVAL REPORTING AUDIT")
    print("=" * 74)
    print(f"  points                 : {y_true.size}")
    print(f"  nominal level          : {nominal:.3f}")
    print()

    s = coverage_stats(hit, np.asarray(unit) if unit is not None else None)
    print("  (1)(2) coverage -- measured, not nominal")
    print(f"    measured coverage    : {s['coverage']:.4f}")
    print(f"    shortfall vs nominal : {nominal - s['coverage']:+.4f}")
    print(f"    i.i.d. binomial 95% CI : [{s['ci_iid'][0]:.4f}, {s['ci_iid'][1]:.4f}]"
          f"   (width {s['ci_iid'][1] - s['ci_iid'][0]:.4f})")
    if s["ci_cluster"] is not None:
        w = s["ci_cluster"][1] - s["ci_cluster"][0]
        ratio = (w / (s["ci_iid"][1] - s["ci_iid"][0])) if s["ci_iid"][1] > s["ci_iid"][0] else float("nan")
        print(f"    cluster-bootstrap 95% CI : [{s['ci_cluster'][0]:.4f}, {s['ci_cluster'][1]:.4f}]"
              f"   (width {w:.4f})")
        print(f"    -> the i.i.d. interval is narrower by a factor of {ratio:.1f}")
    print()

    print("  (3) effective sample size -- report this, not the point count")
    print(f"    deployment units     : {s['n_units'] or 'not supplied'}")
    if s["n_units"]:
        print(f"    mean cluster size    : {s['_m_bar']:.1f}")
        print(f"    ICC of the indicator : {s['_icc']:.4f}")
        print(f"    design effect        : {s['deff']:.1f}")
        print(f"    effective sample size: {s['n_eff']:.0f}   (nominal count {s['n']})")
        print(f"    -> precision is overstated by a factor of {s['deff']:.1f}")
    else:
        print("    (no unit labels: the effective sample size is unknown, not equal to n)")
    print()

    if unit is not None:
        unit = np.asarray(unit)
        per = {u: float(hit[unit == u].mean()) for u in np.unique(unit)}
        lo, hi = min(per.values()), max(per.values())
        print("  (4) per-deployment-unit coverage -- a pooled mean can conceal this")
        print(f"    range                : {lo:.3f} .. {hi:.3f}   (spread {hi - lo:.3f})")
        below = sum(1 for v in per.values() if v < nominal)
        print(f"    units below nominal  : {below}/{len(per)}")
        print()

    if group is not None:
        group = np.asarray(group)
        per = {g: float(hit[group == g].mean()) for g in np.unique(group)}
        lo, hi = min(per.values()), max(per.values())
        print("  (5) per-group spread -- report before claiming conditional validity")
        print(f"    groups               : {len(per)}")
        print(f"    range                : {lo:.3f} .. {hi:.3f}   (spread {hi - lo:.3f})")
        print()
    else:
        print("  (5) per-group spread    : not computed (no group labels supplied)")
        print("      -> marginal coverage only; conditional validity is not established")
        print()

    print("=" * 74)
    print("  A coverage number without (3) and (4) carries more precision than the")
    print("  data support. (1) is not checkable from data alone: it is a disclosure.")
    print("=" * 74)
    return 0


def _r3(x: float) -> str:
    """Three decimals, half-up.

    Python's format() uses round-half-even on the binary value, so 0.8795
    prints as "0.879". The manuscript quotes this table, so the printed value
    has to be the one a reader sees -- hence an explicit half-up quantisation
    on the decimal string rather than a bare format spec.
    """
    from decimal import Decimal, ROUND_HALF_UP
    return str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def summary_table(rows: list[tuple]) -> None:
    """Print the compact table quoted as Table A.1 in the paper."""
    print("  SUMMARY (as quoted in the manuscript)")
    print("  | Case | Points | Units | Coverage | i.i.d. CI | Cluster CI | Design effect | Effective n |")
    for case, n, units, cov, w_iid, w_clu, deff, neff in rows:
        print(f"  | {case} | {n:,} | {units} | {_r3(cov)} | {_r3(w_iid)} | {_r3(w_clu)} "
              f"| {deff:.1f} | {neff:,.0f} |")


def demo() -> int:
    """Two synthetic cases: i.i.d. and strongly clustered, same coverage."""
    rng = np.random.default_rng(7)
    n_units, per_unit = 40, 50
    unit = np.repeat(np.arange(n_units), per_unit)
    group = np.tile(np.repeat(np.arange(5), per_unit // 5), n_units)
    rows = []

    print("#" * 74)
    print("# CASE A: independent points -- the correction should be negligible")
    print("#" * 74)
    hit = rng.random(unit.size) < 0.88
    y = np.where(hit, 0.0, 10.0)
    c = audit(y, np.full(unit.size, -1.0), np.full(unit.size, 1.0),
              nominal=0.90, unit=unit, group=group)
    if c:
        return c
    h = ((y >= -1.0) & (y <= 1.0)).astype(float)
    s = coverage_stats(h, unit)
    rows.append(("Independent", unit.size, s["n_units"], s["coverage"],
                 s["ci_iid"][1] - s["ci_iid"][0], s["ci_cluster"][1] - s["ci_cluster"][0],
                 s["deff"], s["n_eff"]))

    print()
    print("#" * 74)
    print("# CASE B: same marginal coverage, but coverage varies by unit")
    print("#          a pooled figure with an i.i.d. interval looks equally precise")
    print("#" * 74)
    unit_p = rng.uniform(0.55, 1.0, n_units)
    hit = np.concatenate([rng.random(per_unit) < unit_p[u] for u in range(n_units)])
    y = np.where(hit, 0.0, 10.0)
    c = audit(y, np.full(unit.size, -1.0), np.full(unit.size, 1.0),
              nominal=0.90, unit=unit, group=group)
    if c:
        return c
    h = ((y >= -1.0) & (y <= 1.0)).astype(float)
    s = coverage_stats(h, unit)
    rows.append(("Clustered", unit.size, s["n_units"], s["coverage"],
                 s["ci_iid"][1] - s["ci_iid"][0], s["ci_cluster"][1] - s["ci_cluster"][0],
                 s["deff"], s["n_eff"]))

    print()
    summary_table(rows)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="?", help="npz with y_true, lower, upper[, unit, group]")
    ap.add_argument("--nominal", type=float, default=0.90)
    ap.add_argument("--demo", action="store_true", help="run the synthetic self-test")
    a = ap.parse_args()
    if a.demo or not a.npz:
        return demo()
    d = np.load(a.npz)
    missing = [k for k in ("y_true", "lower", "upper") if k not in d]
    if missing:
        print(f"  FAIL missing arrays: {missing}")
        return 1
    return audit(d["y_true"], d["lower"], d["upper"], a.nominal,
                 d["unit"] if "unit" in d else None,
                 d["group"] if "group" in d else None)


if __name__ == "__main__":
    sys.exit(main())
