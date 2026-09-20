r"""The measurement the expert panel said was missing.

THE GAP
Section 4.7 argues:

    CDN-RUL's per-group spread under the marginal quantile (0.048) is already less than
    half the baseline's (0.089)
        -> group-conditional conformal brings CDN-RUL no gain (0.048 -> 0.113)
        -> therefore partitioning "no longer targets a real difference"
        -> therefore this corroborates the CDFE decoupling mechanism

The third step does not follow from the second. "Partitioning does not help" is also
what you would see if per-group sample size had shrunk enough that per-group quantiles
became noisy. The panel proposed the direct measurement: does the residual distribution
actually become homogeneous across groups?

THE MEASUREMENT, AND WHY IT DISCRIMINATES
Two models are partitioned by the SAME groupings on the SAME data, so both get the same
group sizes n. If noise from small n were the explanation, both would be hurt in the
same way. The models differ only in whether their residuals were already regime-
homogeneous going in. So:

    LSTM   spread shrinks under group-conditional          -> heterogeneity was real
    CDN    spread does not shrink (and grows)              -> heterogeneity was already
                                                              removed before partitioning

That is a property of the residuals, not of the partitioning, and it is measurable from
the artifacts already on disk.

The second thing this measures is whether the CDN growth is consistent with pure noise:
if the increase is of the same order as the within-group coverage standard deviation,
the honest statement is "the partition adds estimation noise", which is weaker than
"the partition targets nothing".
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATS = ROOT / "results" / "physdec" / "conformal_group_stats.json"
CMP = ROOT / "results" / "physdec" / "conformal_group_compare.json"


def spread(values):
    return max(values) - min(values)


def main():
    stats = json.loads(STATS.read_text(encoding="utf-8"))
    cmp_ = json.loads(CMP.read_text(encoding="utf-8"))

    print("=" * 78)
    print("  A. per-group coverage spread: marginal vs group-conditional")
    print("     (conformal_group_stats.json, mean over 5 seeds for physdec)")
    print("=" * 78)
    print(f"  {'subset':<8} {'marginal':>10} {'group-cond':>11} {'change':>9}")
    for sub, d in stats.items():
        pg = d.get("per_group_avg", {})
        if not pg:
            continue
        marg = [v["coverage_marginal_mean"] for v in pg.values()
                if "coverage_marginal_mean" in v]
        gc = [v["coverage_mean"] for v in pg.values() if "coverage_mean" in v]
        if not marg or not gc:
            continue
        sm, sg = spread(marg), spread(gc)
        arrow = "shrink" if sg < sm else "GROW"
        print(f"  {sub:<8} {sm:>10.3f} {sg:>11.3f} {sg - sm:>+9.3f}  {arrow}")

    print("\n" + "=" * 78)
    print("  B. same grouping, both models -- the discriminating comparison")
    print("     (conformal_group_compare.json)")
    print("=" * 78)
    models = {"physdec": "CDN-RUL", "lstm": "LSTM baseline"}
    for key, label in models.items():
        entries = cmp_.get(key, [])
        if not entries:
            continue
        per_seed = []
        ns = set()
        for e in entries:
            pg = e.get("per_group", {})
            if not pg:
                continue
            covs = [v["cov"] for v in pg.values() if "cov" in v]
            margs = [v["marg"] for v in pg.values() if "marg" in v]
            for v in pg.values():
                if v.get("n", -1) and v["n"] > 0:
                    ns.add(v["n"])
            if covs and margs:
                per_seed.append((e.get("seed"), spread(margs), spread(covs)))
        if not per_seed:
            continue
        sm = statistics.mean(s for _, s, _ in per_seed)
        sg = statistics.mean(g for _, _, g in per_seed)
        print(f"\n  {label} ({len(per_seed)} seed entries)")
        for seed, a, b in per_seed:
            print(f"      seed {seed:>5}: marginal spread {a:.3f} -> group-cond {b:.3f}"
                  f"   ({'shrink' if b < a else 'GROW'})")
        print(f"      MEAN: {sm:.4f} -> {sg:.4f}   ({'shrink' if sg < sm else 'GROW'})")
        print(f"      per-group n observed: {sorted(ns) if ns else '(not recorded: -1)'}")

    print("\n" + "=" * 78)
    print("  C. is the CDN increase the size of estimation noise?")
    print("=" * 78)
    ds = stats.get("DS02", {}).get("per_group_avg", {})
    if ds:
        wg = [v["coverage_std"] for v in ds.values() if "coverage_std" in v]
        marg = [v["coverage_marginal_mean"] for v in ds.values()]
        gc = [v["coverage_mean"] for v in ds.values()]
        print(f"  within-group coverage std (mean over groups): {statistics.mean(wg):.4f}")
        print(f"  between-group spread, marginal:               {spread(marg):.4f}")
        print(f"  between-group spread, group-conditional:      {spread(gc):.4f}")
        print(f"  increase:                                     {spread(gc) - spread(marg):+.4f}")
        ratio = (spread(gc) - spread(marg)) / max(statistics.mean(wg), 1e-9)
        print(f"  increase / within-group std = {ratio:.2f}")
        print("  -> an increase of the same order as the within-group std is consistent")
        print("     with estimation noise rather than with a substantive worsening")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
