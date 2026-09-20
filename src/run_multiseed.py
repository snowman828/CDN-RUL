"""Multi-seed statistical evaluation for the paper's experimental section.

Runs the FINAL method config on each benchmark subset with N seeds, saves
per-seed metrics, then aggregates mean±std + pairwise significance (vs the
best baseline from final_summary.json).

Configurations:
  FD001/FD003 (single-condition)  -> n_clusters=1 GMM-soft (≈global norm)
  FD002/FD004 (discrete regimes)  -> n_clusters=6 GMM-soft
  DS02       (continuous)         -> n_clusters=0 global norm
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAINER = ROOT / "src" / "train_physdec.py"
SEEDS = [42, 2024, 7, 123, 99]
N_SEEDS = len(SEEDS)

CONFIGS = {
    "FD001": dict(n_clusters=1, norm_mode="soft", window=30, epochs=60),
    "FD002": dict(n_clusters=6, norm_mode="soft", window=30, epochs=60),
    "FD003": dict(n_clusters=1, norm_mode="soft", window=30, epochs=60),
    "FD004": dict(n_clusters=6, norm_mode="soft", window=30, epochs=60),
    "DS02": dict(n_clusters=0, norm_mode="hard", window=50, epochs=40),
}


def run_seed(subset: str, seed: int) -> dict:
    cfg = CONFIGS[subset]
    tag = f"seed{seed}"
    cmd = [PY, str(TRAINER), "--subset", subset, "--tag", tag,
           "--window", str(cfg["window"]), "--epochs", str(cfg["epochs"]),
           "--n-clusters", str(cfg["n_clusters"]), "--norm-mode", cfg["norm_mode"],
           "--mc-samples", "30", "--seed", str(seed)]
    if subset == "DS02":
        cmd += ["--batch-size", "128"]

    # Record both candidate paths' mtimes BEFORE the run. The exit code alone
    # cannot be trusted here: this environment intermittently returns
    # 0xC0000409 (STATUS_FAIL_FAST) during interpreter teardown, i.e. AFTER the
    # metrics have been written. The old code printed the code and carried on,
    # then fell back to whichever file happened to exist -- so a run that died
    # before writing silently contributed the PREVIOUS run's numbers to the
    # aggregate. Artifact freshness is the honest signal: if neither file was
    # touched by this run, the run produced nothing and must abort the sweep.
    # Filenames now follow the committed convention metrics_{subset}_{tag}.json.
    # train_physdec no longer appends _seed{seed}, so tag=seed{seed} reproduces
    # the published names and a re-run overwrites the same file a reader would
    # compare against. The legacy name is still accepted so this driver keeps
    # working against a tree produced by the older trainer.
    out_dir = ROOT / "results" / "physdec"
    cand = [out_dir / f"metrics_{subset}_{tag}.json",
            out_dir / f"metrics_{subset}_{tag}_seed{seed}.json"]
    before = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in cand}

    r = subprocess.run(cmd, capture_output=True, text=True)

    touched = [p for p in cand
               if p.exists() and p.stat().st_mtime_ns != before[p]]
    if not touched:
        raise RuntimeError(
            f"seed {seed} of {subset} produced no fresh metrics file "
            f"(rc={r.returncode}). Refusing to fall back to a stale file: "
            f"that is how an old result silently enters an aggregate. "
            f"stderr tail: {r.stderr[-400:]!r}")

    out = touched[0]
    if r.returncode not in (0, 127, 3221226505):
        # A code we do not recognise, but the artifact is fresh, so the run did
        # its work. Report it and keep going rather than discarding valid output.
        print(f"  [seed {seed}] note: rc={r.returncode} but {out.name} is fresh")
    return json.loads(out.read_text(encoding="utf-8"))


def aggregate(subset: str):
    rows = [run_seed(subset, s) for s in SEEDS]
    rmses = [r["rmse"] for r in rows]
    scores = [r["nasa_score"] for r in rows]
    covs = [r.get("mc90_coverage") for r in rows]
    res = {
        "subset": subset,
        "n_seeds": N_SEEDS,
        "rmse_mean": round(float(np.mean(rmses)), 3),
        "rmse_std": round(float(np.std(rmses, ddof=1)), 3),
        "score_mean": round(float(np.mean(scores)), 1),
        "score_std": round(float(np.std(scores, ddof=1)), 1),
        "mc90_mean": round(float(np.mean([c for c in covs if c is not None])), 3),
        "per_seed": [{"seed": s, "rmse": r, "score": sc}
                     for s, r, sc in zip(SEEDS, rmses, scores)],
    }
    print(f"[{subset}] RMSE {res['rmse_mean']:.3f}±{res['rmse_std']:.3f} | "
          f"Score {res['score_mean']:.1f}±{res['score_std']:.1f} | "
          f"MC90 {res['mc90_mean']}")
    return res


def main():
    subsets = sys.argv[1:] or list(CONFIGS.keys())
    results = {}
    for ds in subsets:
        results[ds] = aggregate(ds)
    out = ROOT / "results" / "physdec" / "multiseed_stats.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved -> {out}")

    # pairwise significance vs best baseline (from final_summary.json)
    try:
        fin = json.loads((ROOT / "results" / "physdec" / "final_summary.json")
                         .read_text(encoding="utf-8"))
        print("\n=== vs best baseline (RMSE one-sample t-test on seed diffs) ===")
        for ds in subsets:
            base = fin["baselines"].get(ds, {}).get("rmse")
            if base is None:
                continue
            diffs = [base - s["rmse"] for s in results[ds]["per_seed"]]
            t, p = stats.ttest_1samp(diffs, 0)
            better = "PhysDec better" if np.mean(diffs) > 0 else "baseline better"
            print(f"[{ds}] baseline {base:.3f} | mean diff {np.mean(diffs):+.3f} "
                  f"| t={t:.2f} p={p:.3f} ({better})")
    except Exception as e:
        print(f"(significance skipped: {e})")


if __name__ == "__main__":
    main()
