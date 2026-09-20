"""Aggregate the FD002 ablation across seeds 7/123/99 under one protocol.

Every run aggregated here uses the same normalization mode (GMM-soft) and the
same seeds, so the four configurations are directly comparable.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["baseline", "nophys", "noorth", "full"]
SEEDS = [7, 123, 99]
TAG = "v16_ab"


def main():
    out = {}
    print("=== FD002 unified-protocol ablation (GMM-soft, 3 seeds) ===")
    for cfg in CONFIGS:
        rows = []
        for s in SEEDS:
            p = ROOT / "results" / "physdec" / f"metrics_FD002_{cfg}_{TAG}_seed{s}.json"
            if p.exists():
                d = json.loads(p.read_text(encoding="utf-8"))
                rows.append((d["rmse"], d["nasa_score"]))
            else:
                print(f"  MISSING {p.name}")
        if not rows:
            continue
        r = np.array([x[0] for x in rows])
        sc = np.array([x[1] for x in rows])
        res = {"rmse_mean": round(float(r.mean()), 3),
               "rmse_std": round(float(r.std(ddof=1)), 3),
               "score_mean": round(float(sc.mean()), 1),
               "score_std": round(float(sc.std(ddof=1)), 1),
               "per_seed": [{"seed": s, "rmse": rm, "score": scv}
                            for s, (rm, scv) in zip(SEEDS, rows)]}
        out[cfg] = res
        print(f"[{cfg:10s}] RMSE {res['rmse_mean']:.3f} ± {res['rmse_std']:.3f} | "
              f"Score {res['score_mean']:.1f} ± {res['score_std']:.1f}")

    outp = ROOT / "results" / "physdec" / "ablation_unified_stats.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nsaved -> {outp}")


if __name__ == "__main__":
    main()
