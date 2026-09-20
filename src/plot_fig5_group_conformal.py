"""Figure 5: group-conditional conformal prediction on DS02.

Four panels. (a),(b) plot coverage as points with error bars rather than as bars
on a truncated axis, so small differences are not exaggerated and the nominal
90% line reads as a reference rather than a baseline to exceed. (c),(d) compare
per-regime spread between marginal and group-conditional conformal for each
model; the legend names all four bars.

SOURCE DISCIPLINE: the CDN-RUL side is read from conformal_group_stats.json and
the LSTM side from conformal_group_compare.json -- exactly the two artifacts
Table 4 is built from, so the figure and the table cannot disagree.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import fig_style as FS
from fig_style import (
    apply_style, save,
    C_PHYSDEC, C_BASELINE, C_TARGET,
    LABEL_MARGINAL, LABEL_GROUP,
    annotate_bars,
)

apply_style()

MODEL_C = {"physdec": C_PHYSDEC, "lstm": C_BASELINE}
MODEL_N = {"physdec": "CDN-RUL", "lstm": "LSTM"}

RESULTS = Path(__file__).parent.parent / "results" / "physdec"

# --- CDN-RUL: latest rerun, same artifact Table 4 uses -------------------
stats = json.loads((RESULTS / "conformal_group_stats.json").read_text())
_s = stats["DS02"]
pga = _s["per_group_avg"]
phys_row = {
    "seed": -1,
    "group_cov": _s["group_conditional"]["coverage_mean"],
    "marginal_cov": _s["marginal"]["coverage_mean"],
    "group_width": _s["group_conditional"]["width_mean"],
    "marginal_width": _s["marginal"]["width_mean"],
    "per_group": {k: {"cov": v["coverage_mean"],
                      "marg": v.get("coverage_marginal_mean", v["coverage_mean"]),
                      "n": -1}
                  for k, v in pga.items()},
}

# --- LSTM baseline: the only artifact carrying its 5-seed runs ---------------
cmp_ = json.loads((RESULTS / "conformal_group_compare.json").read_text())

d = {"physdec": [phys_row], "lstm": cmp_["lstm"]}

# Fail loudly rather than silently drawing a stale series
_fig_spread_g = (max(v["cov"] for v in phys_row["per_group"].values())
                 - min(v["cov"] for v in phys_row["per_group"].values()))
assert abs(_fig_spread_g - 0.113) < 0.002, (
    f"Fig5 PhysDec group-cond spread {_fig_spread_g:.4f} no longer matches the "
    f"0.113 reported in Table 4 — table and figure have diverged again")


def _per_group(model):
    """Per-group means and s.d. of coverage across seeds."""
    gmap = {}
    for r in d[model]:
        for k, v in r["per_group"].items():
            gmap.setdefault(int(k), []).append((v["cov"], v["marg"]))
    groups = sorted(gmap.keys())
    grp = np.array([np.mean([x[0] for x in gmap[g]]) for g in groups])
    marg = np.array([np.mean([x[1] for x in gmap[g]]) for g in groups])
    grp_s = np.array([np.std([x[0] for x in gmap[g]], ddof=1)
                      if len(gmap[g]) > 1 else 0.0 for g in groups])
    marg_s = np.array([np.std([x[1] for x in gmap[g]], ddof=1)
                       if len(gmap[g]) > 1 else 0.0 for g in groups])
    return groups, grp, marg, grp_s, marg_s


fig, axes = plt.subplots(2, 2, figsize=(FS.FIG_WIDTH_IN, 6.8))


def _coverage_panel(ax, model, title, color):
    """Coverage as point + error markers (no truncated-bar artefact)."""
    groups, grp, marg, grp_s, marg_s = _per_group(model)
    x = np.arange(len(groups))
    ax.errorbar(x - 0.09, marg, yerr=marg_s, fmt="o", ms=3.8, lw=0.9,
                mfc="white", mec=color, mew=1.1, ecolor=color, elinewidth=0.8,
                capsize=2.0, label=LABEL_MARGINAL, alpha=0.85)
    ax.errorbar(x + 0.09, grp, yerr=grp_s, fmt="s", ms=3.8, lw=0.9,
                mfc=color, mec=color, ecolor=color, elinewidth=0.8,
                capsize=2.0, label=LABEL_GROUP)
    ax.axhline(0.90, color=C_TARGET, linestyle="--", lw=1.0, zorder=0)
    ax.set_xlabel("operating condition group")
    ax.set_ylabel("coverage")
    ax.set_title(title, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"g{g}" for g in groups])
    ax.set_ylim(0.74, 1.01)
    ax.legend(loc="lower right", fontsize=FS.PT_ANNOT, ncol=2)


# (a) PhysDec per-group coverage
_coverage_panel(axes[0, 0], "physdec", "(a) CDN-RUL", C_PHYSDEC)
# (b) LSTM per-group coverage
_coverage_panel(axes[0, 1], "lstm", "(b) LSTM (no cond. normalization)",
                C_BASELINE)

# (c) interval width — every bar labelled
ax = axes[1, 0]
_bars = {}
for i, m in enumerate(["physdec", "lstm"]):
    rows = d[m]
    gw = float(np.mean([r["group_width"] for r in rows]))
    mw = float(np.mean([r["marginal_width"] for r in rows]))
    ax.bar(i - 0.17, mw, 0.30, label=f"{MODEL_N[m]} — {LABEL_MARGINAL}",
           color=MODEL_C[m], alpha=0.45, edgecolor="none")
    ax.bar(i + 0.17, gw, 0.30, label=f"{MODEL_N[m]} — {LABEL_GROUP}",
           color=MODEL_C[m], alpha=0.95, edgecolor="none")
    _bars[m] = (mw, gw)
ax.set_ylabel("normalized interval width")
ax.set_title("(c) interval width (sharpness)", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["CDN-RUL", "LSTM"])
# No per-panel legend here. The four-entry legend is wide enough to span the panel, and
# while it sat inside (c) the label for the tallest bar was judged to collide and dropped,
# which trades an unreadable label for a missing one. The legend is drawn once for the
# figure instead; the y-limit only needs headroom for the tallest label now.
ax.set_ylim(0, float(max(_bars["physdec"] + _bars["lstm"])) * 1.18)
annotate_bars(ax,
              [0 - 0.17, 0 + 0.17, 1 - 0.17, 1 + 0.17],
              [_bars["physdec"][0], _bars["physdec"][1],
               _bars["lstm"][0], _bars["lstm"][1]],
              labels=[f"{_bars['physdec'][0]:.3f}", f"{_bars['physdec'][1]:.3f}",
                      f"{_bars['lstm'][0]:.3f}", f"{_bars['lstm'][1]:.3f}"],
              dy_frac=0.02, fontsize=FS.PT_ANNOT)
_c_handles = ax.get_legend_handles_labels()

# (d) per-group coverage spread: marginal vs group-conditional
#
# CONVENTION (must match Table 4 exactly): average each group's coverage
# across seeds FIRST, then take the range across groups. The alternative
# (compute a range per seed, then average the ranges) yields 0.113 / 0.074 for
# LSTM instead of the table's 0.089 / 0.052 — a silent figure-vs-table
# contradiction.
ax = axes[1, 1]
spread = {}
for m in ["physdec", "lstm"]:
    _, grp_m, marg_m, _, _ = _per_group(m)
    spread[m] = (float(marg_m.max() - marg_m.min()),
                 float(grp_m.max() - grp_m.min()))

# guard both models, not just PhysDec
assert abs(spread["physdec"][1] - 0.113) < 0.002, \
    f"Fig5 PhysDec group-cond spread {spread['physdec'][1]:.4f} != Table 4 (0.113)"
assert abs(spread["lstm"][0] - 0.089) < 0.002, \
    f"Fig5 LSTM marginal spread {spread['lstm'][0]:.4f} != Table 4 (0.089)"
assert abs(spread["lstm"][1] - 0.052) < 0.002, \
    f"Fig5 LSTM group-cond spread {spread['lstm'][1]:.4f} != Table 4 (0.052)"

for i, m in enumerate(["physdec", "lstm"]):
    ax.bar(i - 0.17, spread[m][0], 0.30,
           label=f"{MODEL_N[m]} — {LABEL_MARGINAL}", color=MODEL_C[m],
           alpha=0.45, edgecolor="none")
    ax.bar(i + 0.17, spread[m][1], 0.30,
           label=f"{MODEL_N[m]} — {LABEL_GROUP}", color=MODEL_C[m],
           alpha=0.95, edgecolor="none")
ax.set_ylabel("per-group coverage spread")
ax.set_title("(d) spread before / after group-conditional", fontsize=9)
ax.set_xticks([0, 1]); ax.set_xticklabels(["CDN-RUL", "LSTM"])
# Headroom for the tallest label; the legend is shared across (c) and (d) and drawn below.
ax.set_ylim(0, float(max(spread["physdec"] + spread["lstm"])) * 1.18)
annotate_bars(ax,
              [0 - 0.17, 0 + 0.17, 1 - 0.17, 1 + 0.17],
              [spread["physdec"][0], spread["physdec"][1],
               spread["lstm"][0], spread["lstm"][1]],
              labels=[f"{spread['physdec'][0]:.3f}", f"{spread['physdec'][1]:.3f}",
                      f"{spread['lstm'][0]:.3f}", f"{spread['lstm'][1]:.3f}"],
              dy_frac=0.02, fontsize=FS.PT_ANNOT)

# One legend for both bar panels. The same four entries were previously drawn twice, once
# inside each panel, and both times wide enough to compete with the value labels.
_handles, _labels = _c_handles
fig.legend(_handles, _labels, loc="lower center", ncol=2, fontsize=FS.PT_ANNOT,
           frameon=False, bbox_to_anchor=(0.5, -0.005))

fig.tight_layout(rect=[0, 0.055, 1, 1])
save(fig, "fig5_group_conformal")
