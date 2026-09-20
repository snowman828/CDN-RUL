"""Generate Figures 1-4 (Figure 5 lives in plot_fig5_group_conformal.py).

  Fig 1  Architecture. Block layout and colour weights give the decoder visual
         prominence as the core component, ahead of the training-time prior.
  Fig 2  Accuracy benchmark. The NASA score spans 6.2e2 (FD001) to 8.4e4
         (DS02), so panel (b) uses a log axis with point+error markers. Linear
         bars would collapse the four C-MAPSS entries to slivers; log-scaled
         bars would instead misrepresent their baseline.
  Fig 3  Interval width across the four C-MAPSS subsets.
  Fig 4  Reliability diagram. The y-limit is derived from the data, because the
         coverage band on FD004 extends below 0.40.

Value labels are placed through annotate_bars() from fig_style, which skips any
label that would collide with the legend.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS_DIR
import fig_style as FS
from fig_style import (
    apply_style, save, annotate_bars,
    C_PHYSDEC, C_BASELINE, C_GRAY, C_TARGET, C_LIGHT,
    LABEL_PHYSDEC, LABEL_MARGINAL, LABEL_GROUP,
    LEGEND_LOC, LEGEND_LOC_LEFT,
)

apply_style()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def fig1_architecture():
    """CDN-RUL architecture schematic.

    Redrawn against the implementation in physdec_model.py, not against the prose.
    The previous version carried three content errors a referee could check:

      * fusion() was labelled h_T (+) z_c, but the model does
        cat(LSTM(z_d), mean_t(z_c)) -> fusion: the LSTM consumes z_d, and z_c
        enters at the fusion stage after pooling over time.
      * z_c = enc_c(x . g) was drawn without any edge carrying x into enc_c.
      * the monotonicity term, a training-time loss acting on y_hat, was drawn as
        if the decoder fed it, with no indication that it is not part of inference.

    Edges are now labelled with the tensor they carry, training-time terms are
    separated into their own band, and the orthogonality regulariser -- the
    mechanism the paper is actually about -- is drawn rather than omitted.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig = plt.figure(figsize=(FS.FIG_WIDTH_IN, 3.30))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    C_IN = "#F2E7D5"      # warm input
    C_CDFE = "#D6E4F0"    # blue tint — CDFE
    C_LSTM = "#E4DFEC"    # purple tint — decoder (core component)
    C_PHY = "#DFECDF"     # green tint — training-time terms
    C_OUT = "#F2E7D5"     # warm output
    EC = {"in": "#B07A4A", "cdfe": "#1F4E79", "lstm": "#6C4F8B",
          "phy": "#3F7D3F", "out": "#B07A4A", "arr": "#4A4A4A"}

    def box(x, y, w, h, text, fc, ec, fs=FS.PT_LABEL, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    fc=fc, ec=ec, lw=lw,
                                    mutation_aspect=0.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=FS.pt(fs), color="#1a1a1a", linespacing=1.5)

    def arr(x1, y1, x2, y2, label=None, color=EC["arr"], lw=0.9,
            style="-|>", ls="-", lx=None, ly=None, lha="center", lva="center"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, lw=lw, color=color,
                                     linestyle=ls, shrinkA=0, shrinkB=0))
        if label:
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2,
                    label, ha=lha, va=lva, fontsize=FS.pt(FS.PT_ANNOT),
                    color=color,
                    bbox=dict(fc="white", ec="none", pad=0.9))

    # ---------------------------------------------------------- inputs ------
    box(1.0, 33.5, 17.0, 8.5, "Sensors $X$\n$(B,T,S)$", C_IN, EC["in"])
    box(1.0, 22.0, 17.0, 8.5, "Op-settings $C$\n$(B,T,C)$", C_IN, EC["in"])

    # ------------------------------------------------------------ CDFE ------
    ax.add_patch(FancyBboxPatch((20.5, 20.5), 39.0, 22.5,
                                boxstyle="round,pad=0.5", fc="none",
                                ec=EC["cdfe"], lw=1.3, ls="--",
                                mutation_aspect=0.5))
    ax.text(40.0, 41.6, "CDFE — condition-decoupled encoder", ha="center",
            va="center", fontsize=FS.pt(FS.PT_TITLE), color=EC["cdfe"])

    box(22.5, 33.0, 34.5, 7.5,
        "GMM-soft per-regime normalization\n"
        "$z_d=\\sum_k w_k(t)\\,(x-\\mu_k)/\\sigma_k$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)
    box(22.5, 22.0, 13.0, 6.5, "gate $g(t)$\n$=\\sigma(W_g c)$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)
    ax.add_patch(FancyBboxPatch((37.0, 24.0), 4.0, 3.4,
                                boxstyle="round,pad=0.2", fc="white",
                                ec=EC["cdfe"], lw=0.9, mutation_aspect=0.5))
    ax.text(39.0, 25.7, "$\\times$", ha="center", va="center",
            fontsize=FS.pt(FS.PT_LABEL))
    box(42.5, 22.0, 14.0, 6.5, "$z_c=\\mathrm{enc}_c(x\\cdot g)$\n$(B,T,16)$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)

    # --------------------------------------------------------- decoder ------
    box(63.0, 32.5, 16.5, 8.0, "LSTM (2 layers)", C_LSTM, EC["lstm"],
        lw=1.3)
    box(63.0, 22.0, 16.5, 7.0,
        "fusion\n$[\\,h_t \\,;\\, \\bar{z}_c\\,]$", C_LSTM, EC["lstm"], lw=1.3)

    # ---------------------------------------------------------- output ------
    box(83.5, 22.0, 15.5, 18.5,
        "point $\\hat{y}$ and $\\mu,\\sigma$\n"
        "MC-Dropout\n"
        "$\\rightarrow$ split conformal\n"
        "$\\hat{y}\\pm q\\,\\sigma$ (90%)", C_OUT, EC["out"])

    # ------------------------------------------- training-time terms --------
    box(24.0, 3.5, 27.5, 8.0,
        "orthogonality\n"
        "$\\mathcal{L}_{\\mathrm{orth}}=\\langle z_d,\\,z_c\\rangle$",
        C_PHY, EC["phy"], fs=FS.PT_ANNOT, lw=0.8)
    box(60.0, 3.5, 34.5, 8.0,
        "monotonicity prior\n"
        "$\\mathcal{L}_{\\mathrm{phy}}=\\sum \\mathrm{ReLU}(\\hat{y}_t-\\hat{y}_{t+1})$",
        C_PHY, EC["phy"], fs=FS.PT_ANNOT, lw=0.8)
    ax.text(58.0, 14.4, "training-time terms (not used at inference)",
            ha="center", va="center", fontsize=FS.pt(FS.PT_ANNOT),
            color=EC["phy"], style="italic")

    # ---------------------------------------------------------- edges -------
    arr(18.0, 37.8, 22.5, 36.8, label="$x$", lx=20.2, ly=38.9)
    arr(18.0, 26.2, 22.5, 25.5, label="$c$", lx=20.2, ly=24.4)
    # x is needed by enc_c as well, not only by the normalizer
    arr(18.0, 35.0, 39.0, 27.4, label="$x$", lx=30.5, ly=32.6)
    arr(35.5, 25.2, 37.0, 25.6, label="$g$", lx=36.3, ly=23.6)
    arr(41.0, 25.7, 42.5, 25.7)
    arr(56.5, 25.2, 63.0, 25.6, label="$x\\cdot g$", lx=59.8, ly=23.4)
    arr(56.5, 36.8, 63.0, 36.5, label="$z_d$", lx=59.8, ly=38.6)
    arr(71.2, 32.5, 71.2, 29.0, label="$h_t$", lx=73.6, ly=30.8)
    arr(56.5, 23.0, 63.0, 23.6, label="$\\bar{z}_c$", lx=59.8, ly=21.0)
    arr(79.5, 25.5, 83.5, 28.0, label="$\\mu$", lx=81.6, ly=25.6)

    # training-time edges: dashed, and drawn from the quantity each loss uses
    arr(79.5, 36.5, 77.2, 11.5, color=EC["phy"], ls="--", lw=0.8,
        label="$\\hat{y}$", lx=80.6, ly=24.5, lha="left")
    arr(56.5, 34.5, 51.5, 11.5, color=EC["phy"], ls="--", lw=0.8,
        label="", lx=0, ly=0)
    arr(56.5, 23.5, 51.5, 11.5, color=EC["phy"], ls="--", lw=0.8,
        label="", lx=0, ly=0)

    save(fig, "fig1_architecture")


# --------------------------------------------------------------------------- #
def fig2_benchmark():
    """Accuracy: RMSE (linear, comparable) + NASA score (log, wide dynamic range)."""
    phys = _load(RESULTS_DIR / "physdec" / "multiseed_stats.json")
    base = _load(RESULTS_DIR / "physdec" / "baseline_multiseed_stats.json")
    subsets = ["FD001", "FD002", "FD003", "FD004", "DS02"]

    p_rmse = np.array([phys[s]["rmse_mean"] for s in subsets])
    p_rmse_s = np.array([phys[s]["rmse_std"] for s in subsets])
    b_rmse = np.array([base[s]["baseline_rmse_mean"] for s in subsets])
    b_rmse_s = np.array([base[s]["baseline_rmse_std"] for s in subsets])
    p_val = [base[s]["paired_p"] for s in subsets]

    p_score = np.array([phys[s]["score_mean"] for s in subsets])
    p_score_s = np.array([phys[s]["score_std"] for s in subsets])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FS.FIG_WIDTH_IN, 3.5))
    x = np.arange(len(subsets)); w = 0.36

    # ---- (a) RMSE ----
    ax1.bar(x - w/2, b_rmse, w, yerr=b_rmse_s, capsize=2.5, color=C_BASELINE,
            label="strongest baseline", error_kw=dict(lw=0.8, ecolor="#333"))
    ax1.bar(x + w/2, p_rmse, w, yerr=p_rmse_s, capsize=2.5, color=C_PHYSDEC,
            label=LABEL_PHYSDEC, error_kw=dict(lw=0.8, ecolor="#333"))
    for i, pv in enumerate(p_val):
        txt = "n.s." if pv >= 0.05 else f"$p$={pv:.3f}"
        y = max(b_rmse[i] + b_rmse_s[i], p_rmse[i] + p_rmse_s[i])
        ax1.text(i, y + 1.0, txt, ha="center", fontsize=FS.PT_ANNOT, color="#555")
    ax1.set_xticks(x); ax1.set_xticklabels(subsets)
    ax1.set_ylabel("RMSE (cycles)")
    ax1.set_title("(a) RMSE", fontsize=9)
    ax1.legend(loc=LEGEND_LOC_LEFT, fontsize=FS.PT_ANNOT)
    ax1.set_ylim(0, float((b_rmse + b_rmse_s).max()) * 1.24)

    # ---- (b) NASA score: log axis, point + error markers ----
    ax2.errorbar(x, p_score, yerr=p_score_s, fmt="o", ms=4.5, lw=1.0,
                 color=C_PHYSDEC, ecolor="#333", elinewidth=0.8, capsize=2.5,
                 label=LABEL_PHYSDEC)
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(subsets)
    ax2.set_ylabel("NASA score (log scale)")
    ax2.set_title("(b) NASA score", fontsize=9)
    lo = float((p_score - p_score_s).min())
    hi = float((p_score + p_score_s).max())
    ax2.set_ylim(max(lo * 0.5, 100), hi * 2.0)
    # Place the legend after the limits are fixed, and route the value labels through
    # annotate_bars rather than ax.annotate. The hand-placed version put DS02's 83,838 --
    # the highest point, at the right-hand edge -- straight onto an upper-right legend,
    # because annotate() knows nothing about what else is on the axes.
    ax2.legend(loc=LEGEND_LOC_LEFT, fontsize=FS.PT_ANNOT)
    annotate_bars(ax2, x, list(p_score), errs=list(p_score_s),
                  labels=[f"{v:,.0f}" for v in p_score], fontsize=FS.PT_ANNOT,
                  dy_frac=0.03)

    fig.tight_layout()
    save(fig, "fig2_benchmark")


# --------------------------------------------------------------------------- #
def fig3_conformal_width():
    """Conformal interval width on the four C-MAPSS subsets."""
    cs = _load(RESULTS_DIR / "physdec" / "conformal_stats.json")
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    pw = [cs[s]["physdec"]["width_mean"] for s in subsets]
    ps = [cs[s]["physdec"]["width_std"] for s in subsets]
    bw = [cs[s]["baseline_lstmcond"]["width"] for s in subsets]

    x = np.arange(len(subsets)); w = 0.36
    fig, ax = plt.subplots(figsize=(FS.FIG_WIDTH_IN, 3.3))
    ax.bar(x - w/2, pw, w, yerr=ps, capsize=2.5, color=C_PHYSDEC,
           label="CDN-RUL (5 seeds)", error_kw=dict(lw=0.8, ecolor="#333"))
    ax.bar(x + w/2, bw, w, color=C_BASELINE, label="LSTM-Cond")
    ax.set_xticks(x); ax.set_xticklabels(subsets)
    ax.set_ylabel("normalized conformal width")
    ax.set_ylim(0, float(max((np.array(pw) + np.array(ps)).max(), max(bw))) * 1.30)
    ax.legend(loc=LEGEND_LOC, fontsize=FS.PT_ANNOT)
    # errs is passed so each label clears its own error bar. Without it the labels for
    # FD001 (0.366), FD003 (0.300) and FD004 (0.283) were drawn through by the error-bar
    # line, because the placer only ever tested against the legend.
    annotate_bars(ax, np.concatenate([x - w/2, x + w/2]),
                  np.array(pw + bw), errs=np.array(ps + [0.0] * len(bw)),
                  fontsize=FS.PT_ANNOT)
    fig.tight_layout()
    save(fig, "fig3_conformal_width")


# --------------------------------------------------------------------------- #
def fig4_reliability():
    """Reliability diagrams on the four C-MAPSS subsets."""
    d = _load(RESULTS_DIR / "physdec" / "uncertainty_stats.json")
    noms = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    fig, axes = plt.subplots(2, 2, figsize=(FS.FIG_WIDTH_IN, 4.9))

    # y-floor derived from the data so no uncertainty band is clipped
    floors = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        p = d[ds]["physdec"]
        floors.append(float((np.array(p["picp_mean"])
                             - np.array(p["picp_std"])).min()))
    y_lo = max(min(floors) - 0.03, 0.0)

    for ax, ds in zip(axes.flat, ["FD001", "FD002", "FD003", "FD004"]):
        p = d[ds]["physdec"]; b = d[ds]["baseline_lstmcond"]
        pm = np.array(p["picp_mean"]); pstd = np.array(p["picp_std"])
        ax.fill_between(noms, pm - pstd, pm + pstd, color=C_PHYSDEC,
                        alpha=0.14, lw=0,
                        label="CDN-RUL $\\pm$1 s.d. (5 seeds)")
        ax.plot(noms, pm, "o-", color=C_PHYSDEC, lw=1.5, ms=4,
                label=LABEL_PHYSDEC)
        ax.plot(noms, b["picp"], "s--", color=C_BASELINE, lw=1.3, ms=4,
                label="LSTM-Cond")
        ax.plot([0.4, 1.0], [0.4, 1.0], ":", color=C_TARGET, lw=0.9,
                label="perfect calibration")
        ax.set_title(ds, fontsize=9)
        ax.set_xlim(0.45, 1.0); ax.set_ylim(y_lo, 1.0)
        ax.set_xticks([0.5, 0.7, 0.9])

    for ax in axes[1, :]:
        ax.set_xlabel("nominal coverage")
    for ax in axes[:, 0]:
        ax.set_ylabel("empirical coverage")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # The figure was 5.6 in tall with a 4.5% strip reserved for the legend, which left a
    # visible white band between the bottom row's tick labels and the legend. Shortening
    # the canvas and reserving a proportionally smaller strip closes the gap without
    # shrinking the panels enough to matter: the axes keep their aspect because the
    # height comes down, not the width.
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=FS.PT_ANNOT,
               bbox_to_anchor=(0.5, -0.015), frameon=False)
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, "fig4_reliability")


if __name__ == "__main__":
    fig1_architecture()
    fig2_benchmark()
    fig3_conformal_width()
    fig4_reliability()
    print("\nAll vector figures regenerated with unified style.")
