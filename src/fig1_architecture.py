r"""Figure 1 — CDN-RUL architecture schematic.

Drawn against physdec_model.py, not against the prose. Three separate attempts got
this wrong before it was checked against the source:

  1. the decoder was labelled fusion(h_T (+ ) z_c), but the model does
     cat(LSTM(z_d), mean_t(z_c)) -> fusion; the LSTM consumes z_d and z_c enters
     at fusion after pooling over time.
  2. z_c = enc_c(x . g) was drawn with no edge carrying x into enc_c.
  3. the monotonicity term was fed from the output y_hat. It is not: the loss is
     monotonicity_loss(step_mu), where step_mu is the per-timestep head output and
     mu = step_mu[:, -1] is only its final slice. The tap has to leave fusion/head.

Standing rules for this figure, applied here:
  * every edge carries the name of the tensor it transports;
  * training-time edges are dashed, labelled, and drop vertically to a loss box
    placed directly beneath their source, so no auxiliary line crosses the
    architecture;
  * routing is explicit polylines through free corridors rather than diagonals
    across boxes.

Written with raw string literals throughout: two LaTeX escapes in the previous
version lost their leading backslash in transit ("\rightarrow" printed as
"ightarrow"), and raw strings make that class of damage impossible here.
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import fig_style as FS

C_IN, C_CDFE = "#F2E7D5", "#D6E4F0"
C_LSTM, C_PHY, C_OUT = "#E4DFEC", "#DFECDF", "#F2E7D5"
EC = {"in": "#B07A4A", "cdfe": "#1F4E79", "lstm": "#6C4F8B",
      "phy": "#3F7D3F", "out": "#B07A4A", "arr": "#4A4A4A"}


def draw() -> None:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(FS.FIG_WIDTH_IN, 2.95))
    # The visible y-range starts above zero. The loss row used to sit at y=3.2 with the
    # architecture ending at 17.9, which left a 5.3-unit empty corridor and made the lower
    # half read as detached. Raising the loss row and trimming the floor closes the gap.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(3.0, 42)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec, fs=FS.PT_LABEL, lw=1.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                    fc=fc, ec=ec, lw=lw, mutation_aspect=0.55))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=FS.pt(fs), color="#1a1a1a", linespacing=1.55)

    def edge(points, label=None, color=EC["arr"], lw=0.9, ls="-",
             lx=None, ly=None, lha="center", lva="center", size=7):
        """Arrow through a polyline; only the last segment is headed."""
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls,
                    solid_capstyle="butt", zorder=1)
        a, b = points[-2], points[-1]
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=size,
                                     lw=lw, color=color, linestyle=ls,
                                     shrinkA=0, shrinkB=0, zorder=2))
        if label:
            ax.text(lx, ly, label, ha=lha, va=lva,
                    fontsize=FS.pt(FS.PT_ANNOT), color=color, zorder=3,
                    bbox=dict(fc="white", ec="none", pad=1.0))

    # =========================================================== inputs =====
    box(0.5, 31.0, 16.0, 8.6, "Sensors $X$\n$(B,T,S)$", C_IN, EC["in"])
    box(0.5, 20.5, 16.0, 8.6, "Op-settings $C$\n$(B,T,C)$", C_IN, EC["in"])

    # ============================================================= CDFE =====
    ax.add_patch(FancyBboxPatch((19.0, 17.9), 41.5, 24.6,
                                boxstyle="round,pad=0.6", fc="none",
                                ec=EC["cdfe"], lw=1.2, ls="--",
                                mutation_aspect=0.55))
    ax.text(39.75, 40.4, "CDFE — condition-decoupled encoder", ha="center",
            va="center", fontsize=FS.pt(FS.PT_TITLE), color=EC["cdfe"])

    box(21.0, 31.0, 37.5, 7.2,
        r"GMM-soft per-regime normalization   $z_d=\sum_k w_k(t)(x-\bar{x}_k)/\sigma_k$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)
    box(22.0, 20.5, 12.0, 6.6, r"gate  $g(t)=\sigma(W_g c)$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)
    ax.add_patch(FancyBboxPatch((36.0, 22.3), 4.0, 3.0,
                                boxstyle="round,pad=0.25", fc="white",
                                ec=EC["cdfe"], lw=0.9, mutation_aspect=0.55))
    ax.text(38.0, 23.8, r"$\times$", ha="center", va="center",
            fontsize=FS.pt(FS.PT_LABEL))
    box(42.0, 20.5, 16.5, 6.6,
        r"$z_c=\mathrm{enc}_c(x\odot g)$   $(B,T,16)$",
        C_CDFE, EC["cdfe"], fs=FS.PT_ANNOT)

    # ========================================================== decoder =====
    box(65.0, 31.0, 15.0, 7.2, "LSTM (2 layers)", C_LSTM, EC["lstm"], lw=1.2)
    box(65.0, 20.5, 15.0, 6.6,
        "fusion   $[\\,h_t\\,;\\,\\bar{z}_c\\,]$" "\n"
        r"$\mathrm{head}\rightarrow$ step_mu$(B,T)$",
        C_LSTM, EC["lstm"], lw=1.2, fs=FS.PT_ANNOT)

    # =========================================================== output =====
    box(84.0, 20.5, 15.5, 17.7,
        r"$\mu=\mathrm{step\_mu}[:,-1]$" "\n"
        r"$\sigma$ from MC-Dropout" "\n"
        "split conformal\n"
        r"$\hat{y}\pm q\,\sigma$   (90%)",
        C_OUT, EC["out"], fs=FS.PT_ANNOT)

    # =================================================== training-time ======
    # Separator and loss row sit closer to the architecture than before: the taps are now
    # short vertical drops rather than long lines across an empty band.
    ax.plot([0.5, 99.5], [16.9, 16.9], color="#C8C8C8", lw=0.7, ls=(0, (3, 3)))
    ax.text(0.8, 11.2, "training-time\nterms   (not used\nat inference)",
            ha="left", va="center", fontsize=FS.pt(FS.PT_ANNOT),
            color=EC["phy"], style="italic", linespacing=1.6)

    box(21.0, 6.5, 37.5, 9.4,
        r"orthogonality   $\mathcal{L}_{\mathrm{orth}}=\langle z_d,\,z_c"
        + "\\" + "rangle$",
        C_PHY, EC["phy"], fs=FS.PT_ANNOT, lw=0.8)
    box(65.0, 6.5, 34.5, 9.4,
        r"monotonicity   $\mathcal{L}_{\mathrm{phy}}=\sum_t \mathrm{ReLU}(\mathrm{step\_mu}_{t+1}-\mathrm{step\_mu}_t)$",
        C_PHY, EC["phy"], fs=FS.PT_ANNOT, lw=0.8)

    # ============================================================ edges =====
    edge([(16.5, 36.8), (21.0, 36.8)], "$x$", lx=18.7, ly=38.6)
    edge([(16.5, 25.5), (22.0, 25.5)], "$c$", lx=19.2, ly=23.7)

    # x also reaches the gated encoder.
    #
    # The previous route ran horizontally at y=31.0, which is exactly the GMM box's bottom
    # edge (that box starts at y=31.0), so the line merged with the box border and read as
    # if it came from the GMM block. It now leaves the Sensors box on its right edge, drops
    # through the free corridor between the CDFE frame (x=19) and the gate box (x=22), and
    # runs right at y=29.2 -- between the GMM box bottom (31.0) and the gate box top (27.1)
    # -- before dropping down the corridor at x=35 to the multiplier.
    edge([(16.5, 32.4), (20.2, 32.4), (20.2, 29.2), (35.0, 29.2), (35.0, 23.8),
          (36.0, 23.8)], "$x$", lx=27.5, ly=28.3)
    edge([(34.0, 23.8), (36.0, 23.8)], "$g$", lx=35.0, ly=21.9)
    edge([(40.0, 23.8), (42.0, 23.8)], r"$x\odot g$", lx=41.0, ly=25.6)
    edge([(58.5, 23.8), (65.0, 23.8)], r"$\bar{z}_c$", lx=61.7, ly=22.0)
    edge([(58.5, 34.6), (65.0, 34.6)], "$z_d$", lx=61.7, ly=36.4)
    edge([(72.5, 31.0), (72.5, 27.1)], "$h_t$", lx=75.0, ly=29.1, lha="left")
    edge([(80.0, 23.8), (84.0, 23.8)], r"$\mu,\sigma$", lx=82.0, ly=22.0)

    # training-time taps: vertical, labelled, straight onto the loss box below. The loss
    # row now sits at y=6.5-15.9, so each tap is a short drop rather than a line across an
    # empty band.
    #
    # The orthogonality tap is labelled z_c, the tensor it actually carries. It previously
    # read "z_d, z_c" while physically attached only to z_c -- a label asserting a
    # connection the drawing did not show. The term itself pairs both outputs, and that is
    # stated on the loss box, where it belongs; every edge in this figure names what it
    # transports.
    edge([(50.25, 20.5), (50.25, 15.9)], "$z_c$",
         color=EC["phy"], ls="--", lw=0.75, lx=50.25, ly=16.95)
    edge([(72.5, 20.5), (72.5, 15.9)], "step_mu",
         color=EC["phy"], ls="--", lw=0.75, lx=72.5, ly=16.95)

    FS.save(fig, "fig1_architecture")
    plt.close(fig)


if __name__ == "__main__":
    from fig_style import apply_style
    apply_style()
    draw()
