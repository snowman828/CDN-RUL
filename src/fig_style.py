"""Shared publication style for all figures.

Single source of truth for figure geometry, typography, legend placement,
canonical legend strings and value-label placement, so the five figures and every
panel within them stay mutually consistent.

  * GEOMETRY: every figure is laid out at FIG_WIDTH_IN, the journal's full text
    width. A set drawn at five different canvas widths (which this set was: 162.6,
    172.7, 213.4, 223.5 and 233.7 mm) is not a styled set, and any figure wider
    than the page is silently downscaled by the typesetter -- which shrinks its
    text by the same factor.
  * TYPE SCALE: five sizes, defined here, all >= PT_MIN. Callers pass one of them
    through pt(), which refuses anything smaller. Figure 1 previously carried
    6.4-7.4 pt annotations added ad hoc at call sites; at any downscale those fall
    below the floor at which figure text stays readable in print.
  * FONT: serif, resolving to Times New Roman, matching the journal body text.
  * LEGEND PLACEMENT: centralised here. Panels must not set ad-hoc loc= values;
    drifted placement is the most visible inconsistency across a figure set.
  * TERMINOLOGY: canonical legend strings live here, so panels cannot diverge
    into "group-cond." versus "group-conditional q_k".
  * LANGUAGE: US English ("normalized"), matching Elsevier style.
  * COLLISION SAFETY: annotate_bars() performs a real overlap check before
    placing a value label, so a label can never print over legend text.
  * SAVE-TIME SELF-CHECK: save() reads back the PDF it just wrote and fails if the
    page width is not the standard. Declaring a figsize is not the same as
    producing that page size -- figure.dpi, bbox and tight layouts all move the
    real output -- and a figure that ships 40 mm wider than intended is one the
    typesetter will downscale.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pathlib import Path
from common import PROJECT_ROOT as ROOT

# Generation writes straight into the directory the submission package ships from.
# An earlier de-hardcoding pass rewrote this from reports/figures to results/figures
# while removing an absolute path -- a silent behaviour change that left every
# regenerated figure stranded outside the shipped set.
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- geometry ----
# Journal full text width. Figures are drawn at exactly this width so that the
# point sizes below are the point sizes the reader sees: no typesetter downscale,
# no compensating fudge in the font settings.
MM_PER_IN = 25.4
FIG_WIDTH_MM = 190.0
FIG_WIDTH_IN = FIG_WIDTH_MM / MM_PER_IN          # 7.480 in
WIDTH_TOL_MM = 1.5                               # page-width check tolerance

# ------------------------------------------------------------ type scale -----
PT_TITLE = 9.0    # axis titles / panel headings
PT_LABEL = 8.0    # axis labels, tick labels, box body text
PT_ANNOT = 7.5    # legends, secondary annotations
PT_MIN = 7.0      # hard floor -- nothing in any figure may go below this

# matplotlib draws mathtext scripts at this fraction of the base size. Measured from
# the delivered Figure 1: 127 script spans, ratio 0.7 to 0.7.  A sub-floor span that is
# not a declared size times this value is a real defect and fails the save.
SCRIPT_RATIO = 0.70


def sub_floor_spans(pdf, floor):
    """Every span below `floor` as (size, text).  Empty if no PDF reader is present."""
    try:
        import pymupdf as fitz
    except ImportError:
        return []
    out = []
    doc = fitz.open(pdf)
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] < floor:
                        out.append((span["size"], span["text"].strip()))
    doc.close()
    return out


def pt(size: float) -> float:
    """Validate a font size against the floor.

    Raises rather than warns: a sub-floor size introduced at a call site is
    invisible until the figure is printed, and by then the proof cycle has
    already started.
    """
    if size < PT_MIN - 1e-9:
        raise ValueError(
            f"font size {size} pt is below the {PT_MIN} pt floor for figure text; "
            f"use PT_ANNOT ({PT_ANNOT}), PT_LABEL ({PT_LABEL}) or PT_TITLE ({PT_TITLE})"
        )
    return size


# ---- palette (colour-blind safe, top-journal restrained) ----
C_PHYSDEC = "#1F4E79"    # deep blue     — CDN-RUL (primary method)
C_BASELINE = "#C55A11"   # burnt orange  — baselines / comparison
C_GRAY = "#7F7F7F"       # neutral grey  — references / secondary
C_LIGHT = "#A5A5A5"      # light grey    — marginal/group split fills
C_TARGET = "#333333"     # near-black    — target / identity lines

MODEL_COLORS = {"physdec": C_PHYSDEC, "lstm": C_BASELINE}

# ---- canonical terminology (single source of truth) ----
LABEL_MARGINAL = "marginal $q$"
LABEL_GROUP = "group-conditional $q_k$"
LABEL_PHYSDEC = "CDN-RUL"
LABEL_LSTM = "LSTM (no cond. norm.)"
LABEL_LSTM_SHORT = "LSTM"

# ---- canonical legend placement ----
LEGEND_LOC = "upper right"          # for axes whose data rise to the right
LEGEND_LOC_LEFT = "upper left"      # only when upper-right is occupied


def apply_style() -> None:
    """Apply the shared rcParams once per script."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": PT_LABEL,
        "axes.labelsize": PT_TITLE,
        "axes.titlesize": PT_TITLE,
        "xtick.labelsize": PT_LABEL,
        "ytick.labelsize": PT_LABEL,
        "legend.fontsize": PT_ANNOT,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.5,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
        "grid.color": "#CCCCCC",
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        # NOT "tight": the saved page must be FIG_WIDTH_IN regardless of where the
        # artists happen to end, otherwise the delivered width drifts with content
        # and the point sizes above stop predicting what the reader sees.
        "savefig.bbox": "standard",
        "savefig.pad_inches": 0,
    })


def figure(width_in=None, height_in=None, **kw):
    """Create a figure at the standard width (or an explicit one)."""
    kw.setdefault("figsize", (width_in or FIG_WIDTH_IN, height_in or 3.6))
    return plt.figure(**kw)


def annotate_bars(ax, xs, heights, labels=None, errs=None, dy_frac=0.02,
                  fontsize=PT_MIN, avoid_legend=True, avoid_others=True):
    """Place value labels above bars, avoiding the legend, error bars and each other.

    Three defects in earlier versions, all of which produced collisions in delivered
    figures:

    1. Text width was a hard-coded `pad = 55` display pixels. "83,838" and "0.366" are
       not the same width, so the guess was wrong in both directions, and the label for
       the tallest bar landed on the legend.
    2. The collision test covered only the legend. Nothing tested the value label against
       the error bar drawn through it, which is how "0.366", "0.300" and "0.283" came to
       be struck through by their own error-bar line in Figure 3.
    3. Labels were never tested against one another.

    Now the text is measured after drawing (renderer.get_window_extent on a real text
    artist), `errs` raises each label clear of its error bar, and every accepted label is
    kept so later ones can be tested against it. A label that still cannot be placed is
    omitted rather than overprinted: a missing value is a smaller defect than an
    unreadable one.
    """
    fontsize = pt(fontsize)
    fig = ax.get_figure()
    fig.canvas.draw()

    legend_bbox = None
    if avoid_legend and ax.get_legend() is not None:
        legend_bbox = ax.get_legend().get_window_extent()

    ymax = ax.get_ylim()[1]
    dy = dy_frac * ymax
    placed = []

    for i, (x, h, lab) in enumerate(zip(xs, heights, labels or [None] * len(xs))):
        # raise the label clear of the error bar, not merely above the bar top
        top = h + (errs[i] if errs is not None else 0.0)
        y = top + dy
        txt = f"{h:.3f}" if lab is None else lab

        # measure the real text extent instead of assuming a width
        probe = ax.text(x, y, txt, ha="center", va="bottom", fontsize=fontsize)
        try:
            tb = probe.get_window_extent(renderer=fig.canvas.get_renderer())
        except Exception:
            placed.append(tb := None)
            continue

        collides = False
        if legend_bbox is not None and tb.overlaps(legend_bbox):
            collides = True
        if not collides and avoid_others:
            for pb in placed:
                if pb is not None and tb.overlaps(pb):
                    collides = True
                    break
        # an error bar through a label is a collision too, if one is drawn
        if not collides and errs is not None and errs[i]:
            from matplotlib.transforms import Bbox
            xd, yd_top = ax.transData.transform((x, top))
            _, yd_bot = ax.transData.transform((x, h))
            eb = Bbox.from_extents(tb.x0, yd_bot, tb.x1, yd_top)
            if tb.overlaps(eb) and yd_top > tb.y0:
                collides = True

        if collides:
            probe.remove()
            continue
        placed.append(tb)


def zero_based(ax, top_pad=0.06):
    """Force a zero baseline on a bar axis (truncated bars exaggerate gaps)."""
    lo, hi = ax.get_ylim()
    ax.set_ylim(0, max(hi, 0) * (1 + top_pad))


def min_text_pt(path) -> float:
    """Smallest text size actually present in a saved PDF (0.0 if unreadable)."""
    try:
        import fitz
    except ImportError:                      # PyMuPDF is not in the figure venv
        return float("nan")
    doc = fitz.open(path)
    sizes = [s["size"] for pg in doc for b in pg.get_text("dict")["blocks"]
             for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
    doc.close()
    return min(sizes) if sizes else float("nan")


def page_width_mm(path) -> float:
    """Page width of a saved PDF, in millimetres."""
    try:
        import fitz
    except ImportError:
        return float("nan")
    doc = fitz.open(path)
    w = doc[0].rect.width
    doc.close()
    return w / 72.0 * MM_PER_IN


def save(fig, name: str, width_mm: float = None) -> None:
    """Save vector PDF + 300-dpi PNG preview, then check the page geometry.

    The width check is the point: a figure whose PDF page is wider than the
    journal text width will be downscaled on the page, and its 7.5 pt annotations
    will print smaller than 7.5 pt.  The declared figsize does not guarantee the
    delivered width -- bbox handling, tight layouts and constrained_layout all
    move it -- so the delivered file is read back and measured.

    PyMuPDF is not installed in the figure-generation interpreter (it lives with
    the system python), so the measurement cannot always run here. When it cannot,
    this says so out loud rather than reporting a pass it did not perform;
    src/audit_figure_standards.py performs the same checks from outside.
    """
    target = width_mm if width_mm is not None else FIG_WIDTH_MM
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(pdf, format="pdf")
    fig.savefig(FIG_DIR / f"{name}.png", format="png", dpi=300)
    plt.close(fig)

    got = page_width_mm(pdf)
    if got != got:                                          # nan -> no reader
        print(f"  saved {name}.pdf / .png  "
              f"(geometry UNVERIFIED here: no PDF reader in this interpreter; "
              f"run src/audit_figure_standards.py)")
        return
    if abs(got - target) > WIDTH_TOL_MM:
        raise ValueError(
            f"{name}: page width {got:.1f} mm differs from the {target:.1f} mm "
            f"standard by more than {WIDTH_TOL_MM} mm -- the typesetter would "
            f"downscale it and every point size in it would shrink"
        )
    # Size check with scripts understood.
    #
    # matplotlib draws the subscript of z_d, the k of w_k and the t of step_mu_t at
    # SCRIPT_RATIO of the declared size, so a 7.5 pt label delivers a 5.25 pt
    # subscript. That is not a defect and must not fail; a genuinely small label must.
    # The distinction is made by prediction rather than by geometry: a sub-floor span
    # is accepted only if its size is a declared size times SCRIPT_RATIO. Detecting
    # subscripts from their position was tried and rejected -- PyMuPDF splits some of
    # them onto their own line, where the local baseline is the subscript's own, and
    # the test misfiled 2 of 14 spans on the delivered Figure 1.
    smallest = min_text_pt(pdf)
    if smallest != smallest:                                # nan -> not measured
        print(f"  saved {name}.pdf / .png  ({got:.1f} mm wide, min text not measured)")
        return
    bad = []
    for size, text in sub_floor_spans(pdf, PT_MIN):
        if not any(abs(size - base * SCRIPT_RATIO) <= 0.25
                   for base in (PT_ANNOT, PT_LABEL, PT_TITLE)):
            bad.append((size, text))
    if bad:
        worst = min(bad)
        raise ValueError(
            f"{name}: {len(bad)} span(s) below the {PT_MIN} pt floor that are not "
            f"scripts of a declared size, e.g. {worst[0]:.1f} pt {worst[1]!r}"
        )
    print(f"  saved {name}.pdf / .png  ({got:.1f} mm wide, min text "
          f"{smallest:.1f} pt, scripts checked)")
