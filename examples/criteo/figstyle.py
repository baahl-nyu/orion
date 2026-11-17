"""One look for every figure in examples/criteo/figures, matched to the paper's
training figure: matplotlib's default sans font at paper size, dark lines with
pastel black-edged markers, framed legends, a light grid, dashed dark-red
reference lines with bold red labels. Figures are drawn at the two-column
paper's column width so the fonts print at the size they are set here.

    from figstyle import apply, COL_W, FILL, LINE, RED, save
    apply()
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    ...
    save(fig, "name")      # figures/name.png at 1200 dpi and figures/name.pdf
"""
import os
from os.path import abspath, dirname, join

import matplotlib

matplotlib.use("Agg")

FIG_DIR = join(dirname(abspath(__file__)), "figures")
COL_W = 3.33  # \columnwidth of the two-column layout, inches
DPI = 1200  # a column-width figure is 4000 px wide

# series palette: pastel fill for the marker, a darker shade of the same hue for the line
# hues: olive green, blue, orange, purple, teal, brown; none near the red of the reference lines
FILL = ["#c9d18f", "#aac5ea", "#f4b98a", "#bcb0e6", "#9fd6cf", "#d4bc9a"]
LINE = ["#6f7a2f", "#3f6fb3", "#d0782e", "#6f5cb8", "#2a8f86", "#8a6a3c"]
RED = "#b22222"  # reference lines and their labels
GREY = "#555555"

RC = {
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7, "legend.title_fontsize": 7.5, "legend.frameon": True, "legend.framealpha": 1,
    "legend.edgecolor": "0.3", "legend.fancybox": True, "legend.handlelength": 1.8,
    "legend.borderpad": 0.4, "legend.labelspacing": 0.3, "legend.columnspacing": 0.8, "legend.handletextpad": 0.4,
    "axes.grid": True, "grid.color": "0.85", "grid.linewidth": 0.6, "axes.linewidth": 0.8,
    "axes.edgecolor": "0.2", "xtick.color": "0.2", "ytick.color": "0.2",
    "lines.linewidth": 1.4, "lines.markersize": 5.5, "lines.markeredgewidth": 0.7, "lines.markeredgecolor": "black",
    "savefig.dpi": DPI, "figure.dpi": 100,
}


def apply():
    matplotlib.rcParams.update(RC)


def series(ax, xs, ys, i, label=None, marker="o", ls="-", **kw):
    """A line in LINE[i] with pastel FILL[i] markers edged in black; the marker carries the legend entry."""
    ax.plot(xs, ys, ls, color=LINE[i], zorder=2, **{k: v for k, v in kw.items() if k in ("lw", "alpha")})
    ax.plot(xs, ys, marker, mfc=FILL[i], mec="black", ls="none", label=label, zorder=3,
            **{k: v for k, v in kw.items() if k in ("ms", "mew")})


def ref_line(ax, y, text, left=True, dy=0.0, fontsize=6.5, lw=1.3):
    """Dashed red horizontal line with a bold red label inside the axes, at the left or right edge."""
    ax.axhline(y, color=RED, lw=lw, ls="--", zorder=1)
    ax.text(0.015 if left else 0.985, y + dy, text, fontsize=fontsize, color=RED, fontweight="bold",
            va="bottom", ha="left" if left else "right", transform=ax.get_yaxis_transform(), zorder=4)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(join(FIG_DIR, name + ".png"), dpi=DPI)
    fig.savefig(join(FIG_DIR, name + ".pdf"))
    return join(FIG_DIR, name + ".png")
