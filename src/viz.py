"""
Shared chart styling for all matplotlib/seaborn figures in this project.

Centralizes the palette and chart chrome (one validated categorical order for
PCOS=No/Yes, one sequential ramp, one diverging pair) so every figure across
every notebook looks like it belongs to the same report, instead of each plot
picking its own default colors.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from src.config import FIGURES_DIR

# Categorical: fixed order, never cycled. Slot 1 = PCOS-negative, slot 2 = PCOS-positive
# everywhere in this project, so the same color always means the same class across figures.
COLOR_NEGATIVE = "#2a78d6"   # categorical slot 1 (blue)
COLOR_POSITIVE = "#eb6834"   # categorical slot 2 (orange)
CLASS_COLORS = {0: COLOR_NEGATIVE, 1: COLOR_POSITIVE}
CLASS_LABELS = {0: "No PCOS", 1: "PCOS"}

# Sequential: one hue (blue), light -> dark, for magnitude-only encodings (e.g. missingness).
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# Diverging: blue <-> red with a neutral gray midpoint, for signed quantities (correlation).
DIVERGING_CMAP = "RdBu_r"  # blue (negative) <-> gray (0) <-> red (positive), matches the pair

# Chart chrome
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_chart_style(ax) -> None:
    """Apply consistent, recessive chrome to a matplotlib Axes: light gridlines,
    muted ticks, no top/right spines, so the data (not the frame) carries attention.
    """
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.title.set_color(INK_PRIMARY)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)


def boxplot_by_class(ax, df, col: str, target_col: str, showfliers: bool = True) -> None:
    """Draw a two-box (No PCOS / PCOS) boxplot of `col` on `ax`, styled with the
    project's fixed class colors. Used by every "<numeric variable> by class"
    EDA figure so the styling never drifts between charts.
    """
    data = [df.loc[df[target_col] == c, col].dropna() for c in [0, 1]]
    bp = ax.boxplot(
        data, patch_artist=True, widths=0.5,
        tick_labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
        medianprops=dict(color=INK_PRIMARY, linewidth=1.5),
        showfliers=showfliers,
    )
    for patch, c in zip(bp["boxes"], [0, 1]):
        patch.set_facecolor(CLASS_COLORS[c])
        patch.set_alpha(0.75)
        patch.set_edgecolor(CLASS_COLORS[c])
    for element in ("whiskers", "caps"):
        for line in bp[element]:
            line.set_color(INK_SECONDARY)


def grouped_bar_by_class(ax, categories, values_negative, values_positive, value_fmt="{:.0f}%") -> None:
    """Draw a two-series (No PCOS / PCOS) grouped bar chart across `categories`,
    with direct value labels above each bar. Used for proportion/rate comparisons.
    """
    import numpy as np

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width / 2, values_negative, width, label=CLASS_LABELS[0], color=COLOR_NEGATIVE)
    ax.bar(x + width / 2, values_positive, width, label=CLASS_LABELS[1], color=COLOR_POSITIVE)
    y_max = max(max(values_negative), max(values_positive))
    for i, (v0, v1) in enumerate(zip(values_negative, values_positive)):
        ax.text(i - width / 2, v0 + y_max * 0.02, value_fmt.format(v0), ha="center", fontsize=8.5, color=INK_PRIMARY)
        ax.text(i + width / 2, v1 + y_max * 0.02, value_fmt.format(v1), ha="center", fontsize=8.5, color=INK_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)


def save_fig(fig, name: str, dpi: int = 160, tight_layout_rect=None) -> Path:
    """Save a figure to reports/figures/<name>.png with consistent DPI and tight layout.

    Pass `tight_layout_rect=(left, bottom, right, top)` (all in 0-1 figure
    fraction) when the figure has a `suptitle` - matplotlib's default
    tight_layout does not reserve space for it, so it gets clipped on save.
    """
    fig.set_facecolor(SURFACE)
    out_path = FIGURES_DIR / f"{name}.png"
    fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    return out_path
