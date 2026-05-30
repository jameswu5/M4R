import matplotlib as mpl
from cycler import cycler

figsize1 = (8, 5)
figsize2 = (12, 5)
figsize3 = (18, 5)

cmap = "RdBu_r"

CRIMSON = "#67001f"
CORAL = "#d6604d"
GRAPHITE = "#555555"
BLUE = "#4393c3"
NAVY = "#053061"

# fixed line palette, in cycle order
LINE_COLOURS = [BLUE, CORAL, GRAPHITE, NAVY, CRIMSON]


def use_line_palette(colors=LINE_COLOURS):
    """Set the default line-color cycle globally to the fixed palette."""
    mpl.rcParams["axes.prop_cycle"] = cycler(color=colors)


# apply the palette as the default for all plots that import this module
use_line_palette()
