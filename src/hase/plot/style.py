"""Shared plot styling.

One place so every chart in the toolkit reads as the same family, and so the
choices below are made once rather than per plot.

It also owns the timezone. matplotlib converts datetimes to plain floats and
forgets the zone, so a tz-aware series plots correctly but is *labelled* in UTC
unless the formatter is told otherwise. A chart titled JST with a UTC axis is
worse than one with no timezone at all, so every axis formatter here is given
the zone explicitly.
"""

from zoneinfo import ZoneInfo

JST = ZoneInfo("Asia/Tokyo")

BUY = "#2f7d5d"
SELL = "#b4553f"
BID = "#2f5d8a"
ASK = "#a04a63"
INK = "#1b1b1a"
MUTED = "#6b6b66"
GRID = "#e2e2dd"


def prepare(figsize=(13, 7), rows=1, height_ratios=None):
    import matplotlib

    matplotlib.use("Agg")  # no display on a server or in a test
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        rows, 1, figsize=figsize, sharex=True,
        gridspec_kw={"height_ratios": height_ratios or [1] * rows},
    )
    return fig, (axes if rows > 1 else [axes])


def finish(ax, title=None, ylabel=None, legend=False):
    import matplotlib.dates as mdates

    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED, fontsize=9)
    if legend:
        ax.legend(frameon=False, fontsize=9, labelcolor=MUTED)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=JST))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3, tz=JST))


def save(fig, output, dpi=140):
    from pathlib import Path

    path = Path(output).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path
