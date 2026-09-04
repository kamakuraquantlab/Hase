"""Best bid and ask."""

from pathlib import Path

from ..store import load_order_book
from . import style


def plot_order_book(root: Path, market: str, file_date: str, output: str,
                    resample: str | None = "1s") -> Path:
    """Best bid and ask over the day, with the spread beneath.

    Resampled by default. A day is around 230,000 snapshots, which is more
    points than a chart has pixels; drawing them all is slow and no more
    informative. Passing `--no-resample` keeps every snapshot for a short
    window where the detail is the point.
    """
    df = load_order_book(root, market, file_date, depth=1)
    if df.empty:
        raise ValueError(f"No order book for {market} on {file_date}")

    df = df.set_index("time")
    if resample:
        # last() rather than mean(): a mean of a bid and its neighbours is a
        # price that never existed, and the spread computed from two such means
        # can even be negative.
        df = df[["bid0_price", "ask0_price"]].resample(resample).last().dropna()

    fig, (book_ax, spread_ax) = style.prepare(rows=2, height_ratios=[3, 1])

    book_ax.plot(df.index, df["ask0_price"], color=style.ASK, linewidth=0.8, label="best ask")
    book_ax.plot(df.index, df["bid0_price"], color=style.BID, linewidth=0.8, label="best bid")
    book_ax.fill_between(df.index, df["bid0_price"], df["ask0_price"],
                         color=style.MUTED, alpha=0.15, linewidth=0)
    style.finish(book_ax, title=f"{market}   {file_date}  (JST)", ylabel="price", legend=True)

    spread = df["ask0_price"] - df["bid0_price"]
    spread_ax.fill_between(df.index, spread, color=style.MUTED, alpha=0.55, linewidth=0)

    # A handful of momentary wide spreads would otherwise set the scale and
    # flatten the rest of the day into a line. Clipping the view to the 99.5th
    # percentile keeps the normal range readable; the count of points above the
    # cut is labelled rather than hidden, since a spread spike is real and a
    # reader should know some were left off the top.
    cap = float(spread.quantile(0.995))
    above = int((spread > cap).sum())
    if above and cap > 0:
        spread_ax.set_ylim(0, cap * 1.15)
        spread_ax.text(
            0.995, 0.88, f"{above:,} point(s) above {cap:,.0f}",
            transform=spread_ax.transAxes, ha="right", fontsize=8, color=style.MUTED,
        )
    style.finish(spread_ax, ylabel="spread")
    spread_ax.set_xlabel(
        f"{len(df):,} points" + (f", resampled {resample}" if resample else ", every snapshot"),
        color=style.MUTED, fontsize=9,
    )
    return style.save(fig, output)
