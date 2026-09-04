"""Trade price and volume."""

from pathlib import Path

from ..store import load_trades
from . import style

SIDE_BUY = 0


def plot_trades(root: Path, market: str, file_date: str, output: str,
                volume_bucket: str = "5min") -> Path:
    """Price over the day, with buy and sell volume beneath it.

    Volume is bucketed rather than drawn per trade: a day is tens of thousands
    of trades and a bar per trade is an unreadable smear. Buys and sells are
    separated because their balance is the point of looking, and a single
    combined bar hides it.
    """
    df = load_trades(root, market, file_date)
    if df.empty:
        raise ValueError(f"No trades for {market} on {file_date}")

    fig, (price_ax, vol_ax) = style.prepare(rows=2, height_ratios=[3, 1])

    price_ax.plot(df["time"], df["price"], color=style.INK, linewidth=0.7)
    style.finish(price_ax, title=f"{market}   {file_date}  (JST)", ylabel="price")

    buys = df[df["side"] == SIDE_BUY].set_index("time")["size"].resample(volume_bucket).sum()
    sells = df[df["side"] != SIDE_BUY].set_index("time")["size"].resample(volume_bucket).sum()
    width = (buys.index[1] - buys.index[0]).total_seconds() / 86_400 if len(buys) > 1 else 0.002

    vol_ax.bar(buys.index, buys.values, width=width, color=style.BUY, label="buy", align="edge")
    vol_ax.bar(sells.index, -sells.values, width=width, color=style.SELL, label="sell", align="edge")
    vol_ax.axhline(0, color=style.GRID, linewidth=0.8)
    style.finish(vol_ax, ylabel=f"volume / {volume_bucket}", legend=True)

    n = len(df)
    vol_ax.set_xlabel(f"{n:,} trades", color=style.MUTED, fontsize=9)
    return style.save(fig, output)
