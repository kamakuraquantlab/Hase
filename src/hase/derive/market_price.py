"""MarketPrice: what an order of a given size would actually pay.

The quoted spread is what the book advertises; this is what it costs. Walking
the book to a size is the difference between the two, and article 1 is about
exactly that gap.

**Size is given in base currency by default** -- 0.002 BTC, 0.1 ETH, 150 XRP,
matching the warehouse. A quote-currency notional is available as a separate
partition key, never the same one, because the two are not interchangeable: a
JPY notional buys a different quantity every day as the price moves, so
`execution_notional=1000000` is not comparable with itself across time the way
`execution_size=0.002` is. Both are offered; only one is the default, and a
cross-venue comparison has to use the base-currency one because a USDT-quoted
book cannot be walked to a JPY target without an FX series.

A size the visible book cannot fill yields NaN rather than a partial price. A
partial fill priced as if complete is the kind of number that looks reasonable
and is wrong.
"""

import numpy as np

from ..store import load_order_book

LEVELS = 20
FILL_TOLERANCE = 0.999
COLUMNS = ["ts", "bid", "ask", "mid", "spread", "spread_bps", "filled", "degenerate"]

# Makalu's `PopulateMarketPriceStep.DEGENERATE_SPREAD_BPS`, and the reason is
# worth carrying across because the artefact is in the bronze this reads.
#
# The collector used to emit partial books after a snapshot resync -- one to
# three levels with stale far-away prices behind them. Walking one of those
# lands hundreds of bps deep and produces a reverting spike of 200 to 1100 bps
# that looks like a real move. The collector was fixed, but historical bronze
# keeps the artefacts, so anyone deriving from an archive still meets them.
#
# The test is on the *walked* spread, not the quoted one: a partial book often
# keeps a tight top of book with almost nothing behind it, so the quote looks
# healthy and only the walk reveals it. 100 bps is four to ten times any real
# walked stress spread on these venues and far below the artefact spreads.
#
# Hase marks these rather than dropping them. How many snapshots a day were
# artefacts is a fact about the archive a buyer is entitled to see, and
# dropping rows silently is how a gap becomes invisible.
DEGENERATE_SPREAD_BPS = 100.0


def _walk_size(price, qty, target):
    """VWAP of taking `target` base units, level by level. NaN if it cannot fill."""
    taken = np.clip(target - (np.cumsum(qty, axis=1) - qty), 0.0, qty)
    filled = taken.sum(axis=1)
    cost = (taken * price).sum(axis=1)
    enough = filled >= target * FILL_TOLERANCE
    return np.where(enough, cost / np.where(filled > 0, filled, np.nan), np.nan), filled


def _walk_notional(price, qty, target):
    """VWAP of spending `target` quote units. NaN if it cannot fill."""
    notional = price * qty
    taken = np.clip(target - (np.cumsum(notional, axis=1) - notional), 0.0, notional)
    spent = taken.sum(axis=1)
    base = (taken / np.where(price > 0, price, np.nan)).sum(axis=1)
    enough = spent >= target * FILL_TOLERANCE
    return np.where(enough, spent / np.where(base > 0, base, np.nan), np.nan), spent


def derive(root, market: str, file_date: str, *,
           execution_size: float | None = None,
           execution_notional: float | None = None,
           degenerate_spread_bps: float = DEGENERATE_SPREAD_BPS):
    """Effective bid and ask for one trade size, per snapshot."""
    import pandas as pd

    if (execution_size is None) == (execution_notional is None):
        raise ValueError("Give exactly one of execution_size or execution_notional")

    raw = load_order_book(root, market, file_date, depth=LEVELS)
    ts = raw["ts"].to_numpy(dtype="float64")

    def side(prefix):
        price = np.column_stack([raw[f"{prefix}{i}_price"].to_numpy(dtype="float64") for i in range(LEVELS)])
        qty = np.column_stack([raw[f"{prefix}{i}_qty"].to_numpy(dtype="float64") for i in range(LEVELS)])
        # A level with no price or no size contributes nothing; exchanges pad
        # the far end of the book with zeros and nulls both.
        bad = ~np.isfinite(price) | ~np.isfinite(qty) | (price <= 0) | (qty <= 0)
        return np.where(bad, 0.0, price), np.where(bad, 0.0, qty)

    bid_price, bid_qty = side("bid")
    ask_price, ask_qty = side("ask")

    walk = _walk_size if execution_size is not None else _walk_notional
    target = execution_size if execution_size is not None else execution_notional
    bid, bid_filled = walk(bid_price, bid_qty, target)
    ask, ask_filled = walk(ask_price, ask_qty, target)

    mid = (bid + ask) / 2
    with np.errstate(invalid="ignore"):
        spread_bps = (ask - bid) / mid * 1e4

    return pd.DataFrame({
        "ts": ts,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": ask - bid,
        "spread_bps": spread_bps,
        # Whether both sides filled, kept so a caller can tell "the book was
        # too thin" apart from "no data", which are different findings.
        "filled": np.isfinite(bid) & np.isfinite(ask),
        # A resync artefact rather than a market state. See the constant above.
        "degenerate": np.isfinite(spread_bps) & (spread_bps > degenerate_spread_bps),
    })


def _format_size(value: float) -> str:
    """A number a human would type, and never scientific notation.

    `%g` switches to exponents at a million, which would have put a directory
    called `execution_notional=1e+06` in the tree -- unreadable in a path and
    a different string from the `1000000` anyone would search for.
    """
    text = f"{value:.10g}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def partition(execution_size: float | None, execution_notional: float | None) -> dict[str, str]:
    """The partition key, named for the unit so the two can never be confused."""
    if execution_size is not None:
        return {"execution_size": _format_size(execution_size)}
    return {"execution_notional": _format_size(execution_notional)}
