"""BookState: what the top of the book looked like, snapshot by snapshot.

The cheapest derivation and the one every article reads. Articles 1 to 4 each
opened the raw OrderBook file and recomputed mid and spread from `bid0_price`
and `ask0_price` themselves; this is that calculation, written once.

A crossed or empty book is dropped rather than carried as a negative spread.
Those rows are not a market state, they are a snapshot taken mid-update, and
every article already filtered them the same way -- finite, `bid > 0`,
`ask >= bid`.
"""

import numpy as np

from ..store import load_order_book

COLUMNS = ["ts", "bid", "ask", "mid", "spread", "spread_bps", "bid_qty", "ask_qty", "top_of_book"]


def derive(root, market: str, file_date: str):
    """One row per order book snapshot with a usable two-sided quote."""
    import pandas as pd

    raw = load_order_book(root, market, file_date, depth=1)
    bid = raw["bid0_price"].to_numpy(dtype="float64")
    ask = raw["ask0_price"].to_numpy(dtype="float64")
    bid_qty = raw["bid0_qty"].to_numpy(dtype="float64")
    ask_qty = raw["ask0_qty"].to_numpy(dtype="float64")

    ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask >= bid)
    if not ok.any():
        return pd.DataFrame(columns=COLUMNS)

    bid, ask = bid[ok], ask[ok]
    bid_qty, ask_qty = bid_qty[ok], ask_qty[ok]
    mid = (bid + ask) / 2

    return pd.DataFrame({
        "ts": raw["ts"].to_numpy(dtype="float64")[ok],
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread": ask - bid,
        "spread_bps": (ask - bid) / mid * 1e4,
        "bid_qty": bid_qty,
        "ask_qty": ask_qty,
        # Quote-currency notional resting at the touch, which is the depth
        # figure article 1 reports. The smaller side, because a round trip is
        # limited by whichever side is thinner.
        "top_of_book": np.minimum(bid_qty * bid, ask_qty * ask),
    })
