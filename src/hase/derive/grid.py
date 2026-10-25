"""The quote on a regular clock: what Spread and Volatility are both measured on.

A book updates whenever the venue sends one -- twice a second on one venue,
twenty times on another, never during an outage. A return over a fixed horizon
and a window of fixed length only mean something on evenly spaced points, so
both measurements start by putting the quote on a clock: the last quote in each
step, carried forward through a step with no update, because a book that has
not updated has not changed.

This is not a dataset. It is a fixed, cheap function of data already on disk
(`BookState` or `MarketPrice`), about a tenth of a second a day, with no
parameter of its own beyond the step; storing it would only store a copy.
`quote_grid` hands it to anyone who wants the mid on a clock.

**Two sources.** With no `execution_size`, the top of the book (`BookState`).
With one, what that size would actually pay (`MarketPrice`). The spread of a
walked book is a different number from the quoted one, so every dataset
measured on the grid keeps the two in different partitions.

**The day is a Tokyo day, read with the eight hours before it.** A window or an
EMA at 00:00 JST needs the evening before, so the grid starts at 16:00 JST the
previous day when that day is on disk. Eight hours is decades of half-lives for
any span used here, so an EMA has forgotten where it started. Without the
previous day the grid is NaN until the day's first quote.

A value at grid point `t` is the last quote in `[t, t + step)`.
"""

import datetime as dt
from functools import lru_cache

import numpy as np

from .book_state import derive as derive_book_state

DAY = 86400
JST_OFFSET = 9 * 3600
WARMUP = 8 * 3600
DEFAULT_STEP = 1
TOUCH = "touch"


def second_grid(timestamps, values, *, min_seconds: int | None = None):
    """One value per second of the Tokyo day, from UTC epoch timestamps.

    Public because it is the primitive the articles share: mid from the book,
    price from the trades, and any other series that has to be put on a common
    clock before two of them can be compared.

    `min_seconds` returns None when fewer than that many seconds carried an
    observation of their own. Counted before the forward fill, which is the
    only place the question can be asked: afterwards every second from the
    first observation to the end of the day is filled, so counting then would
    say a day with four trades in it was complete.
    """
    ts = np.asarray(timestamps, dtype="float64")
    v = np.asarray(values, dtype="float64")
    ok = np.isfinite(ts) & np.isfinite(v) & (v > 0)
    if not ok.any():
        return None if min_seconds else np.full(DAY, np.nan)
    second = ((ts[ok] + JST_OFFSET) % DAY).astype(np.int64)
    if min_seconds is not None and len(np.unique(second)) < min_seconds:
        return None
    return last_in_slot(second, v[ok], DAY)


def last_in_slot(slot, values, n: int):
    """Last value in each of `n` slots, forward filled, NaN before the first."""
    grid = np.full(n, np.nan)
    grid[slot] = values  # later writes win, so this is last-in-slot
    seen = np.isfinite(grid)
    if not seen.any():
        return grid
    first = int(np.argmax(seen))
    carry = np.maximum.accumulate(np.where(seen, np.arange(n), 0))
    grid = grid[carry]
    grid[:first] = np.nan
    return grid


def day_start(file_date: str) -> float:
    """00:00 JST of a partition date, as a UTC epoch."""
    jst = dt.timezone(dt.timedelta(hours=9))
    return dt.datetime.fromisoformat(file_date).replace(tzinfo=jst).timestamp()


def size_key(execution_size: float | None) -> str:
    """The `execution_size=` partition value: the size, or `touch` for the top of the book."""
    from .market_price import _format_size

    return TOUCH if execution_size is None else _format_size(execution_size)


def check_step(step: int) -> int:
    step = int(step)
    if step <= 0 or DAY % step or WARMUP % step:
        raise ValueError(f"--step {step} does not divide a day into whole steps")
    return step


def _quotes(root, market: str, file_date: str, execution_size):
    if execution_size is None:
        book = derive_book_state(root, market, file_date)
    else:
        from ..dataset import load

        book = load(root, "MarketPrice", market, file_date, execution_size=execution_size)
    return (
        book["ts"].to_numpy(dtype="float64"),
        book["bid"].to_numpy(dtype="float64"),
        book["ask"].to_numpy(dtype="float64"),
    )


@lru_cache(maxsize=8)
def warm_grid(root, market: str, file_date: str, execution_size, step: int):
    """Bid and ask from 16:00 JST the day before to the end of the day.

    Returns `(bid, ask)`, `(WARMUP + DAY) // step` points each; the day's own
    points are the last `DAY // step`. Cached, so deriving Spread and several
    Volatility windows of one day reads its source once: the reading is most
    of the cost, and the grid is small.
    """
    from ..store import MissingDataError

    step = check_step(step)
    start = day_start(file_date) - WARMUP
    parts = []
    previous = (dt.date.fromisoformat(file_date) - dt.timedelta(days=1)).isoformat()
    try:
        parts.append(_quotes(root, market, previous, execution_size))
    except MissingDataError:
        pass  # no evening before: the day's first windows are short, and NaN
    parts.append(_quotes(root, market, file_date, execution_size))

    ts = np.concatenate([p[0] for p in parts])
    bid = np.concatenate([p[1] for p in parts])
    ask = np.concatenate([p[2] for p in parts])
    n = (WARMUP + DAY) // step
    slot = np.floor((ts - start) / step)
    ok = (slot >= 0) & (slot < n) & np.isfinite(bid) & np.isfinite(ask)
    order = np.argsort(ts[ok], kind="stable")  # last in slot must mean latest
    slot = slot[ok][order].astype(np.int64)
    return last_in_slot(slot, bid[ok][order], n), last_in_slot(slot, ask[ok][order], n)


def own_day(values, step: int):
    """The day's points of a warm grid, dropping the evening before."""
    return np.asarray(values)[WARMUP // step :]


def day_ts(file_date: str, step: int):
    """The epoch second each of the day's grid points stands for."""
    return day_start(file_date) + np.arange(0, DAY, step)


def quote_grid(root, market: str, file_date: str, *, execution_size: float | None = None, step: int = DEFAULT_STEP):
    """The day's quote on the clock: ts, bid, ask, mid and spread_bps."""
    import pandas as pd

    bid, ask = warm_grid(root, market, file_date, execution_size, check_step(step))
    bid, ask = own_day(bid, step), own_day(ask, step)
    mid = (bid + ask) / 2
    return pd.DataFrame(
        {"ts": day_ts(file_date, step), "bid": bid, "ask": ask, "mid": mid, "spread_bps": (ask - bid) / mid * 1e4}
    )
