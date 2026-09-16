"""VolSpread: mid and spread on a regular grid, and volatility measured on it.

Realized volatility depends on how often you sample, so a volatility figure is
meaningless without the grid it was measured on. That grid is the dataset;
the volatility is a function of it.

Articles 2, 3 and 4 each built their own version of this -- per second, per
minute, per hour -- from the same order book. They are one grid at three
decimations, so Hase derives the finest of them and the others fall out.

The day is a Tokyo day, and the grid is indexed by second within it. A second
with no snapshot carries the last price seen, because a book with no update is
a book that has not changed; seconds before the first snapshot of the day are
NaN, because there is nothing to carry forward from.
"""

import numpy as np

from .book_state import derive as derive_book_state

DAY = 86400
JST_OFFSET = 9 * 3600
DEFAULT_STEP = 1
DEFAULT_LOOKBACK = 300
COLUMNS = ["ts", "sec", "mid", "spread_bps", "rv_bps"]


def second_grid(timestamps, values):
    """One value per second of the Tokyo day, from UTC epoch timestamps.

    Public because it is the primitive the articles share: mid from the book,
    price from the trades, and any other series that has to be put on a common
    clock before two of them can be compared.
    """
    import numpy as _np

    ts = _np.asarray(timestamps, dtype="float64")
    v = _np.asarray(values, dtype="float64")
    ok = _np.isfinite(ts) & _np.isfinite(v) & (v > 0)
    if not ok.any():
        return _np.full(DAY, _np.nan)
    return _grid(((ts[ok] + JST_OFFSET) % DAY).astype(_np.int64), v[ok])


def _grid(second_of_day, values):
    """Last value in each second, forward filled, NaN before the first."""
    grid = np.full(DAY, np.nan)
    grid[second_of_day] = values          # later writes win, so this is last-in-second
    seen = np.isfinite(grid)
    if not seen.any():
        return grid
    first = int(np.argmax(seen))
    carry = np.maximum.accumulate(np.where(seen, np.arange(DAY), 0))
    grid = grid[carry]
    grid[:first] = np.nan
    return grid


def realized_vol_bps(mid_grid, step: int = 1):
    """Square root of the sum of squared log returns, sampling every `step` seconds.

    In bps over the whole grid, which is what makes two sampling intervals
    comparable. Sample fast enough and every observation carries the bid-ask
    bounce rather than a price move, and this inflates without bound -- that
    inflation is the volatility signature plot, and its slope is the spread.
    """
    sample = np.asarray(mid_grid)[::step]
    with np.errstate(invalid="ignore", divide="ignore"):
        returns = np.diff(np.log(sample))
    returns = returns[np.isfinite(returns)]
    if len(returns) < 10:
        return None
    return float(np.sqrt((returns ** 2).sum()) * 1e4)


def _rolling_rv_bps(mid_grid, lookback: int):
    """Realized volatility over a trailing window, one value per grid point."""
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.diff(np.log(mid_grid)) ** 2
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(r2)))
    hi = np.arange(len(mid_grid))
    lo = np.maximum(hi - lookback, 0)
    out = np.sqrt(cumulative[hi] - cumulative[lo]) * 1e4
    out[:lookback] = np.nan          # not a full window yet, so not comparable
    return out


def derive(root, market: str, file_date: str, *,
           step: int = DEFAULT_STEP, lookback: int = DEFAULT_LOOKBACK):
    """One row per `step` seconds of the Tokyo day."""
    import pandas as pd

    book = derive_book_state(root, market, file_date)
    if book.empty:
        return pd.DataFrame(columns=COLUMNS)

    ts = book["ts"].to_numpy(dtype="float64")
    sec = ((ts + JST_OFFSET) % DAY).astype(np.int64)
    mid = _grid(sec, book["mid"].to_numpy(dtype="float64"))
    spread = _grid(sec, book["spread_bps"].to_numpy(dtype="float64"))
    rv = _rolling_rv_bps(mid, lookback)

    # The epoch second each grid slot stands for: JST midnight of this
    # partition, plus the offset. Derived from the partition rather than from
    # the data, so a day that starts late still lines up with every other day.
    day_start = ts[0] - ((ts[0] + JST_OFFSET) % DAY)
    index = np.arange(0, DAY, step)
    return pd.DataFrame({
        "ts": day_start + index,
        "sec": index,
        "mid": mid[::step],
        "spread_bps": spread[::step],
        "rv_bps": rv[::step],
    })


def param_id(step: int, lookback: int) -> str:
    return f"g{step}s_rv{lookback}s"
