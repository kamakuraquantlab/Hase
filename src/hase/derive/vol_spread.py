"""VolSpread: bid and ask on a regular grid, and the spread and volatility measured on it.

Realized volatility depends on how often you sample, so a volatility figure is
meaningless without the grid it was measured on. That grid is the dataset;
the volatility is a function of it.

Articles 2, 3 and 4 each built their own version of this -- per second, per
minute, per hour -- from the same order book, and the warehouse pipeline built
a fourth from the execution price, for its activity curves and labels. They are
one calculation, and this is where it lives: every spread and volatility figure
on the grid comes out of `measure`, whoever asks for it.

**Two sources, one calculation.** By default the grid is the top of the book
(`BookState`), which is what the articles read. Given an `execution_size`, it
is what that size would actually pay (`MarketPrice`), which is what a strategy
pays and what the warehouse's activity curves are built on. The two land in
different partitions -- `execution_size=touch` and `execution_size=0.002` --
because the spread of a walked book is a different number from the quoted one.

**Parameters.** `step` is the grid, in seconds. `horizon` is the return the
volatility is measured on, `lookback` the window it is measured over, and
`smooth` the EMA span that smooths it; the spread's EMA uses `lookback`. All in
seconds, and all in the partition (`param_id=vs_<horizon>_<lookback>_<smooth>_<step>s`),
so two parameterisations sit beside each other rather than overwriting.

**Columns.**

- `bid`, `ask`, `mid`, `spread`, `spread_pct`, `spread_bps`: the last quote in
  each step, carried forward through a step with no update.
- `rv_bps`: realized volatility over the trailing `lookback`, the square root of
  the summed squared one-step log returns, in bps.
- `volatility_raw`: the standard deviation of `horizon` returns over the
  trailing `lookback`, as a fraction; `volatility_ema` is it smoothed over
  `smooth`, and `spread_pct_ema` the spread smoothed over `lookback`.
- `vol_spread_ratio`: `volatility_ema / spread_pct_ema` -- how far the price
  moves for what a round trip costs.

**The day is a Tokyo day, read with the eight hours before it.** The windows
and EMAs at 00:00 JST need the evening before, so the grid starts at 16:00 JST
the previous day when that day is on disk, and only the day's own rows are
returned. Eight hours is what the warehouse pipeline read, and is decades of
half-lives for any span used here, so the EMA has forgotten where it started.
Without the previous day the first windows of the day are short, and say so
by being NaN. A step with no quote yet is NaN too: there is nothing to carry.

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
DEFAULT_HORIZON = 60
DEFAULT_LOOKBACK = 300
DEFAULT_SMOOTH = 120

TOUCH = "touch"
COLUMNS = [
    "ts",
    "sec",
    "bid",
    "ask",
    "mid",
    "spread",
    "spread_pct",
    "spread_bps",
    "rv_bps",
    "volatility_raw",
    "volatility_ema",
    "spread_pct_ema",
    "vol_spread_ratio",
]


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
    return _grid(second, v[ok], DAY)


def _grid(slot, values, n: int):
    """Last value in each slot, forward filled, NaN before the first."""
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
    return float(np.sqrt((returns**2).sum()) * 1e4)


def _periods(seconds: float, step: float) -> int:
    return max(1, round(seconds / step))


def _rolling_rv_bps(mid, window: int):
    """Realized volatility over a trailing window of `window` returns, per grid point."""
    mid = np.asarray(mid, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.diff(np.log(mid), prepend=np.nan) ** 2
    r2 = np.where(np.isfinite(r2), r2, 0.0)
    cumulative = np.concatenate(([0.0], np.cumsum(r2)))
    hi = np.arange(1, len(mid) + 1)
    lo = np.maximum(hi - window, 0)
    out = np.sqrt(cumulative[hi] - cumulative[lo]) * 1e4
    # Not a full window of returns since the first quote: not comparable.
    seen = np.isfinite(mid)
    first = int(np.argmax(seen)) if seen.any() else len(mid)
    out[: first + window] = np.nan
    return out


def measure(
    mid,
    spread,
    *,
    step: float = DEFAULT_STEP,
    horizon: float = DEFAULT_HORIZON,
    lookback: float = DEFAULT_LOOKBACK,
    smooth: float = DEFAULT_SMOOTH,
    spread_smooth: float | None = None,
):
    """Spread and volatility of a mid and spread already on a grid of `step` seconds.

    The one implementation: VolSpread calls it, and so does anything else that
    needs these figures on its own grid -- a labeller resampling a tick frame,
    say. Windows are in seconds and become grid points here. `spread_smooth`
    defaults to `lookback`, the span VolSpread smooths the spread over.

    Returns a dict of numpy arrays, the grid's length each.
    """
    import pandas as pd

    mid = pd.Series(np.asarray(mid, dtype="float64"))
    spread = pd.Series(np.asarray(spread, dtype="float64"))
    window = _periods(lookback, step)

    spread_pct = spread / mid
    returns = mid / mid.shift(_periods(horizon, step)) - 1
    volatility_raw = returns.rolling(window=window, min_periods=1).std()
    volatility_ema = volatility_raw.ewm(span=_periods(smooth, step), adjust=False).mean()
    spread_span = _periods(spread_smooth if spread_smooth is not None else lookback, step)
    spread_pct_ema = spread_pct.ewm(span=spread_span, adjust=False).mean()
    return {
        "spread_pct": spread_pct.to_numpy(),
        "spread_bps": (spread_pct * 1e4).to_numpy(),
        "rv_bps": _rolling_rv_bps(mid, window),
        "volatility_raw": volatility_raw.to_numpy(),
        "volatility_ema": volatility_ema.to_numpy(),
        "spread_pct_ema": spread_pct_ema.to_numpy(),
        "vol_spread_ratio": (volatility_ema / spread_pct_ema.where(spread_pct_ema > 0)).to_numpy(),
    }


def day_start(file_date: str) -> float:
    """00:00 JST of a partition date, as a UTC epoch."""
    jst = dt.timezone(dt.timedelta(hours=9))
    return dt.datetime.fromisoformat(file_date).replace(tzinfo=jst).timestamp()


def _quotes(root, market: str, file_date: str, execution_size):
    """Timestamps, bid and ask of one day from the source this VolSpread is measured on."""
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
def _source_grid(root, market: str, file_date: str, execution_size, step: int):
    """Bid and ask on the grid from 16:00 JST the day before to the end of the day.

    Cached, so deriving several parameterisations of one day reads its source
    once: the reading is most of the cost, and the grid is small.
    """
    from ..store import MissingDataError

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
    return start, _grid(slot, bid[ok][order], n), _grid(slot, ask[ok][order], n)


def derive(
    root,
    market: str,
    file_date: str,
    *,
    execution_size: float | None = None,
    step: int = DEFAULT_STEP,
    horizon: int = DEFAULT_HORIZON,
    lookback: int = DEFAULT_LOOKBACK,
    smooth: int = DEFAULT_SMOOTH,
):
    """One row per `step` seconds of the Tokyo day."""
    import pandas as pd

    if DAY % step or WARMUP % step:
        raise ValueError(f"--step {step} does not divide a day into whole steps")
    _, bid, ask = _source_grid(root, market, file_date, execution_size, step)
    mid = (bid + ask) / 2
    spread = ask - bid
    figures = measure(mid, spread, step=step, horizon=horizon, lookback=lookback, smooth=smooth)

    own = slice(WARMUP // step, None)  # the day's rows, after the evening before
    index = np.arange(0, DAY, step)
    frame = {"ts": day_start(file_date) + index, "sec": index}
    frame.update(bid=bid[own], ask=ask[own], mid=mid[own], spread=spread[own])
    frame.update({name: values[own] for name, values in figures.items()})
    return pd.DataFrame(frame, columns=COLUMNS)


def param_id(step: int, horizon: int, lookback: int, smooth: int) -> str:
    """The partition name of one parameterisation, the warehouse's own spelling."""
    return f"vs_{horizon}_{lookback}_{smooth}_{step}s"


def partition(execution_size: float | None, step: int, horizon: int, lookback: int, smooth: int) -> dict[str, str]:
    from .market_price import _format_size

    size = TOUCH if execution_size is None else _format_size(execution_size)
    return {"execution_size": size, "param_id": param_id(step, horizon, lookback, smooth)}
