"""Volatility: how far the mid moves, measured on the quote grid.

Realized volatility depends on how often you sample, so a volatility figure is
meaningless without the grid it was measured on; the grid is `grid.py`, and
every figure here names its step.

Three columns, all in bps:

- `rv_bps`: realized volatility over the trailing `lookback`, the square root
  of the summed squared one-step log returns. Sample fast enough and each
  return carries the bid-ask bounce rather than a move -- the volatility
  signature plot -- which is why the step is a parameter.
- `vol_bps`: the standard deviation of `horizon` returns over the trailing
  `lookback`. The horizon is the move a strategy holds for; the lookback is
  how much history estimates it. They are different parameters on purpose.
- `vol_bps_ema`: `vol_bps` smoothed over `smooth`.

Partitioned by the source (`execution_size=`) and the parameters
(`param_id=g<step>s_h<horizon>s_lb<lookback>s_ema<smooth>s`).
"""

import numpy as np

from .grid import DEFAULT_STEP, check_step, day_ts, own_day, warm_grid
from .grid import size_key as _size_key

DEFAULT_HORIZON = 60
DEFAULT_LOOKBACK = 300
DEFAULT_SMOOTH = 120
COLUMNS = ["ts", "rv_bps", "vol_bps", "vol_bps_ema"]


def _periods(seconds: float, step: float) -> int:
    return max(1, round(seconds / step))


def realized_vol_bps(mid_grid, step: int = 1):
    """Square root of the sum of squared log returns, sampling every `step` points.

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
    *,
    step: float = DEFAULT_STEP,
    horizon: float = DEFAULT_HORIZON,
    lookback: float = DEFAULT_LOOKBACK,
    smooth: float = DEFAULT_SMOOTH,
):
    """Volatility of a mid already on a grid of `step` seconds. Windows in seconds.

    The one implementation: the Volatility dataset calls it, and so does
    anything that needs the figures on a grid of its own -- a labeller
    resampling ticks. Returns a dict of numpy arrays, the grid's length each.
    """
    import pandas as pd

    mid = pd.Series(np.asarray(mid, dtype="float64"))
    window = _periods(lookback, step)
    returns = mid / mid.shift(_periods(horizon, step)) - 1
    vol = returns.rolling(window=window, min_periods=1).std() * 1e4
    vol_ema = vol.ewm(span=_periods(smooth, step), adjust=False).mean()
    return {"rv_bps": _rolling_rv_bps(mid, window), "vol_bps": vol.to_numpy(), "vol_bps_ema": vol_ema.to_numpy()}


def derive(
    root,
    market: str,
    file_date: str,
    *,
    execution_size=None,
    step=DEFAULT_STEP,
    horizon=DEFAULT_HORIZON,
    lookback=DEFAULT_LOOKBACK,
    smooth=DEFAULT_SMOOTH,
):
    """One row per `step` seconds of the Tokyo day."""
    import pandas as pd

    step = check_step(step)
    bid, ask = warm_grid(root, market, file_date, execution_size, step)
    figures = measure((bid + ask) / 2, step=step, horizon=horizon, lookback=lookback, smooth=smooth)
    frame = {"ts": day_ts(file_date, step)}
    frame.update({name: own_day(values, step) for name, values in figures.items()})
    return pd.DataFrame(frame, columns=COLUMNS)


def param_id(step: int, horizon: int, lookback: int, smooth: int) -> str:
    return f"g{step}s_h{horizon}s_lb{lookback}s_ema{smooth}s"


def partition(execution_size, step: int, horizon: int, lookback: int, smooth: int) -> dict[str, str]:
    return {"execution_size": _size_key(execution_size), "param_id": param_id(step, horizon, lookback, smooth)}
