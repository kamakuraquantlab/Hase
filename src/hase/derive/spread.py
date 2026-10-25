"""Spread: what a round trip costs, smoothed over a window, on the quote grid.

The spread itself is a fact about each quote and is on the grid already
(`grid.quote_grid`'s `spread_bps`); what this dataset adds is the part with a
parameter, the spread's exponential moving average. That is the number a
strategy compares its expected move with: the spread right now is noisy, the
spread it should expect to pay is not.

In bps, as every spread and volatility figure Hase makes is, so the two can be
divided without a conversion. Partitioned by the source (`execution_size=`) and
the parameters (`param_id=g<step>s_ema<span>s`).
"""

import numpy as np

from .grid import DEFAULT_STEP, check_step, day_ts, own_day, size_key, warm_grid

DEFAULT_SPAN = 300
COLUMNS = ["ts", "spread_bps_ema"]


def _periods(seconds: float, step: float) -> int:
    return max(1, round(seconds / step))


def measure(spread_bps, *, step: float = DEFAULT_STEP, span: float = DEFAULT_SPAN):
    """The EMA of a spread already on a grid of `step` seconds, over `span` seconds.

    The one implementation: the Spread dataset calls it, and so does anything
    that needs the figure on a grid of its own -- a labeller resampling ticks.
    """
    import pandas as pd

    series = pd.Series(np.asarray(spread_bps, dtype="float64"))
    return series.ewm(span=_periods(span, step), adjust=False).mean().to_numpy()


def derive(root, market: str, file_date: str, *, execution_size=None, step=DEFAULT_STEP, span=DEFAULT_SPAN):
    """One row per `step` seconds of the Tokyo day."""
    import pandas as pd

    step = check_step(step)
    bid, ask = warm_grid(root, market, file_date, execution_size, step)
    spread_bps = (ask - bid) / ((bid + ask) / 2) * 1e4
    ema = measure(spread_bps, step=step, span=span)
    return pd.DataFrame({"ts": day_ts(file_date, step), "spread_bps_ema": own_day(ema, step)}, columns=COLUMNS)


def param_id(step: int, span: int) -> str:
    return f"g{step}s_ema{span}s"


def partition(execution_size, step: int, span: int) -> dict[str, str]:
    return {"execution_size": size_key(execution_size), "param_id": param_id(step, span)}
