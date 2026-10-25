"""Deriving silver and gold from the bronze Komachi downloads.

One entry point, `run`, so the CLI does not have to know which dataset takes
which parameters. Each derivation is a pure function of one market-day: give it
the same bronze and it produces the same rows, which is what lets an article
cite a number and a reader reproduce it.

Which layer a dataset belongs to, and whether it is worth keeping on disk at
all, follows the medallion convention: silver is point-in-time and cheap, gold
is windowed and parameterised.
"""

from pathlib import Path

from ..layout import derived_path
from ..store import write_derived
from . import book_state, market_price, spread, volatility

DERIVATIONS = ("BookState", "MarketPrice", "Spread", "Volatility")
MODULES = {"BookState": book_state, "MarketPrice": market_price, "Spread": spread, "Volatility": volatility}

# Size in base currency by default, matching the warehouse: a quote-currency
# notional buys a different quantity every day as the price moves, so it is not
# comparable with itself over time. `--execution-notional` is there for the
# questions that really are about a cash amount.
DEFAULT_EXECUTION_SIZE = 0.002


def plan(
    dataset: str,
    *,
    execution_size=None,
    execution_notional=None,
    step=None,
    span=None,
    horizon=None,
    lookback=None,
    smooth=None,
):
    """The partition and keyword arguments one derivation will run with."""
    if dataset == "MarketPrice":
        if execution_size is None and execution_notional is None:
            execution_size = DEFAULT_EXECUTION_SIZE
        if execution_size is not None and execution_notional is not None:
            raise ValueError("Give --execution-size or --execution-notional, not both")
        kwargs = {"execution_size": execution_size, "execution_notional": execution_notional}
        return market_price.partition(execution_size, execution_notional), kwargs

    if dataset in ("Spread", "Volatility"):
        # No size is the top of the book, not a default size: the quoted
        # spread and a walked one are different numbers, kept apart.
        if execution_notional is not None:
            raise ValueError(f"{dataset} is measured at an --execution-size, not a notional")
        kwargs = {"execution_size": execution_size, "step": int(step or spread.DEFAULT_STEP)}
        if dataset == "Spread":
            kwargs["span"] = int(span or spread.DEFAULT_SPAN)
            return spread.partition(**kwargs), kwargs
        kwargs["horizon"] = int(horizon or volatility.DEFAULT_HORIZON)
        kwargs["lookback"] = int(lookback or volatility.DEFAULT_LOOKBACK)
        kwargs["smooth"] = int(smooth or volatility.DEFAULT_SMOOTH)
        return volatility.partition(**kwargs), kwargs

    if dataset == "BookState":
        return {}, {}

    raise ValueError(f"Unknown dataset {dataset!r}. Known: {', '.join(DERIVATIONS)}")


def run(
    root: Path, dataset: str, market: str, file_date: str, *, recreate: bool = False, **params
) -> tuple[Path, bool]:
    """Derive one market-day. Returns the path and whether it was computed.

    `params` are `plan`'s. Existing output is left alone unless `recreate`:
    re-running a range to fill a gap should cost the gap, not the range.
    """
    partition, kwargs = plan(dataset, **params)
    path = derived_path(root, dataset, market, file_date, partition)
    if path.is_file() and not recreate:
        return path, False
    frame = MODULES[dataset].derive(root, market, file_date, **kwargs)
    return write_derived(root, dataset, market, file_date, frame, partition), True
