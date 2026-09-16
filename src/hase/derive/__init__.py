"""Deriving silver and gold from the bronze Komachi downloads.

One entry point, `run`, so the CLI does not have to know which dataset takes
which parameters. Each derivation is a pure function of one market-day: give it
the same bronze and it produces the same rows, which is what lets an article
cite a number and a reader reproduce it.

`system/04_hase-analysis-toolkit.md` section 5 decides what lives in which
layer, and which of these are worth keeping on disk at all.
"""

from pathlib import Path

from ..layout import derived_path
from ..store import write_derived
from . import book_state, market_price, vol_spread

DERIVATIONS = ("BookState", "MarketPrice", "VolSpread")

# Size in base currency by default, matching the warehouse: a quote-currency
# notional buys a different quantity every day as the price moves, so it is not
# comparable with itself over time. `--execution-notional` is there for the
# questions that really are about a cash amount.
DEFAULT_EXECUTION_SIZE = 0.002


def plan(dataset: str, *, execution_size=None, execution_notional=None, step=None, lookback=None):
    """The partition and keyword arguments one derivation will run with."""
    if dataset == "MarketPrice":
        if execution_size is None and execution_notional is None:
            execution_size = DEFAULT_EXECUTION_SIZE
        if execution_size is not None and execution_notional is not None:
            raise ValueError("Give --execution-size or --execution-notional, not both")
        kwargs = {"execution_size": execution_size, "execution_notional": execution_notional}
        return market_price.partition(execution_size, execution_notional), kwargs

    if dataset == "VolSpread":
        step = step or vol_spread.DEFAULT_STEP
        lookback = lookback or vol_spread.DEFAULT_LOOKBACK
        return {"param_id": vol_spread.param_id(step, lookback)}, {"step": step, "lookback": lookback}

    if dataset == "BookState":
        return {}, {}

    raise ValueError(f"Unknown dataset {dataset!r}. Known: {', '.join(DERIVATIONS)}")


def run(root: Path, dataset: str, market: str, file_date: str, *, recreate: bool = False,
        execution_size=None, execution_notional=None, step=None, lookback=None) -> tuple[Path, bool]:
    """Derive one market-day. Returns the path and whether it was computed.

    Existing output is left alone unless `recreate`, matching Makalu's
    `populate`: re-running a range to fill a gap should cost the gap, not the
    range.
    """
    params, kwargs = plan(dataset, execution_size=execution_size,
                          execution_notional=execution_notional, step=step, lookback=lookback)
    path = derived_path(root, dataset, market, file_date, params)
    if path.is_file() and not recreate:
        return path, False

    module = {"BookState": book_state, "MarketPrice": market_price, "VolSpread": vol_spread}[dataset]
    frame = module.derive(root, market, file_date, **kwargs)
    return write_derived(root, dataset, market, file_date, frame, params), True
