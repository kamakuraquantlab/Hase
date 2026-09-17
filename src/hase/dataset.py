"""One way to get a derived dataset, whether or not it is on disk yet.

`hase derive` materialises. This loads: it returns the stored file when there
is one and computes the day in memory when there is not, so a script reads the
same frame either way.

That matters in two directions. A buyer runs `hase derive` once and every
script afterwards reads files. The seller points the same script at the
warehouse, where `silver/dataset=MarketPrice/...` may already exist because the
pipeline that produced the data built it -- and it is read rather than
recomputed, and never written over.
Nothing here writes; materialising is `hase derive`'s job alone.
"""

import sys
from pathlib import Path

from .derive import plan
from .derive import book_state, market_price, vol_spread
from .layout import derived_path

_MODULES = {"BookState": book_state, "MarketPrice": market_price, "VolSpread": vol_spread}

# Said once per dataset and market, not once per day.
_ANNOUNCED: set[tuple[str, str]] = set()


def load(root: Path, dataset: str, market: str, file_date: str, *, prefer_stored: bool = True,
         execution_size=None, execution_notional=None, step=None, lookback=None):
    """A derived dataset for one market-day, read if stored and derived if not."""
    import pyarrow.parquet as pq

    params, kwargs = plan(dataset, execution_size=execution_size,
                          execution_notional=execution_notional, step=step, lookback=lookback)
    path = derived_path(root, dataset, market, file_date, params)
    if prefer_stored and path.is_file():
        own = pq.ParquetFile(path).schema_arrow.names
        if _usable(dataset, own):
            return _normalise(dataset, pq.read_table(path, columns=own).to_pandas(), kwargs)
        _announce(dataset, market, path)
    return _MODULES[dataset].derive(root, market, file_date, **kwargs)


def _usable(dataset: str, columns) -> bool:
    """Whether a stored file is the dataset Hase means by that name.

    A path is not a schema. Another tool may well write a `BookState` of its
    own at the same address, holding a different table -- `best_bid` and
    `best_spread_bps` where Hase has `bid` and `spread_bps`, say. Reading it
    because the directory name matched would hand a script a frame missing the
    columns it is about to ask for, which is the good case; the bad case is a
    column that exists under the same name and means something else.

    So the name is not enough. The columns have to be there too.
    """
    required = set(_MODULES[dataset].COLUMNS)
    return required.issubset(set(columns))


def _announce(dataset: str, market: str, path: Path) -> None:
    key = (dataset, market)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    print(f"note: {path.parent.parent} holds a different {dataset}; deriving instead",
          file=sys.stderr)


def _normalise(dataset: str, frame, kwargs):
    """Give a stored frame the columns a derived one would have.

    A MarketPrice written by an older tool may predate `filled` and
    `degenerate` and drop the rows they would have described. Rebuilding them
    from what is there means a script sees one shape whichever source it read,
    instead of branching on where the data came from.
    """
    import numpy as np

    if dataset != "MarketPrice":
        return frame
    if "filled" not in frame:
        frame["filled"] = np.isfinite(frame["bid"].to_numpy()) & np.isfinite(frame["ask"].to_numpy())
    if "degenerate" not in frame:
        limit = kwargs.get("degenerate_spread_bps", market_price.DEGENERATE_SPREAD_BPS)
        spread = frame["spread_bps"].to_numpy()
        frame["degenerate"] = np.isfinite(spread) & (spread > limit)
    return frame


def stored(root: Path, dataset: str, market: str, file_date: str, **params) -> bool:
    """Whether the day is already on disk, without reading it."""
    partition, _ = plan(dataset, **params)
    return derived_path(root, dataset, market, file_date, partition).is_file()
