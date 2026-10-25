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

from .derive import MODULES, plan
from .layout import derived_path

# Said once per dataset and market, not once per day.
_ANNOUNCED: set[tuple[str, str]] = set()


def load(root: Path, dataset: str, market: str, file_date: str, *, prefer_stored: bool = True, **params):
    """A derived dataset for one market-day, read if stored and derived if not."""
    import pyarrow.parquet as pq

    params, kwargs = plan(dataset, **params)
    path = derived_path(root, dataset, market, file_date, params)
    if prefer_stored and path.is_file():
        own = pq.ParquetFile(path).schema_arrow.names
        if _usable(dataset, own):
            return pq.read_table(path, columns=own).to_pandas()
        _announce(dataset, market, path)
    return MODULES[dataset].derive(root, market, file_date, **kwargs)


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
    required = set(MODULES[dataset].COLUMNS)
    return required.issubset(set(columns))


def _announce(dataset: str, market: str, path: Path) -> None:
    key = (dataset, market)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    print(f"note: {path.parent.parent} holds a different {dataset}; deriving instead", file=sys.stderr)


def stored(root: Path, dataset: str, market: str, file_date: str, **params) -> bool:
    """Whether the day is already on disk, without reading it."""
    partition, _ = plan(dataset, **params)
    return derived_path(root, dataset, market, file_date, partition).is_file()


def trade_count(root: Path, market: str, file_date: str) -> int | None:
    """How many trades a day holds, or None if the day is not downloaded.

    From the parquet footer, so it costs a file open rather than a read. The
    count is a fact about the day that needs no derivation, and a caller
    wanting it should not have to open a parquet file to get it.
    """
    import komachi.bronze
    import pyarrow.parquet as pq

    try:
        path = komachi.bronze.file(market, "Trade", file_date, root)
    except komachi.bronze.MissingData:
        return None
    return pq.ParquetFile(path).metadata.num_rows
