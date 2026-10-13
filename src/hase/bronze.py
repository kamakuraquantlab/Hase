"""Bronze as trading data: `Market`, `Trade` and `OrderBook`, not files.

The question a backtest or a strategy asks is "the BINANCE:BTC_USDT order books
from the 1st to the 7th". Komachi answers a different one -- which file holds
that market-day, and whether it is complete -- and knows nothing about what is
inside. This module translates the first question into the second, opens what
Komachi hands over, and returns `qate`'s objects:

    hase.bronze.load(market, DataType.ORDER_BOOK, "2026-09-01", "2026-09-07")

Nothing here builds a path or decides what a complete day is; that is
`komachi.bronze`, and asking it is what keeps every reader in step with the
downloader. Turning a frame into objects is here and nowhere else.

## The two datasets

`Trade` is `ts, side, price, size`, with `side` 0 for a buy and 1 for a sell.
`OrderBook` is `ts` plus `bid<i>_price, bid<i>_qty` and `ask<i>_price,
ask<i>_qty` per level.

Depth is read from the file's columns rather than assumed: it differs by venue
and has changed across the archive. A level whose price is absent ends the
book -- and absent has two spellings, because a column that is null for every
row in a file comes back as `None` where parquet typed it null, while one null
for only some rows comes back as `NaN` in a float column.

## Dates are JST

Every `date=` partition is an Asia/Tokyo day spanning 15:00-14:59 UTC, and the
`ts` inside the files is a UTC epoch. A range here is a range of those days,
inclusive; trimming to the second is the caller's.
"""

import datetime as dt
from enum import StrEnum
from logging import getLogger
from pathlib import Path

import komachi
import komachi.bronze
from qate.core.model import Market, OrderBook, OrderLevel, Side, Trade

from . import store
from .store import MissingDataError, book_depth  # noqa: F401  (raised by load, so importable here)

LOG = getLogger(__name__)


class DataType(StrEnum):
    """A bronze dataset. The values are Komachi's partition names."""

    TRADE = "Trade"
    ORDER_BOOK = "OrderBook"


class MalformedDataError(ValueError):
    """A file that is on disk but is not the dataset its path says."""


Day = str | dt.date


def data_root(override: str | Path | None = None, *, env: bool = False) -> Path:
    """Where the data is: Komachi's answer, asked once here for callers of Hase.

    `env=True` is Komachi's rule for a command line -- `ROOT_PATH` in the
    environment is a per-run answer somebody typed on purpose. A script passes
    nothing and gets the settings file.
    """
    return komachi.data_root(str(override) if override else None, env=env)


def _root(root: str | Path | None) -> Path:
    return Path(root).expanduser() if root else data_root()


def markets(root: str | Path | None = None) -> list[Market]:
    """Every market on disk that `qate` has a name for.

    A market Komachi holds but `qate` has no `ExchangeName` or `Symbol` for is
    skipped rather than raised on: it is real data, readable as a frame, and
    simply not something this install can turn into objects.
    """
    found = []
    for market_id in komachi.bronze.markets(_root(root)):
        try:
            found.append(Market.from_str(market_id))
        except KeyError:
            LOG.info(f"Skipping {market_id}: not a market qate knows")
    return found


def dates(market: Market, data_type: DataType, root: str | Path | None = None) -> list[str]:
    """Which JST days of a dataset are complete on disk, sorted."""
    return komachi.bronze.dates(market.id, DataType(data_type), _root(root))


def load_day(market: Market, data_type: DataType, day: Day,
             root: str | Path | None = None) -> list[Trade] | list[OrderBook]:
    """One market-day, in file order -- which for bronze is timestamp order.

    Raises `MissingDataError` if the day is not complete on disk.
    """
    data_type = DataType(data_type)
    root, day = _root(root), str(day)
    if data_type is DataType.TRADE:
        return to_trades(store.load_trades(root, market.id, day), market)
    return to_order_books(store.load_order_book(root, market.id, day, depth=None), market)


def load(market: Market, data_type: DataType, start: Day, end: Day,
         root: str | Path | None = None) -> list[Trade] | list[OrderBook]:
    """Every day from `start` to `end` inclusive, concatenated in order.

    Strict: a missing day raises `MissingDataError` rather than leaving a hole
    that looks like a quiet market. A caller that would rather skip one asks
    `dates` first, or calls `load_day` and catches it.
    """
    out: list = []
    for day in _days(start, end):
        out.extend(load_day(market, data_type, day, root))
    return out


def _days(start: Day, end: Day) -> list[str]:
    first, last = (d if isinstance(d, dt.date) else dt.date.fromisoformat(d) for d in (start, end))
    return [(first + dt.timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def to_trades(df, market: Market) -> list[Trade]:
    """A bronze trade frame as `Trade`s."""
    if df.empty:
        return []
    symbol, exchange_name = market.symbol, market.exchange_name
    return [
        Trade("", Side.BUY if row.side == 0 else Side.SELL, row.price, row.size,
              symbol, exchange_name, row.ts)
        for row in df.itertuples(index=False)
    ]


def to_order_books(df, market: Market) -> list[OrderBook]:
    """A bronze order book frame as `OrderBook`s, at whatever depth it holds."""
    if df.empty:
        return []
    depth = book_depth(df.columns)
    if depth == 0:
        raise MalformedDataError(
            f"OrderBook frame for {market.id} has no bid/ask level columns: {list(df.columns)[:8]}"
        )
    symbol, exchange_name = market.symbol, market.exchange_name

    # Read by position, not by name: itertuples over 80+ named columns is
    # measurably slower, and a full-depth day is millions of rows.
    ordered = ["ts"]
    for side in ("bid", "ask"):
        for i in range(depth):
            ordered += [f"{side}{i}_price", f"{side}{i}_qty"]
    ask_start = 1 + 2 * depth
    return [
        OrderBook(_levels(row, 1, depth), _levels(row, ask_start, depth), symbol, exchange_name, row[0])
        for row in df[ordered].itertuples(index=False, name=None)
    ]


def _levels(row: tuple, start: int, depth: int) -> list[OrderLevel]:
    """Price/qty pairs from `start`, stopping at the first absent level.

    Absent levels are only ever a tail: a book with four levels fills 0-3 and
    leaves the rest null. Stopping is right -- levels below the touch are not
    optional, and a book with a hole in it is not a book.

    Both spellings of absent are handled. `price != price` is the NaN test,
    cheaper than `math.isnan` in a loop this hot, and it does not catch `None` --
    which is what a wholly-null column reads back as.
    """
    levels = []
    for i in range(depth):
        price = row[start + 2 * i]
        if price is None or price != price:  # noqa: PLR0124
            break
        levels.append(OrderLevel(price, row[start + 2 * i + 1]))
    return levels

