"""Bronze as qate objects: schemas, ranges, and the ragged edges of a book.

Which file, and whether a day is complete, is Komachi's answer; these tests
check that `hase.bronze` asks it and converts what comes back.
"""

import datetime as dt

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from qate.core.model import ExchangeName, Market, Side
from qate.core.symbol import Symbol

from hase import bronze
from hase.bronze import DataType
from hase.layout import data_path

MARKET = Market(ExchangeName.COINCHECK, Symbol.BTC_SPOT)
DATE = "2026-01-15"
TRADE_COLUMNS = ["ts", "side", "price", "size"]


def write_parquet(root, data_type, market, date_str, rows: list[dict], columns: list[str]):
    path = data_path(root, market.id, data_type, date_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows, columns=columns), preserve_index=False), path)
    return path


def trade(ts: float, side: int = 0) -> dict:
    return {"ts": ts, "side": side, "price": 1.0, "size": 1.0}


def book_columns(depth: int) -> list[str]:
    cols = ["ts"]
    for side in ("bid", "ask"):
        for i in range(depth):
            cols += [f"{side}{i}_price", f"{side}{i}_qty"]
    return cols


def book_row(ts: float, bids: list[tuple], asks: list[tuple], depth: int) -> dict:
    row = {"ts": ts}
    for side, levels in (("bid", bids), ("ask", asks)):
        for i in range(depth):
            price, qty = levels[i] if i < len(levels) else (None, None)
            row[f"{side}{i}_price"] = price
            row[f"{side}{i}_qty"] = qty
    return row


def test_trades_are_read_in_file_order(root):
    write_parquet(
        root,
        DataType.TRADE,
        MARKET,
        DATE,
        [
            {"ts": 100.0, "side": 0, "price": 15_000_000.0, "size": 0.01},
            {"ts": 100.5, "side": 1, "price": 15_000_100.0, "size": 0.02},
        ],
        TRADE_COLUMNS,
    )

    first, second = bronze.load_day(MARKET, DataType.TRADE, DATE, root)
    assert (first.side, first.price, first.size, first.get_ts()) == (Side.BUY, 15_000_000.0, 0.01, 100.0)
    assert second.side == Side.SELL
    assert first.market == MARKET


def test_order_books_are_read_with_depth_from_the_columns(root):
    depth = 3
    write_parquet(
        root,
        DataType.ORDER_BOOK,
        MARKET,
        DATE,
        [
            book_row(100.0, [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)], [(101.0, 1.5), (102.0, 2.5), (103.0, 3.5)], depth),
        ],
        book_columns(depth),
    )

    [book] = bronze.load_day(MARKET, DataType.ORDER_BOOK, DATE, root)
    assert [level.price for level in book.bids] == [99.0, 98.0, 97.0]
    assert [level.amount for level in book.asks] == [1.5, 2.5, 3.5]
    assert (book.best_bid, book.best_ask, book.get_ts()) == (99.0, 101.0, 100.0)
    assert book.market == MARKET


def test_a_short_book_stops_at_the_absent_level(root):
    """A venue that showed two levels must not read back as twenty."""
    depth = 20
    write_parquet(
        root,
        DataType.ORDER_BOOK,
        MARKET,
        DATE,
        [book_row(100.0, [(99.0, 1.0), (98.0, 2.0)], [(101.0, 1.5)], depth)],
        book_columns(depth),
    )

    [book] = bronze.load_day(MARKET, DataType.ORDER_BOOK, DATE, root)
    assert (len(book.bids), len(book.asks)) == (2, 1)


def test_rows_of_different_depths_each_stop_at_their_own_absent_level(root):
    """Depth is found for the whole frame at once; it must still be each row's own.

    Rows of mixed depth make the level columns hold NaN in some rows, and a
    column that is null in every row reads back as None: both are absent.
    """
    depth = 4
    write_parquet(
        root,
        DataType.ORDER_BOOK,
        MARKET,
        DATE,
        [
            book_row(100.0, [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)], [(101.0, 1.0)], depth),
            book_row(101.0, [(99.5, 1.0)], [(101.5, 1.0), (102.5, 2.0)], depth),
            book_row(102.0, [], [(103.0, 1.0), (104.0, 2.0), (105.0, 3.0)], depth),
        ],
        book_columns(depth),
    )

    books = bronze.load_day(MARKET, DataType.ORDER_BOOK, DATE, root)
    assert [[(lv.price, lv.amount) for lv in b.bids] for b in books] == [
        [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)],
        [(99.5, 1.0)],
        [],
    ]
    assert [[lv.price for lv in b.asks] for b in books] == [[101.0], [101.5, 102.5], [103.0, 104.0, 105.0]]
    assert all(type(lv.price) is float for b in books for lv in b.bids + b.asks)


def test_a_recorded_book_computes_each_market_price_once(root):
    """Recorded books never change, so the answer is shared -- and is the same answer."""
    from qate.core.model import OrderBook

    depth = 3
    write_parquet(
        root,
        DataType.ORDER_BOOK,
        MARKET,
        DATE,
        [
            book_row(100.0, [(99.0, 1.0), (98.0, 2.0), (97.0, 3.0)], [(101.0, 1.5), (102.0, 2.5), (103.0, 3.5)], depth),
        ],
        book_columns(depth),
    )
    [book] = bronze.load_day(MARKET, DataType.ORDER_BOOK, DATE, root)

    assert isinstance(book, OrderBook)
    first = book.market_price(2.0)
    assert book.market_price(2.0) is first
    assert first == OrderBook.market_price(book, 2.0)
    assert book.market_price(1.0) == OrderBook.market_price(book, 1.0)
    assert book.market_price(100.0) is None  # more than the book holds, and remembered as such
    assert book.market_price(100.0) is None


def test_a_book_file_with_no_levels_is_an_error(root):
    write_parquet(root, DataType.ORDER_BOOK, MARKET, DATE, [{"ts": 1.0}], ["ts"])
    with pytest.raises(bronze.MalformedDataError):
        bronze.load_day(MARKET, DataType.ORDER_BOOK, DATE, root)


def test_a_range_is_inclusive_and_in_order(root):
    for i, day in enumerate(("2026-01-14", "2026-01-15", "2026-01-16")):
        write_parquet(root, DataType.TRADE, MARKET, day, [trade(float(i))], TRADE_COLUMNS)

    trades = bronze.load(MARKET, DataType.TRADE, "2026-01-14", dt.date(2026, 1, 16), root)
    assert [t.get_ts() for t in trades] == [0.0, 1.0, 2.0]


def test_a_missing_day_in_a_range_raises(root):
    """A hole must not read as a quiet market."""
    write_parquet(root, DataType.TRADE, MARKET, "2026-01-14", [trade(1.0)], TRADE_COLUMNS)
    with pytest.raises(bronze.MissingDataError, match="komachi download"):
        bronze.load(MARKET, DataType.TRADE, "2026-01-14", "2026-01-15", root)


def test_dates_ignore_a_partial_download(root):
    write_parquet(root, DataType.TRADE, MARKET, "2026-01-15", [trade(1.0)], TRADE_COLUMNS)
    # A date directory with no data.parquet: interrupted mid-download.
    data_path(root, MARKET.id, DataType.TRADE, "2026-01-16").parent.mkdir(parents=True)
    assert bronze.dates(MARKET, DataType.TRADE, root) == ["2026-01-15"]


def test_markets_are_qate_markets_and_unknown_ones_are_skipped(root):
    write_parquet(root, DataType.TRADE, MARKET, DATE, [trade(1.0)], TRADE_COLUMNS)
    # On disk, readable as a frame, but qate has no ExchangeName for it.
    unknown = root / "bronze" / "dataset=Trade" / "exchange=SOMEVENUE" / "symbol=BTC_SPOT" / f"date={DATE}"
    unknown.mkdir(parents=True)
    (unknown / "data.parquet").write_bytes(b"")

    assert bronze.markets(root) == [MARKET]


def test_an_explicit_root_wins(tmp_path):
    assert bronze.data_root(tmp_path) == tmp_path
