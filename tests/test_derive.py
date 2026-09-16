"""The derivations, and the one property that matters most about them.

Hase reimplements what Makalu computes rather than importing it, because
`system/04_hase-analysis-toolkit.md` section 3 forbids a customer tool from
depending on trading code. Reimplementation is only safe if it agrees, so the
last test here checks Hase against Makalu's own stored output when that
warehouse is present, and skips when it is not.
"""

import math
from pathlib import Path

import pytest

from hase.derive import plan, run
from hase.derive.book_state import derive as book_state
from hase.derive.market_price import DEGENERATE_SPREAD_BPS
from hase.derive.market_price import derive as market_price
from hase.derive.vol_spread import DAY, derive as vol_spread, realized_vol_bps
from hase.layout import derived_path
from hase.store import MissingDataError, read_derived

MARKET = "COINCHECK:BTC_SPOT"
DATE = "2025-07-01"
WAREHOUSE = Path("/panda/makalu-data")


# ---- BookState -------------------------------------------------------------

def test_book_state_derives_mid_and_spread(make_data):
    root = make_data()
    df = book_state(root, MARKET, DATE)
    assert len(df) == 240
    row = df.iloc[0]
    assert row["bid"] == 15_000_000.0 and row["ask"] == 15_002_000.0
    assert row["mid"] == 15_001_000.0
    assert row["spread"] == 2_000.0
    assert row["spread_bps"] == pytest.approx(2_000 / 15_001_000 * 1e4)


def test_book_state_drops_a_crossed_book(root, make_data):
    """A crossed book is a snapshot taken mid-update, not a market state.

    Carrying it through as a negative spread is how one bad row becomes a
    negative median.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hase.layout import data_path

    make_data()
    path = data_path(root, MARKET, "OrderBook", DATE)
    # Read the file's own columns only. Reading the path plainly also recovers
    # `dataset`, `exchange`, `symbol` and `date` from the directory names, and
    # writing those back collides with the originals.
    own = pq.ParquetFile(path).schema_arrow.names
    table = pq.read_table(path, columns=own).to_pydict()
    table["ask0_price"][5] = table["bid0_price"][5] - 1.0     # crossed
    table["bid0_price"][6] = 0.0                              # empty
    pq.write_table(pa.Table.from_pydict(table), path)

    df = book_state(root, MARKET, DATE)
    assert len(df) == 238
    assert (df["spread"] >= 0).all()


def test_book_state_reports_the_thinner_side_as_depth():
    """Depth at the touch is limited by whichever side is smaller."""
    # 1.0 at 15,000,000 on the bid and 1.0 at 15,002,000 on the ask: the bid
    # side is worth less, so that is the round-trip constraint.
    assert min(1.0 * 15_000_000, 1.0 * 15_002_000) == 15_000_000


# ---- MarketPrice -----------------------------------------------------------

def test_market_price_walks_deeper_than_the_quote(make_data):
    """The whole point of the dataset: size costs more than the top of book."""
    root = make_data()
    quote = book_state(root, MARKET, DATE)["spread_bps"].median()
    small = market_price(root, MARKET, DATE, execution_size=0.5)["spread_bps"].median()
    large = market_price(root, MARKET, DATE, execution_size=8.0)["spread_bps"].median()
    assert small == pytest.approx(quote)      # inside the first level
    assert large > small                       # walked into the book


def test_market_price_will_not_price_a_size_the_book_cannot_fill(make_data):
    """A partial fill priced as a whole one is a plausible, wrong number."""
    root = make_data()
    df = market_price(root, MARKET, DATE, execution_size=1_000.0)
    assert not df["filled"].any()
    assert df["bid"].isna().all()


def test_market_price_marks_degenerate_snapshots_without_dropping_them(make_data):
    """How many snapshots were collector artefacts is a fact about the archive."""
    root = make_data()
    df = market_price(root, MARKET, DATE, execution_size=0.5, degenerate_spread_bps=0.5)
    assert df["degenerate"].all()
    assert len(df) == 240, "marked, not dropped"
    assert DEGENERATE_SPREAD_BPS == 100.0


def test_size_and_notional_are_different_partitions():
    """They are different units and must never share a directory name."""
    size, _ = plan("MarketPrice", execution_size=0.002)
    notional, _ = plan("MarketPrice", execution_notional=1_000_000)
    assert size == {"execution_size": "0.002"}
    assert notional == {"execution_notional": "1000000"}
    with pytest.raises(ValueError):
        plan("MarketPrice", execution_size=0.002, execution_notional=1_000_000)


# ---- VolSpread -------------------------------------------------------------

def test_vol_spread_fills_a_whole_tokyo_day(make_data):
    root = make_data()
    df = vol_spread(root, MARKET, DATE)
    assert len(df) == DAY
    assert df["sec"].iloc[0] == 0 and df["sec"].iloc[-1] == DAY - 1
    assert df["mid"].notna().sum() == DAY       # first snapshot is at second 0


def test_vol_spread_carries_the_last_price_through_a_quiet_second(make_data):
    """A book with no update has not changed; it is not missing."""
    root = make_data(n=24)                      # one snapshot an hour
    df = vol_spread(root, MARKET, DATE)
    assert df["mid"].notna().sum() == DAY
    assert df["mid"].iloc[0] == df["mid"].iloc[3599]


def test_realized_volatility_needs_enough_returns():
    assert realized_vol_bps([100.0] * 5) is None


def test_realized_volatility_of_a_flat_series_is_zero():
    assert realized_vol_bps([100.0] * 100) == pytest.approx(0.0)


def test_realized_volatility_matches_the_closed_form():
    prices = [100.0 * (1.001 ** i) for i in range(101)]
    expected = math.sqrt(100 * math.log(1.001) ** 2) * 1e4
    assert realized_vol_bps(prices) == pytest.approx(expected)


# ---- running and storing ---------------------------------------------------

def test_run_writes_then_skips(make_data):
    root = make_data()
    path, made = run(root, "MarketPrice", MARKET, DATE)
    assert made and path.is_file()
    assert path == derived_path(root, "MarketPrice", MARKET, DATE, {"execution_size": "0.002"})
    _, again = run(root, "MarketPrice", MARKET, DATE)
    assert not again
    _, forced = run(root, "MarketPrice", MARKET, DATE, recreate=True)
    assert forced


def test_run_leaves_no_partial_file_behind(make_data):
    root = make_data()
    run(root, "VolSpread", MARKET, DATE)
    assert not list(root.rglob("*.partial"))


def test_reading_something_underived_says_how_to_derive_it(root):
    with pytest.raises(MissingDataError) as exc:
        read_derived(root, "VolSpread", MARKET, DATE, {"param_id": "g1s_rv300s"})
    assert "hase derive VolSpread" in str(exc.value)


# ---- agreement with the warehouse ------------------------------------------

@pytest.mark.skipif(not WAREHOUSE.is_dir(), reason="the seller's warehouse is not mounted")
@pytest.mark.parametrize("market,size", [("COINCHECK:BTC_SPOT", 0.002), ("GMO:BTC_JPY", 0.002)])
def test_market_price_agrees_with_makalu(market, size):
    """Hase reimplements Makalu's walk; it has to land in the same place.

    Matched on timestamp rather than position, and not required to be a total
    match. The warehouse file was written at some point in the past and the
    bronze underneath it has been repopulated since, so a handful of snapshot
    timestamps differ between the two -- 2 of 176,339 on GMO, none on
    Coincheck. That is the stored file being older than its source, not the
    walk disagreeing, and it is a good reason to derive rather than to trust a
    silver file someone left lying about.

    On the timestamps both have, agreement is to floating point rather than
    exact: the two accumulate levels in a different order, so the last bit
    moves. 1e-12 is four orders looser than the 1e-16 observed and still far
    tighter than anything that could move a published figure.
    """
    import numpy as np
    import pyarrow.parquet as pq

    exchange, symbol = market.split(":")
    reference = (WAREHOUSE / "silver" / "dataset=MarketPrice" / f"exchange={exchange}"
                 / f"symbol={symbol}" / f"execution_size={size:g}" / f"date={DATE}" / "data.parquet")
    if not reference.is_file():
        pytest.skip(f"no warehouse MarketPrice for {market}")

    want = pq.read_table(reference, columns=["ts", "bid", "ask", "mid"]).to_pandas()
    got = market_price(WAREHOUSE, market, DATE, execution_size=size)
    assert len(got) == len(want), "one snapshot in, one row out"

    # Compared row by row rather than joined on ts: a venue can stamp two
    # snapshots with the same millisecond, so a join on ts is a cross product.
    # The rows are already in snapshot order on both sides.
    drifted = int((want["ts"].to_numpy() != got["ts"].to_numpy()).sum())
    assert drifted < 0.001 * len(want), f"{drifted} timestamps differ; the files are not aligned"

    for column in ("bid", "ask", "mid"):
        a, b = want[column].to_numpy(), got[column].to_numpy()
        usable = np.isfinite(a) & np.isfinite(b)
        relative = np.abs(a[usable] - b[usable]) / np.maximum(np.abs(a[usable]), 1e-12)
        assert relative.max() < 1e-12, f"{column} drifted from the warehouse"
