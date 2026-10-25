"""The derivations, and the one property that matters most about them.

Hase reimplements what the seller's own pipeline computes rather than
importing it: a customer tool must not depend on trading code.
Reimplementation is only safe if it agrees, so the last test here checks Hase
against that pipeline's stored output when the warehouse is mounted, and skips
when it is not -- which is everywhere but the seller's own machine.

Point `HASE_WAREHOUSE` at it to run those; without it they skip.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

from hase.dataset import trade_count
from hase.derive import plan, run, spread, volatility
from hase.derive.book_state import derive as book_state
from hase.derive.grid import DAY, quote_grid
from hase.derive.market_price import DEGENERATE_SPREAD_BPS, dropped
from hase.derive.market_price import derive as market_price
from hase.derive.volatility import realized_vol_bps
from hase.layout import derived_path
from hase.store import MissingDataError, derived_attrs, read_derived

MARKET = "COINCHECK:BTC_SPOT"
DATE = "2025-07-01"
# Read from the environment rather than written here: the path is the seller's
# own mount and means nothing to anyone else reading this package.
WAREHOUSE = Path(os.environ.get("HASE_WAREHOUSE", "/nonexistent"))


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
    table["ask0_price"][5] = table["bid0_price"][5] - 1.0  # crossed
    table["bid0_price"][6] = 0.0  # empty
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
    assert small == pytest.approx(quote)  # inside the first level
    assert large > small  # walked into the book


def test_market_price_will_not_price_a_size_the_book_cannot_fill(make_data):
    """A partial fill priced as a whole one is a plausible, wrong number."""
    root = make_data()
    df = market_price(root, MARKET, DATE, execution_size=1_000.0)
    assert df.empty
    assert dropped(df) == {"snapshots": 240, "unfilled": 240, "degenerate": 0}


def test_market_price_needs_the_whole_size_not_most_of_it(root, make_data):
    """A book one part in a thousand short of the size is still short."""
    import pyarrow.parquet as pq

    from hase.layout import data_path

    make_data()
    book = pq.read_table(data_path(root, MARKET, "OrderBook", DATE)).to_pandas()
    depth = min(
        book[[f"bid{i}_qty" for i in range(20)]].sum(axis=1).min(),
        book[[f"ask{i}_qty" for i in range(20)]].sum(axis=1).min(),
    )
    assert len(market_price(root, MARKET, DATE, execution_size=depth)) == 240
    assert market_price(root, MARKET, DATE, execution_size=depth * 1.0005).empty


def test_market_price_drops_degenerate_snapshots_and_counts_them(make_data):
    """How many snapshots were collector artefacts is a fact about the archive."""
    root = make_data()
    df = market_price(root, MARKET, DATE, execution_size=0.5, degenerate_spread_bps=0.5)
    assert df.empty
    assert dropped(df) == {"snapshots": 240, "unfilled": 0, "degenerate": 240}
    assert DEGENERATE_SPREAD_BPS == 100.0
    assert list(market_price(root, MARKET, DATE, execution_size=0.5).columns) == [
        "ts",
        "bid",
        "ask",
        "mid",
        "spread",
        "spread_bps",
    ]


def test_the_count_of_dropped_snapshots_is_written_with_the_day(make_data):
    root = make_data()
    path, _ = run(root, "MarketPrice", MARKET, DATE, execution_size=1_000.0)
    assert derived_attrs(path)["dropped"]["unfilled"] == 240


def test_a_notional_fills_from_a_book_that_is_padded_with_empty_levels(root, make_data):
    """An exchange pads the far end of the book, and that must not matter.

    Walking to a cash amount converts notional to base units by dividing by the
    price, and a padded level has no price. Dividing by NaN there poisoned the
    whole row even though nothing was taken from that level, so a book whose
    first level covered the amount several times over reported that it could
    not fill at all. Measured on a real day it marked 95% of books unfillable.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hase.layout import data_path

    make_data()
    path = data_path(root, MARKET, "OrderBook", DATE)
    own = pq.ParquetFile(path).schema_arrow.names
    table = pq.read_table(path, columns=own).to_pydict()
    for level in range(3, 20):  # only three real levels
        n = len(table["ts"])
        for side in ("bid", "ask"):
            table[f"{side}{level}_price"] = [0.0] * n
            table[f"{side}{level}_qty"] = [0.0] * n
    pq.write_table(pa.Table.from_pydict(table), path)

    # One level holds 1.0 at about 15,000,000, so 1,000,000 yen is a fraction
    # of the first level and must fill from it alone.
    df = market_price(root, MARKET, DATE, execution_notional=1_000_000)
    assert len(df) == 240, "a padded book refused an amount its first level covers"
    assert (df["spread_bps"] > 0).all()


def test_size_and_notional_are_different_partitions():
    """They are different units and must never share a directory name."""
    size, _ = plan("MarketPrice", execution_size=0.002)
    notional, _ = plan("MarketPrice", execution_notional=1_000_000)
    assert size == {"execution_size": "0.002"}
    assert notional == {"execution_notional": "1000000"}
    with pytest.raises(ValueError):
        plan("MarketPrice", execution_size=0.002, execution_notional=1_000_000)


# ---- the quote grid --------------------------------------------------------


def test_the_grid_fills_a_whole_tokyo_day(make_data):
    root = make_data()
    df = quote_grid(root, MARKET, DATE)
    assert len(df) == DAY
    assert df["mid"].notna().sum() == DAY  # first snapshot is at second 0


def test_the_grid_carries_the_last_price_through_a_quiet_second(make_data):
    """A book with no update has not changed; it is not missing."""
    root = make_data(n=24)  # one snapshot an hour
    df = quote_grid(root, MARKET, DATE)
    assert df["mid"].notna().sum() == DAY
    assert df["mid"].iloc[0] == df["mid"].iloc[3599]


def _two_days(make_data, n=1440):
    """The day before DATE and DATE, one snapshot a minute, a price that moves."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from hase.layout import data_path

    root = None
    for file_date in ("2025-06-30", DATE):
        root = make_data(file_date=file_date, n=n)
        book = data_path(root, MARKET, "OrderBook", file_date)
        table = pq.ParquetFile(book).read().to_pandas()  # the file alone, not its partitions
        wave = 1 + 0.002 * np.sin(np.arange(n) / 37.0) + (0.001 if file_date == DATE else 0.0)
        for column in table.columns:
            if column.endswith("_price"):
                table[column] = table[column] * wave
        pq.write_table(pa.Table.from_pandas(table, preserve_index=False), book)
    return root


def test_the_grid_walks_the_book_when_given_a_size(make_data):
    root = _two_days(make_data)
    touch = quote_grid(root, MARKET, DATE)
    walked = quote_grid(root, MARKET, DATE, execution_size=1.5)  # deeper than the first level
    assert (walked["spread_bps"] > touch["spread_bps"]).all()


# ---- Spread and Volatility -------------------------------------------------


def test_spread_and_volatility_partition_by_source_then_windows():
    assert plan("Spread")[0] == {"execution_size": "touch", "param_id": "g1s_ema300s"}
    assert plan("Spread", execution_size=0.002, span=900)[0] == {"execution_size": "0.002", "param_id": "g1s_ema900s"}
    walked, kwargs = plan("Volatility", execution_size=0.002, horizon=180, lookback=900, smooth=300)
    assert walked == {"execution_size": "0.002", "param_id": "g1s_h180s_lb900s_ema300s"}
    assert kwargs["execution_size"] == 0.002


def test_each_dataset_holds_only_what_its_windows_make(make_data):
    root = _two_days(make_data)
    assert list(spread.derive(root, MARKET, DATE).columns) == ["ts", "spread_bps_ema"]
    assert list(volatility.derive(root, MARKET, DATE).columns) == ["ts", "rv_bps", "vol_bps", "vol_bps_ema"]


def test_volatility_is_warmed_up_by_the_evening_before(make_data):
    """00:00 JST has a full window behind it when the day before is on disk."""
    warm = volatility.derive(_two_days(make_data), MARKET, DATE)
    assert np.isfinite(warm.loc[0, ["rv_bps", "vol_bps", "vol_bps_ema"]].astype(float)).all()


def test_volatility_without_the_day_before_starts_short_and_says_so(make_data):
    cold = volatility.derive(make_data(n=1440), MARKET, DATE)
    assert cold["rv_bps"].iloc[:300].isna().all() and cold["rv_bps"].iloc[300:].notna().all()


def test_spread_and_volatility_agree_with_the_pipeline_they_replace(make_data):
    """The warehouse pipeline's VolSpread formulas, written out in pandas, over the same quotes.

    That pipeline resampled bid and ask to the last value in each second,
    carried them forward, and measured on 8 hours before the day and the day.
    Against its stored files on the seller's warehouse, Hase agreed to 2e-9
    relative on every column (GMO BTC_JPY, XRP_JPY; four parameterisations).
    Its columns were fractions; Hase's are bps.
    """
    import pandas as pd

    root = _two_days(make_data)
    vol = volatility.derive(root, MARKET, DATE, execution_size=0.5, horizon=60, lookback=300, smooth=120)
    spr = spread.derive(root, MARKET, DATE, execution_size=0.5, span=300)

    quotes = pd.concat([market_price(root, MARKET, d, execution_size=0.5) for d in ("2025-06-30", DATE)]).set_index(
        "ts"
    )
    quotes.index = pd.to_datetime(quotes.index, unit="s")
    start = pd.Timestamp(vol["ts"].iloc[0], unit="s")
    quotes = quotes[quotes.index >= start - pd.Timedelta(hours=8)]
    bid = quotes["bid"].resample("1s").last().ffill()
    ask = quotes["ask"].resample("1s").last().ffill()
    mid = (bid + ask) / 2
    volatility_raw = mid.pct_change(periods=60).rolling(window=300, min_periods=1).std()
    want = pd.DataFrame(
        {
            "vol_bps": volatility_raw * 1e4,
            "vol_bps_ema": volatility_raw.ewm(span=120, adjust=False).mean() * 1e4,
            "spread_bps_ema": ((ask - bid) / mid).ewm(span=300, adjust=False).mean() * 1e4,
        }
    )
    want = want[want.index >= start]
    got = vol.merge(spr, on="ts")
    got = got.set_index(pd.to_datetime(got["ts"], unit="s")).loc[want.index]
    for column in want.columns:
        a, b = want[column].to_numpy(), got[column].to_numpy()
        usable = np.isfinite(a)
        assert np.isfinite(b[usable]).all(), column
        assert np.allclose(a[usable], b[usable], rtol=1e-9, atol=0), column


def test_a_steady_trend_has_no_volatility_and_a_constant_spread_its_own_ema():
    mid = 100 * 1.0001 ** np.arange(2000)
    out = volatility.measure(mid, horizon=60, lookback=300, smooth=120)
    assert np.nanmax(out["vol_bps"][400:]) < 1e-8  # every 60 s return is the same
    assert np.allclose(spread.measure(np.full(2000, 2.0), span=300), 2.0)


def test_windows_are_counted_in_grid_points_not_seconds():
    """On a 5 s grid, a 300 s lookback is 60 points: 60 returns, so from the 61st price."""
    rng = np.random.default_rng(0)
    mid = 100 * np.exp(np.cumsum(rng.normal(0, 1e-4, 400)))
    coarse = volatility.measure(mid, step=5, horizon=5, lookback=300, smooth=120)
    assert np.isnan(coarse["rv_bps"][59]) and np.isfinite(coarse["rv_bps"][60])


def test_realized_volatility_needs_enough_returns():
    assert realized_vol_bps([100.0] * 5) is None


def test_realized_volatility_of_a_flat_series_is_zero():
    assert realized_vol_bps([100.0] * 100) == pytest.approx(0.0)


def test_realized_volatility_matches_the_closed_form():
    prices = [100.0 * (1.001**i) for i in range(101)]
    expected = math.sqrt(100 * math.log(1.001) ** 2) * 1e4
    assert realized_vol_bps(prices) == pytest.approx(expected)


def test_cli_takes_several_windows_at_once():
    from hase.cli import _param_sets, build_parser

    def sets(*argv):
        return _param_sets(build_parser().parse_args(["derive", *argv, "--market", MARKET, "--start", DATE]))

    assert sets("Volatility", "--params", "60:300:120", "--params", "300:900:300") == [
        {"horizon": 60, "lookback": 300, "smooth": 120},
        {"horizon": 300, "lookback": 900, "smooth": 300},
    ]
    assert sets("Spread", "--params", "300", "--params", "900") == [{"span": 300}, {"span": 900}]
    with pytest.raises(ValueError):
        sets("Spread", "--params", "60:300:120")


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
    run(root, "Volatility", MARKET, DATE)
    assert not list(root.rglob("*.partial"))


def test_reading_something_underived_says_how_to_derive_it(root):
    with pytest.raises(MissingDataError) as exc:
        read_derived(root, "Spread", MARKET, DATE, {"execution_size": "touch", "param_id": "g1s_ema300s"})
    assert "hase derive Spread" in str(exc.value)


# ---- agreement with the warehouse ------------------------------------------


@pytest.mark.skipif(not WAREHOUSE.is_dir(), reason="the seller's warehouse is not mounted")
@pytest.mark.parametrize("market,size", [("COINCHECK:BTC_SPOT", 0.002), ("GMO:BTC_JPY", 0.002)])
def test_market_price_agrees_with_the_warehouse(market, size):
    """Hase reimplements the same walk; it has to land in the same place.

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
    reference = (
        WAREHOUSE
        / "silver"
        / "dataset=MarketPrice"
        / f"exchange={exchange}"
        / f"symbol={symbol}"
        / f"execution_size={size:g}"
        / f"date={DATE}"
        / "data.parquet"
    )
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


# ---- what an analysis is handed ---------------------------------------------


def test_trade_count_reads_the_footer_and_admits_a_missing_day(root, make_data):
    make_data(n=240)
    assert trade_count(root, MARKET, DATE) == 240
    assert trade_count(root, MARKET, "1999-01-01") is None
