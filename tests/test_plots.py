"""Plotting, and the timezone it labels with."""

import argparse
from pathlib import Path

import pytest

from hase.layout import available_dates
from hase.plot import plot_order_book, plot_trades
from hase.store import MissingDataError, load_order_book, load_trades

from conftest import DATE, MARKET


def test_the_root_is_asked_of_komachi_not_worked_out_again(tmp_path, monkeypatch):
    """Hase reads the tree Komachi downloads into, so Komachi is what says
    where it is. A second reader of that file is a second answer waiting to
    disagree with the first."""
    from komachi import settings

    from hase import cli

    env = tmp_path / ".kamakuraquantlab.env"
    env.write_text("ROOT_PATH=/somewhere/data\nTOKEN=hk_secret\n")
    monkeypatch.setattr(settings, "ENV_FILE", env)
    monkeypatch.delenv(settings.ROOT_KEY, raising=False)

    args = argparse.Namespace(root=None)
    assert cli._root(args).as_posix() == "/somewhere/data"
    args.root = "/explicit"
    assert cli._root(args).as_posix() == "/explicit"  # a flag still wins


def test_hase_reads_the_root_and_nothing_else_from_that_file():
    """The file holds Komachi's token. Hase has no use for one and must not
    grow a reason to read it."""
    import hase.layout as layout

    source = Path(layout.__file__).read_text()
    assert "TOKEN" not in source, "layout.py should have no notion of a token"
    assert "ENV_FILE" not in source, "and no notion of the settings file at all"


def test_missing_data_says_how_to_get_it(root):
    with pytest.raises(MissingDataError, match="komachi download"):
        load_trades(root, MARKET, DATE)


def test_timestamps_are_converted_to_tokyo(root, make_data):
    make_data()
    df = load_trades(root, MARKET, DATE)
    # The partition is a Tokyo day, so the first row is Tokyo midnight.
    assert str(df["time"].dt.tz) == "Asia/Tokyo"
    assert df["time"].iloc[0].strftime("%Y-%m-%d %H:%M") == f"{DATE} 00:00"


def test_order_book_reads_only_the_levels_asked_for(root, make_data):
    """81 columns at 230k rows a day; best bid and ask need four."""
    make_data()
    df = load_order_book(root, MARKET, DATE, depth=1)
    assert set(df.columns) == {"ts", "bid0_price", "bid0_qty", "ask0_price", "ask0_qty", "time"}


def test_available_dates_lists_what_is_downloaded(root, make_data):
    make_data(file_date="2025-07-01")
    make_data(file_date="2025-07-02")
    assert available_dates(root, MARKET, "Trade") == ["2025-07-01", "2025-07-02"]
    assert available_dates(root, "GMO:BTC_JPY", "Trade") == []


def test_plot_trades_writes_a_png(root, make_data, tmp_path):
    make_data()
    out = plot_trades(root, MARKET, DATE, tmp_path / "t.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_plot_order_book_writes_a_png(root, make_data, tmp_path):
    make_data()
    out = plot_order_book(root, MARKET, DATE, tmp_path / "b.png")
    assert out.exists() and out.stat().st_size > 5_000


def test_plot_order_book_without_resampling(root, make_data, tmp_path):
    make_data()
    out = plot_order_book(root, MARKET, DATE, tmp_path / "b.png", resample=None)
    assert out.exists()


def test_axes_are_labelled_in_tokyo_not_utc(root, make_data):
    """A chart titled JST with a UTC axis is worse than no timezone at all.

    matplotlib converts datetimes to plain floats and forgets the zone, so the
    formatter renders UTC unless told otherwise. Asserted on the formatter
    rather than on which ticks the locator happens to choose.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates

    from hase.plot import style

    make_data()
    df = load_trades(root, MARKET, DATE)
    first = df["time"].iloc[0]

    fig, (ax,) = style.prepare(rows=1)
    ax.plot(df["time"], df["price"])
    style.finish(ax)
    formatter = ax.xaxis.get_major_formatter()

    # The first row is Tokyo midnight; in UTC the same instant is 15:00.
    assert formatter(mdates.date2num(first)) == "00:00"
    assert isinstance(formatter, mdates.DateFormatter)

    import matplotlib.pyplot as plt

    plt.close(fig)
