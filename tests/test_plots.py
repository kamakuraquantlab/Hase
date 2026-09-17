"""Plotting, and the timezone it labels with."""

from pathlib import Path

import pytest

from hase.layout import available_dates, root_path
from hase.plot import plot_order_book, plot_trades
from hase.store import MissingDataError, load_order_book, load_trades

from conftest import DATE, MARKET


def test_root_comes_from_the_settings_file_komachi_writes(tmp_path, monkeypatch):
    """One file in the home directory, shared by both tools."""
    import hase.layout as layout

    monkeypatch.delenv("ROOT_PATH", raising=False)
    env = tmp_path / ".kamakuraquantlab.env"
    env.write_text("ROOT_PATH=/somewhere/data\nTOKEN=hk_secret\n")
    monkeypatch.setattr(layout, "ENV_FILE", env)

    assert root_path(None).as_posix() == "/somewhere/data"
    assert root_path("/explicit").as_posix() == "/explicit"  # a flag still wins


def test_hase_reads_the_root_and_nothing_else_from_that_file(tmp_path, monkeypatch):
    """The file holds Komachi's token. Hase has no use for one and must not
    grow a reason to read it."""
    import hase.layout as layout

    monkeypatch.delenv("ROOT_PATH", raising=False)
    env = tmp_path / ".kamakuraquantlab.env"
    env.write_text("ROOT_PATH=/somewhere/data\nTOKEN=hk_secret\n")
    monkeypatch.setattr(layout, "ENV_FILE", env)

    assert layout._from_env_file(env) == "/somewhere/data"
    source = (Path(layout.__file__)).read_text()
    assert "TOKEN" not in source, "layout.py should have no notion of a token"


def test_an_unset_root_is_reported_as_unset(tmp_path, monkeypatch):
    """`root_path` falls back to the default; `configured_root` says there was
    nothing to fall back from, which is what triggers setup."""
    import hase.layout as layout

    monkeypatch.delenv("ROOT_PATH", raising=False)
    monkeypatch.setattr(layout, "ENV_FILE", tmp_path / "absent.env")
    assert layout.configured_root() is None
    assert root_path(None) == Path(layout.DEFAULT_ROOT).expanduser()


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
