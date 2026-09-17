import pytest

MARKET = "COINCHECK:BTC_SPOT"
DATE = "2025-07-01"


@pytest.fixture
def root(tmp_path):
    return tmp_path / "kql-data"


def jst_day(file_date: str, n: int, every: int = 60) -> list[float]:
    """Timestamps across a Tokyo day, as UTC epochs."""
    from datetime import datetime, timedelta, timezone

    jst = timezone(timedelta(hours=9))
    d = datetime.fromisoformat(file_date).replace(tzinfo=jst)
    start = d.timestamp()
    return [start + i * every for i in range(n)]


@pytest.fixture
def make_data(root):
    """Write a day of trades and order book into the tree."""

    def _make(market=MARKET, file_date=DATE, n=240):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from hase.layout import data_path

        ts = jst_day(file_date, n, every=86_400 // n)

        trade = data_path(root, market, "Trade", file_date)
        trade.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pydict({
            "ts": ts,
            "side": [i % 2 for i in range(n)],
            "price": [15_000_000.0 + i for i in range(n)],
            "size": [0.1] * n,
        }), trade)

        book = data_path(root, market, "OrderBook", file_date)
        book.parent.mkdir(parents=True, exist_ok=True)
        cols = {"ts": ts}
        for i in range(20):
            cols[f"bid{i}_price"] = [15_000_000.0 - i * 100 + j for j in range(n)]
            cols[f"bid{i}_qty"] = [1.0] * n
            cols[f"ask{i}_price"] = [15_002_000.0 + i * 100 + j for j in range(n)]
            cols[f"ask{i}_qty"] = [1.0] * n
        pq.write_table(pa.Table.from_pydict(cols), book)
        return root

    return _make


@pytest.fixture(autouse=True)
def _never_touch_the_real_home(tmp_path, monkeypatch):
    """Keep the settings file and the default root inside the test's own
    directory. Without this, a test that resolves a root with nothing
    configured runs setup against the developer's home and creates a data
    directory there."""
    import hase.layout as layout

    monkeypatch.setattr(layout, "ENV_FILE", tmp_path / "settings.env")
    monkeypatch.setattr(layout, "DEFAULT_ROOT", str(tmp_path / "default-data"))
    monkeypatch.setenv(layout.ROOT_KEY, str(tmp_path / "data"))
