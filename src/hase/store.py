"""Loading market data into dataframes."""

from pathlib import Path

from .layout import data_path


class MissingDataError(FileNotFoundError):
    pass


def _read(root: Path, market: str, data_type: str, file_date: str, columns=None):
    import pandas as pd
    import pyarrow.parquet as pq

    path = data_path(root, market, data_type, file_date)
    if not path.is_file():
        raise MissingDataError(
            f"No {data_type} for {market} on {file_date} under {root}. "
            f"Download it first:  komachi download --market {market} "
            f"--start {file_date} --end {file_date}"
        )
    df = pq.read_table(path, columns=columns).to_pandas()
    # ts is a UTC epoch; the partition date is a Tokyo day. Converting here
    # means every plot is labelled in the timezone the day was cut in.
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.tz_convert("Asia/Tokyo")
    return df


def load_trades(root: Path, market: str, file_date: str):
    """Trades, with `side` as the aggressor: 0 buy, 1 sell."""
    return _read(root, market, "Trade", file_date)


def load_order_book(root: Path, market: str, file_date: str, depth: int = 1):
    """Order book, reading only the levels asked for.

    A day is 230,000 rows across 81 columns and around 35 MB. Best bid and ask
    need four of those columns, so reading them all would cost roughly twenty
    times the memory for nothing.
    """
    columns = ["ts"]
    for i in range(depth):
        columns += [f"bid{i}_price", f"bid{i}_qty", f"ask{i}_price", f"ask{i}_qty"]
    return _read(root, market, "OrderBook", file_date, columns=columns)
