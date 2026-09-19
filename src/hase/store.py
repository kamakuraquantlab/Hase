"""Loading market data into dataframes, and writing derived data back."""

from pathlib import Path

from .layout import data_path, derived_path


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


def write_derived(root: Path, dataset: str, market: str, file_date: str, frame,
                  params: dict[str, str] | None = None) -> Path:
    """Write one day of a derived dataset, atomically.

    Via a temporary file in the same directory and then a rename, because a
    derivation interrupted half way through leaves a parquet file that opens,
    reads, and is wrong. A partial day is worse than a missing one: the missing
    day is obvious and the partial one is not.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = derived_path(root, dataset, market, file_date, params)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.partial")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    tmp.replace(path)
    return path


def read_derived(root: Path, dataset: str, market: str, file_date: str,
                 params: dict[str, str] | None = None, columns=None):
    """Read one day of a derived dataset, or say how to make it.

    Whether the frame comes back wider than it was written depends on the
    reader: pyarrow recovered `dataset`, `exchange`, `symbol`, `date` and any
    parameter partition from the directory names up to 22 and stops at 25, and
    both are allowed by our floor of 15. Read a directory rather than a file to
    get them for certain. Nothing here drops them, and nothing here needs them.
    """
    import pyarrow.parquet as pq

    path = derived_path(root, dataset, market, file_date, params)
    if not path.is_file():
        raise MissingDataError(
            f"No {dataset} for {market} on {file_date} under {root}. Derive it first:  "
            f"hase derive {dataset} --market {market} --start {file_date} --end {file_date}"
        )
    return pq.read_table(path, columns=columns).to_pandas()
