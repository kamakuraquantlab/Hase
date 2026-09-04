"""Reading the tree Komachi writes.

Duplicated from Komachi rather than imported: the two ship separately and a
buyer may install either alone, so a shared package would be a third thing to
version for the sake of forty lines.

`system/04_hase-analysis-toolkit.md` section 2 sets the rule this module obeys.
Hase reads local files. It reaches no API, holds no credential, and knows
nothing about entitlement.
"""

import os
import re
from pathlib import Path

BRONZE = "bronze"
DATA_TYPES = ("Trade", "OrderBook")
DEFAULT_ROOT = "~/kql-data"
ENV_FILE = Path(".env")

_MARKET_RE = re.compile(r"^[A-Z0-9]+:[A-Z0-9_]+$")


class InvalidMarketError(ValueError):
    pass


def parse_market(market: str) -> tuple[str, str]:
    if not _MARKET_RE.match(market or ""):
        raise InvalidMarketError(f"Market must look like EXCHANGE:SYMBOL, got {market!r}")
    exchange, symbol = market.split(":", 1)
    return exchange, symbol


def _from_env_file(path: Path = ENV_FILE) -> str | None:
    """Read KQL_ROOT_PATH from the .env Komachi writes, if it is there."""
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip() == "KQL_ROOT_PATH":
            return value.strip().strip('"').strip("'")
    return None


def root_path(override: str | None = None) -> Path:
    raw = (
        override
        or os.environ.get("ROOT_PATH")
        or os.environ.get("KQL_ROOT_PATH")
        or _from_env_file()
        or DEFAULT_ROOT
    )
    return Path(raw).expanduser()


def data_path(root: Path, market: str, data_type: str, file_date: str) -> Path:
    exchange, symbol = parse_market(market)
    return (
        root / BRONZE / f"dataset={data_type}" / f"exchange={exchange}"
        / f"symbol={symbol}" / f"date={file_date}" / "data.parquet"
    )


def available_dates(root: Path, market: str, data_type: str) -> list[str]:
    exchange, symbol = parse_market(market)
    base = root / BRONZE / f"dataset={data_type}" / f"exchange={exchange}" / f"symbol={symbol}"
    if not base.is_dir():
        return []
    return sorted(
        d.name.split("=", 1)[1] for d in base.glob("date=*") if (d / "data.parquet").is_file()
    )
