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
SILVER = "silver"
GOLD = "gold"
DATA_TYPES = ("Trade", "OrderBook")

# Where each derived dataset lives, and which bronze dataset it is built from.
# `system/04_hase-analysis-toolkit.md` section 5 is the authority on the split.
DERIVED = {
    "BookState": (SILVER, "OrderBook"),
    "MarketPrice": (SILVER, "OrderBook"),
    "VolSpread": (GOLD, "OrderBook"),
}
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


def derived_path(root: Path, dataset: str, market: str, file_date: str,
                 params: dict[str, str] | None = None) -> Path:
    """Where a derived dataset lands, mirroring the warehouse exactly.

    Derived output sits beside `bronze/` under the same root rather than in a
    tree of its own. That is what lets one DuckDB connection reach raw and
    derived data together, and what keeps a warehouse notebook running
    unchanged against a buyer's copy. Which layer is which already says what
    was purchased: bronze was, nothing else was.

    `params` become partition directories between symbol and date, in the order
    given, so two parameterisations coexist instead of overwriting each other.
    `execution_size` for MarketPrice, `param_id` for VolSpread.
    """
    if dataset not in DERIVED:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {', '.join(sorted(DERIVED))}")
    layer, _ = DERIVED[dataset]
    exchange, symbol = parse_market(market)
    path = root / layer / f"dataset={dataset}" / f"exchange={exchange}" / f"symbol={symbol}"
    for key, value in (params or {}).items():
        path = path / f"{key}={value}"
    return path / f"date={file_date}" / "data.parquet"


def source_of(dataset: str) -> str:
    """The bronze dataset a derivation reads."""
    if dataset not in DERIVED:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {', '.join(sorted(DERIVED))}")
    return DERIVED[dataset][1]


def available_derived_dates(root: Path, dataset: str, market: str,
                            params: dict[str, str] | None = None) -> list[str]:
    base = derived_path(root, dataset, market, "X", params).parent.parent
    if not base.is_dir():
        return []
    return sorted(
        d.name.split("=", 1)[1] for d in base.glob("date=*") if (d / "data.parquet").is_file()
    )
