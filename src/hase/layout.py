"""Reading the tree Komachi writes.

Paths only. Where the tree *is* was duplicated here once, and two readers of
one settings file is one reader too many: `komachi.data_root` is now the only
answer, and the CLI is where Hase asks it. Every function here takes the root
it was given.

The rule this module obeys: Hase reads local files. It reaches no API, holds no
credential, and knows nothing about entitlement.
"""

import re
from pathlib import Path

import komachi.bronze

BRONZE = "bronze"
SILVER = "silver"
GOLD = "gold"
DATA_TYPES = ("Trade", "OrderBook")

# Where each derived dataset lives, and which bronze dataset it is built from.
# Silver is point-in-time and derived per market; gold is windowed and carries
# its parameters in the path.
DERIVED = {
    "BookState": (SILVER, "OrderBook"),
    "MarketPrice": (SILVER, "OrderBook"),
    "VolSpread": (GOLD, "OrderBook"),
}
_MARKET_RE = re.compile(r"^[A-Z0-9]+:[A-Z0-9_]+$")


class InvalidMarketError(ValueError):
    pass


def parse_market(market: str) -> tuple[str, str]:
    if not _MARKET_RE.match(market or ""):
        raise InvalidMarketError(f"Market must look like EXCHANGE:SYMBOL, got {market!r}")
    exchange, symbol = market.split(":", 1)
    return exchange, symbol






def data_path(root: Path, market: str, data_type: str, file_date: str) -> Path:
    exchange, symbol = parse_market(market)
    return (
        root / BRONZE / f"dataset={data_type}" / f"exchange={exchange}"
        / f"symbol={symbol}" / f"date={file_date}" / "data.parquet"
    )


def available_dates(root: Path, market: str, data_type: str) -> list[str]:
    """Which bronze dates are on disk, asked of Komachi rather than the tree.

    Komachi writes bronze, so Komachi says what is in it. Hase walked the
    directories itself, which meant two implementations of one question: a
    change to the layout, to the partition names, or to what counts as a
    complete day had to land in both, and the day they disagreed would be the
    day a derivation quietly skipped a date the downloader believed it had.

    Komachi is a dependency of this package, so it is always there to ask.
    """
    return komachi.bronze.dates(market, data_type, root)

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
