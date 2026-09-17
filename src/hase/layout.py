"""Reading the tree Komachi writes.

Duplicated from Komachi rather than imported: the two ship separately and a
buyer may install either alone, so a shared package would be a third thing to
version for the sake of forty lines.

The rule this module obeys: Hase reads local files. It reaches no API, holds no
credential, and knows nothing about entitlement.
"""

import os
import re
from pathlib import Path

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
DEFAULT_ROOT = "~/kamakuraquantlab-data"
ENV_FILE = Path("~/.kamakuraquantlab.env").expanduser()
ROOT_KEY = "ROOT_PATH"

_MARKET_RE = re.compile(r"^[A-Z0-9]+:[A-Z0-9_]+$")


class InvalidMarketError(ValueError):
    pass


def parse_market(market: str) -> tuple[str, str]:
    if not _MARKET_RE.match(market or ""):
        raise InvalidMarketError(f"Market must look like EXCHANGE:SYMBOL, got {market!r}")
    exchange, symbol = market.split(":", 1)
    return exchange, symbol


def _from_env_file(path: Path | None = None) -> str | None:
    """The data root from the shared settings file, if it is there.

    Only the root. The file also holds Komachi's purchase token, and Hase has
    no use for one: it reads local files and reaches no service. Parsing past
    the key it needs would make that harder to claim and easier to break.
    """
    # Resolved at call time, not bound as a default: a default freezes the
    # module attribute at import, which makes the path impossible to redirect
    # and the behaviour impossible to test.
    path = path or ENV_FILE
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.strip().partition("=")
        if key.strip() == ROOT_KEY:
            return value.strip().strip('"').strip("'")
    return None


def configured_root(override: str | None = None) -> str | None:
    """The root as settled by flag, environment or file. None if unset."""
    return override or os.environ.get(ROOT_KEY) or _from_env_file()


def root_path(override: str | None = None) -> Path:
    return Path(configured_root(override) or DEFAULT_ROOT).expanduser()


class SetupRequired(SystemExit):
    """Raised, and printed, when the tool has not been set up yet."""


def run_setup(path: Path | None = None) -> None:
    """Ask for a data root, write the settings file, show it, and stop.

    The same file and the same question Komachi asks, so whichever tool a
    reader installs first settles it for both. Stopping afterwards is
    deliberate: setup is a different act from the command that triggered it,
    and the next run starts from a settled state.

    The file written here names only the root, which is why the whole of it can
    be printed. Komachi's `token set` is what ever puts a credential in it, and
    Hase has no reason to read that key.
    """
    import sys

    path = path or ENV_FILE
    print("hase keeps data under one root directory, shared with Komachi.")
    print(f"Leave blank for {DEFAULT_ROOT}.\n")
    answer = ""
    if sys.stdin.isatty():
        try:
            answer = input("Data root: ").strip()
        except EOFError:
            answer = ""
    resolved = Path(answer or DEFAULT_ROOT).expanduser()
    resolved.mkdir(parents=True, exist_ok=True)

    values = {ROOT_KEY: str(resolved)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Kamakura Quant Lab settings. Read and edit freely.\n"
        "# Komachi and Hase both use this file.\n\n"
        + "".join(f"{k}={v}\n" for k, v in values.items())
    )
    path.chmod(0o600)

    print(f"\nWrote {path}\n")
    print(path.read_text().rstrip())
    print("\nSetup done. Run hase again.")
    raise SetupRequired(0)


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
