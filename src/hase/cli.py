"""Hase CLI.

Run as: python src/cli.py <command> [options]

Reads the data Komachi downloaded, derives what the articles need from it,
and draws it.
"""

import argparse
import sys
from pathlib import Path

import komachi.bronze
from komachi import data_root
from komachi.settings import DEFAULT_ROOT, ENV_FILE

from hase.derive import DERIVATIONS, run as run_derivation
from hase.layout import (
    DERIVED,
    InvalidMarketError,
    available_derived_dates,
)
from hase.plot import plot_order_book, plot_trades
from hase.store import MissingDataError


def _root(args) -> Path:
    """The data root, which is Komachi's to answer.

    Hase reads the tree Komachi downloads into, so asking anywhere else would
    be inventing a second answer to a question already settled. `data_root`
    runs first-time setup when nothing has settled it, and stops.
    """
    return data_root(args.root, env=True, setup=True, tool="hase")


def _date_range(start: str, end: str) -> list[str]:
    import datetime as dt

    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    if hi < lo:
        raise ValueError(f"--end {end} is before --start {start}")
    return [(lo + dt.timedelta(days=i)).isoformat() for i in range((hi - lo).days + 1)]


def _wanted_days(args) -> list[str]:
    """The JST days a derive covers, from --days or --end.

    The same two ways of ending a range that Komachi takes, and the same
    refusal to accept both: they say one thing, so one would have to silently
    win. A buyer moving between the two tools should not have to remember
    which verb wants which.
    """
    import datetime as dt

    if args.end and args.days:
        raise SystemExit("Give --days or --end, not both: they say the same thing.")
    if args.end:
        return _date_range(args.start, args.end)
    span = max(args.days or 1, 1)
    last = (dt.date.fromisoformat(args.start) + dt.timedelta(days=span - 1)).isoformat()
    return _date_range(args.start, last)


def cmd_derive(args) -> int:
    """Compute a dataset for a range of days, skipping what is already there."""
    root = _root(args)
    wanted = _wanted_days(args)
    made = skipped = missing = 0
    for file_date in wanted:
        try:
            path, was_made = run_derivation(
                root, args.dataset, args.market, file_date,
                recreate=args.recreate,
                execution_size=getattr(args, "execution_size", None),
                execution_notional=getattr(args, "execution_notional", None),
                step=getattr(args, "step", None),
                lookback=getattr(args, "lookback", None),
            )
        except MissingDataError as exc:
            missing += 1
            print(f"  {file_date}  no source data", file=sys.stderr)
            if missing == 1:
                print(f"    {exc}", file=sys.stderr)
            continue
        if was_made:
            made += 1
            print(f"  {file_date}  {path.stat().st_size / 1e6:.1f} MB")
        else:
            skipped += 1
    print(f"\n{args.dataset} for {args.market}: {made} derived, {skipped} already present, "
          f"{missing} without source data")
    return 1 if made == 0 and skipped == 0 else 0


def _periods(dates: list[str]) -> str:
    """Consecutive dates as spans, so a gap is visible rather than implied.

    Three days in January and a month in June is 33 days in two stretches;
    printing the outer bounds says six months, and that is the line a reader
    uses to decide what still needs downloading or deriving.
    """
    import datetime as dt

    if not dates:
        return ""
    days = sorted({dt.date.fromisoformat(d) for d in dates})
    spans, first, last = [], days[0], days[0]
    for day in days[1:]:
        if (day - last).days == 1:
            last = day
            continue
        spans.append((first, last))
        first = last = day
    spans.append((first, last))
    text = "; ".join(a.isoformat() if a == b else f"{a} .. {b}" for a, b in spans[:3])
    return text if len(spans) <= 3 else f"{text}; +{len(spans) - 3} more"


def cmd_local(args) -> int:
    """What has been downloaded and what has been derived from it.

    Bronze comes from Komachi, which writes that layer and therefore answers
    for it; the derived rows are Hase's own. One command: `dates` reported the
    bronze half and nothing else, which left two ways to ask one question.
    """
    root = _root(args)
    print(f"{args.market}   {root}\n")
    for data_type, held in komachi.bronze.market_state(args.market, root).items():
        state = (f"{held.days:>4} day(s)  {_periods(held.dates)}" if held.days
                 else "   nothing downloaded")
        print(f"bronze  {data_type:12} {state}")
    for dataset in sorted(DERIVED):
        layer = DERIVED[dataset][0]
        for params in _parameterisations(root, dataset, args.market):
            dates = available_derived_dates(root, dataset, args.market, params)
            if not dates:
                continue
            label = " ".join(f"{k}={v}" for k, v in params.items()) or ""
            print(f"{layer:7} {dataset:12} {len(dates):>4} day(s)  {_periods(dates)}  {label}")
    return 0


def _parameterisations(root, dataset, market):
    """Every parameter partition present on disk for this dataset, plus the bare one."""
    from hase.layout import parse_market

    layer = DERIVED[dataset][0]
    exchange, symbol = parse_market(market)
    base = root / layer / f"dataset={dataset}" / f"exchange={exchange}" / f"symbol={symbol}"
    if not base.is_dir():
        return []
    found = [{}]
    for child in sorted(base.iterdir()):
        if child.is_dir() and "=" in child.name and not child.name.startswith("date="):
            key, value = child.name.split("=", 1)
            found.append({key: value})
    return found


def cmd_trades(args) -> int:
    out = plot_trades(_root(args), args.market, args.date, args.output, args.bucket)
    print(f"Wrote {out}")
    return 0


def cmd_book(args) -> int:
    out = plot_order_book(
        _root(args), args.market, args.date, args.output,
        resample=None if args.no_resample else args.resample,
    )
    print(f"Wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hase",
        description="Kamakura Quant Lab analysis toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
data root
  Hase reads the tree Komachi writes, and asks Komachi where it is: --root,
  then ROOT_PATH in the environment, then ROOT_PATH in {ENV_FILE}. With none
  of those it asks once, writes that file and stops; the next run carries on.
  Default {DEFAULT_ROOT}.

  That file is shared with Komachi, so whichever tool is installed first
  settles the root for both. Hase reads only the root from it.

  Hase holds no credential and contacts no service. If a day is missing,
  download it with komachi.

dates
  A date is an Asia/Tokyo day, matching how the data is partitioned.
  Timestamps inside the files are UTC epochs; plots are labelled in JST.
""")
    # Hidden, like Komachi's. It still works and is what makes a second root
    # possible; the settings file is where a buyer changes it, and listing a
    # developer switch at the top of every --help made it the first thing read.
    parser.add_argument("--root", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("derive", help="Compute a silver or gold dataset from bronze")
    p.add_argument("dataset", choices=sorted(DERIVATIONS))
    p.add_argument("--market", required=True)
    p.add_argument("--start", required=True, help="First JST day, YYYY-MM-DD")
    p.add_argument("--days", type=int, help="How many days from --start. Default 1")
    p.add_argument("--end", help="Last JST day, inclusive. Use instead of --days")
    p.add_argument("--recreate", action="store_true", help="Rebuild days that already exist")
    p.add_argument("--execution-size", type=float,
                   help="MarketPrice: trade size in base currency, e.g. 0.002")
    p.add_argument("--execution-notional", type=float,
                   help="MarketPrice: trade size in quote currency, e.g. 1000000")
    p.add_argument("--step", type=int, help="VolSpread: grid interval in seconds. Default 1")
    p.add_argument("--lookback", type=int,
                   help="VolSpread: realized volatility window in seconds. Default 300")
    p.set_defaults(func=cmd_derive)

    p = sub.add_parser("local", help="What is downloaded, and what has been derived")
    p.add_argument("--market", required=True)
    p.set_defaults(func=cmd_local)

    p = sub.add_parser("plot-trades", help="Trade price and buy/sell volume")
    p.add_argument("--market", required=True)
    p.add_argument("--date", required=True, help="JST day, YYYY-MM-DD")
    p.add_argument("--output", required=True, help="PNG path")
    p.add_argument("--bucket", default="5min", help="Volume bucket, e.g. 1min, 5min, 15min")
    p.set_defaults(func=cmd_trades)

    p = sub.add_parser("plot-book", help="Best bid and ask, with the spread")
    p.add_argument("--market", required=True)
    p.add_argument("--date", required=True, help="JST day, YYYY-MM-DD")
    p.add_argument("--output", required=True, help="PNG path")
    p.add_argument("--resample", default="1s", help="Resample interval")
    p.add_argument("--no-resample", action="store_true", help="Plot every snapshot")
    p.set_defaults(func=cmd_book)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (MissingDataError, InvalidMarketError, ValueError) as exc:
        print(f"{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
