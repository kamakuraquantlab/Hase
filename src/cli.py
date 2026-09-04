"""Hase CLI.

Run as: python src/cli.py <command> [options]

Reads the data Komachi downloaded and draws it. Only the plotting half is
implemented so far; derivations, analysis and export come later.
"""

import argparse
import sys
from pathlib import Path

from hase.layout import DEFAULT_ROOT, InvalidMarketError, available_dates, root_path
from hase.plot import plot_order_book, plot_trades
from hase.store import MissingDataError


def _root(args) -> Path:
    return root_path(args.root)


def cmd_dates(args) -> int:
    """What is available locally. Hase never asks the API."""
    root = _root(args)
    print(f"{args.market}   {root}\n")
    for data_type in ("Trade", "OrderBook"):
        dates = available_dates(root, args.market, data_type)
        if dates:
            print(f"{data_type:10} {len(dates):>4} day(s)  {dates[0]} .. {dates[-1]}")
        else:
            print(f"{data_type:10} nothing downloaded")
    return 0


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
  Hase reads the tree Komachi writes. It finds the root from --root, then
  ROOT_PATH or KQL_ROOT_PATH in the environment, then KQL_ROOT_PATH in a .env
  in the current directory, and finally {DEFAULT_ROOT}.

  Hase holds no credential and contacts no service. If a day is missing,
  download it with komachi.

dates
  A date is an Asia/Tokyo day, matching how the data is partitioned.
  Timestamps inside the files are UTC epochs; plots are labelled in JST.
""")
    parser.add_argument("--root", help=f"Data root. Default {DEFAULT_ROOT}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("dates", help="What is downloaded for a market")
    p.add_argument("--market", required=True)
    p.set_defaults(func=cmd_dates)

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
