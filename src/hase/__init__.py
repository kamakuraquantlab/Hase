"""Hase: the Kamakura Quant Lab analysis toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("kamakuraquantlab-hase")
except PackageNotFoundError:        # a source tree that was never installed
    __version__ = "0.0.0+source"
