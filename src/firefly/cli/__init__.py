"""Firefly CLI — flat command surface, organized into command modules."""

from firefly.cli import drill, parity, quant  # noqa: F401  (register commands on `app`)
from firefly.cli._app import app

__all__ = ["app"]
