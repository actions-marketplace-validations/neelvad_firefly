"""The shared Typer app + cross-command helpers.

Command implementations live in sibling modules (parity / quant / drill) and
register onto this one flat ``app`` — the public surface stays ``firefly <cmd>``.
"""

from __future__ import annotations

from pathlib import Path

import typer

from firefly.storage import publish_reference, resolve_reference

app = typer.Typer(
    name="firefly",
    help="Diagnose, quantize, and ship a faster servable model (firefly optimize) "
    "— on a numerical-divergence attribution engine that also runs as a parity CI gate.",
    no_args_is_help=True,
)


def _parse_runner_opts(opts: list[str]) -> dict[str, str]:
    """Parse repeated ``--runner-opt key=value`` flags into a dict."""
    parsed: dict[str, str] = {}
    for item in opts:
        if "=" not in item:
            raise typer.BadParameter(
                f"--runner-opt must be key=value, got {item!r}",
                param_hint="--runner-opt",
            )
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _resolve_or_exit(reference: str) -> Path:
    """Resolve a reference URI to a local path, exiting cleanly on errors."""
    try:
        return resolve_reference(reference)
    except NotImplementedError as e:
        raise typer.BadParameter(str(e), param_hint="--reference") from e
    except (RuntimeError, ValueError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


def _publish_or_exit(local_path: Path, uri: str, *, commit_message: str) -> None:
    """Publish a reference dir to a URI, exiting cleanly on errors."""
    try:
        publish_reference(local_path, uri, commit_message=commit_message)
    except NotImplementedError as e:
        raise typer.BadParameter(str(e), param_hint="--to") from e
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(code=2) from e


