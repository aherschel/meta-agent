"""Logging setup for the meta_agent tree.

Single source of truth for "where do log messages go." CLI entrypoints call
`configure_logging()` once at startup; library code just imports `get_logger()`.

Previously the codebase used `print(f"[LOOP] ...")` and similar bracket-tagged
prints, which produced 80-line walls per run with no way to filter or
timestamp output. This module replaces the tag convention with named
loggers like `meta_agent.loop` / `meta_agent.eval` / `meta_agent.commands.propose`,
so users can silence sections (`--quiet`) or persist them (`--log-file`).

Result tables (e.g. `list_candidates`, per-task eval summaries) still use
print() — those are user-facing output, not diagnostic logs.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

_CONFIGURED = False

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-5s %(name)-20s  %(message)s"
_DEFAULT_DATEFMT = "%H:%M:%S"


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> None:
    """Configure the `meta_agent` logger tree. Idempotent across calls.

    Honors the `META_AGENT_LOG_LEVEL` env var (DEBUG/INFO/WARNING/ERROR) as
    an override, which is useful when the CLI caller hasn't plumbed a flag
    through yet.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    override = os.environ.get("META_AGENT_LOG_LEVEL", "").strip().upper()
    if override and hasattr(logging, override):
        level = getattr(logging, override)

    root = logging.getLogger("meta_agent")
    root.setLevel(level)
    root.propagate = False  # we own our handlers; don't bubble to Python root

    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `meta_agent` tree.

    Pass a short name like "loop" or "eval" and it will be namespaced as
    "meta_agent.loop" etc. Pre-namespaced names are accepted too.
    """
    if not name.startswith("meta_agent"):
        name = f"meta_agent.{name}"
    return logging.getLogger(name)
