"""Tiny shell-command wrapper.

Lives in its own module so `__init__.py` and `research.py` can both import
it without a circular dependency (importing `run_command` from the package
root while the package is still loading would cycle).
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Union


def run_command(
    cmd: Union[str, List[str]], cwd: Path, timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    """Run a verify or setup command. Accepts str (shell) or list (argv)."""
    if isinstance(cmd, list):
        return subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    return subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
