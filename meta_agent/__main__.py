"""Unified `meta-agent` CLI dispatcher.

One entry point for every workflow:
    meta-agent run-task   — run one task from a benchmark
    meta-agent eval       — evaluate a harness over a full benchmark
    meta-agent loop       — run the optimizer's propose/eval loop
    meta-agent propose    — run the proposer once against the experience store
    meta-agent list       — list candidates
    meta-agent show       — show one candidate's summary
    meta-agent diff       — diff two candidates
    meta-agent failures   — list failed tasks for one candidate
    meta-agent pareto     — show the accuracy/cost Pareto frontier

All workflows go through this entrypoint. The standalone `python -m
meta_agent.loop` is also kept for users who want the loop directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from meta_agent.commands import inspect as cli_mod
from meta_agent.commands import evaluate, propose, run_task
from meta_agent import loop as loop_mod
from meta_agent.utils.logging import configure_logging, get_logger
from meta_agent.core.paths import get_experience_root

logger = get_logger("cli")


def _resolve_experience_dir(
    benchmark_name: Optional[str], explicit_dir: Optional[str]
) -> Path:
    if explicit_dir:
        return Path(explicit_dir)
    if benchmark_name:
        return get_experience_root() / benchmark_name / "candidates"
    return get_experience_root() / "candidates"


# --- Subcommand handlers ---------------------------------------------------

def _handle_run_task(args: argparse.Namespace) -> int:
    try:
        result = run_task.run(
            benchmark_path=args.benchmark,
            config_path=args.config,
            task_name=args.task,
            split=args.split,
            model=args.model,
            keep_workspace=args.keep_workspace,
        )
    except Exception as exc:
        logger.error(f"{type(exc).__name__}: {exc}")
        return 2
    return 0 if result.passed else 1


def _handle_eval(args: argparse.Namespace) -> int:
    scores = evaluate.run(
        benchmark_path=args.benchmark,
        config_path=args.config,
        name=args.name,
        split=args.split,
        model=args.model,
        fast=args.fast,
        tasks=args.tasks,
        concurrency=args.concurrency,
        keep_workspaces=args.keep_workspaces,
        keep_failed=args.keep_failed,
        dry_run=args.dry_run,
    )
    return 0 if scores is not None else 1


def _handle_loop(loop_argv: list[str]) -> int:
    """Parse loop's flags via the loop package's own parser, then call run()."""
    args = loop_mod.parse_args(loop_argv)
    return loop_mod.run(args)


def _handle_propose(args: argparse.Namespace) -> int:
    from meta_agent.services.llm import default_proposer_cli, default_proposer_model

    model = args.model or default_proposer_model()
    proposer_cli = args.proposer_cli or default_proposer_cli()
    ok = propose.propose(
        project=args.project,
        harness=args.harness,
        model=model,
        proposer_cli=proposer_cli,
        apply=args.apply,
    )
    return 0 if ok else 1


def _handle_list(args: argparse.Namespace) -> int:
    cli_mod.list_candidates(_resolve_experience_dir(args.benchmark, args.dir))
    return 0


def _handle_show(args: argparse.Namespace) -> int:
    cli_mod.show_candidate(_resolve_experience_dir(args.benchmark, args.dir), args.name)
    return 0


def _handle_diff(args: argparse.Namespace) -> int:
    cli_mod.diff_candidates(
        _resolve_experience_dir(args.benchmark, args.dir), args.name1, args.name2,
    )
    return 0


def _handle_failures(args: argparse.Namespace) -> int:
    cli_mod.candidate_failures(
        _resolve_experience_dir(args.benchmark, args.dir), args.name,
    )
    return 0


def _handle_pareto(args: argparse.Namespace) -> int:
    cli_mod.pareto_frontier(_resolve_experience_dir(args.benchmark, args.dir))
    return 0


# --- Parser construction ---------------------------------------------------

def _add_inspect_target(p: argparse.ArgumentParser) -> None:
    """Shared flags for list/show/diff/failures/pareto to locate the store."""
    p.add_argument("--benchmark", default=None, help="Benchmark name (maps to experience/<name>/candidates)")
    p.add_argument("--dir", default=None, help="Explicit candidates directory (overrides --benchmark)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="meta-agent",
        description="Harness optimizer — one CLI for every workflow.",
    )
    sub = parser.add_subparsers(dest="subcommand", required=True)

    # run-task
    p = sub.add_parser("run-task", help="Run one task from a benchmark")
    p.add_argument("--benchmark", required=True,
                   help="Benchmark YAML path (or path:split, or dir:split)")
    p.add_argument("--split", default=None,
                   help="Split name when the family YAML defines multiple splits")
    p.add_argument("--config", required=True, help="Harness config directory or harness.py")
    p.add_argument("--task", default=None, help="Task name (default: first task)")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--keep-workspace", action="store_true")
    p.set_defaults(handler=_handle_run_task)

    # eval
    p = sub.add_parser("eval", help="Evaluate a harness over a full benchmark")
    p.add_argument("--benchmark", required=True,
                   help="Benchmark YAML path (or path:split, or dir:split)")
    p.add_argument("--split", default=None,
                   help="Split name when the family YAML defines multiple splits")
    p.add_argument("--config", required=True, help="Harness config directory or harness.py")
    p.add_argument("--name", required=True, help="Candidate name to write to the experience store")
    p.add_argument("--model", default="gpt-5.4")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--tasks", default=None, help="Comma-separated task names")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--keep-workspaces", action="store_true")
    p.add_argument("--keep-failed", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(handler=_handle_eval)

    # loop is intercepted before argparse (see main); this stub only serves
    # `meta-agent -h` / `meta-agent loop -h`.
    sub.add_parser(
        "loop",
        help="Run the optimizer loop (forwards remaining flags to meta_agent.loop)",
        description=(
            "Flags after `loop` are parsed by meta_agent.loop. "
            "Run `python -m meta_agent.loop --help` for the full flag set."
        ),
        add_help=False,
    )

    # propose
    p = sub.add_parser("propose", help="Run the proposer once against the experience store")
    p.add_argument("--project", required=True, help="Benchmark/project name (experience/<project>/)")
    p.add_argument(
        "--harness",
        default="codex",
        choices=[
            "claude_agent_sdk",
            "claude_code",
            "codex",
            "research_single_file",
            "program_harness",
        ],
    )
    p.add_argument("--model", default=None,
                   help="Proposer model (default: $META_AGENT_PROPOSER_MODEL / "
                   "$META_AGENT_MODEL; falls back to gpt-5.4)")
    p.add_argument("--proposer-cli", default=None,
                   choices=["claude", "codex", "inprocess", "api"],
                   help="Proposer backend. 'inprocess'/'api' need no external "
                   "CLI. Default: 'inprocess' for openrouter/anthropic providers, "
                   "else 'codex'.")
    p.add_argument("--apply", action="store_true")
    p.set_defaults(handler=_handle_propose)

    # list
    p = sub.add_parser("list", help="List candidates ranked by reward")
    _add_inspect_target(p)
    p.set_defaults(handler=_handle_list)

    # show
    p = sub.add_parser("show", help="Show one candidate's summary")
    p.add_argument("name", help="Candidate name")
    _add_inspect_target(p)
    p.set_defaults(handler=_handle_show)

    # diff
    p = sub.add_parser("diff", help="Diff two candidates by per-task outcome")
    p.add_argument("name1")
    p.add_argument("name2")
    _add_inspect_target(p)
    p.set_defaults(handler=_handle_diff)

    # failures
    p = sub.add_parser("failures", help="List failed tasks for a candidate")
    p.add_argument("name")
    _add_inspect_target(p)
    p.set_defaults(handler=_handle_failures)

    # pareto
    p = sub.add_parser("pareto", help="Show the accuracy/cost Pareto frontier")
    _add_inspect_target(p)
    p.set_defaults(handler=_handle_pareto)

    return parser


def main() -> None:
    configure_logging()
    argv = sys.argv[1:]
    if argv and argv[0] == "loop":
        sys.exit(_handle_loop(argv[1:]))

    parser = _build_parser()
    args = parser.parse_args(argv)
    rc = args.handler(args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
