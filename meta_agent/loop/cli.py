"""Loop CLI: argparse + the typed `run(args)` body the unified CLI calls.

Both `python -m meta_agent.loop` (via `main()`) and `meta-agent loop ...`
(via `__main__._handle_loop`) share the same parser and `run(args)` entry,
so flags and behavior stay in one place.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from meta_agent.utils.logging import configure_logging
from meta_agent.loop.proposer import DEFAULT_PROPOSER_MAX_TURNS

from meta_agent.loop.epoch import (
    _maybe_evolve_skill,
    _maybe_run_baseline,
    _run_one_epoch,
    _seed_history_from_baseline_on_disk,
)
from meta_agent.loop.final_eval import run_final_eval
from meta_agent.loop.proposer import SURFACE_LOCK_CHOICES
from meta_agent.loop.state import (
    _benchmark_candidates_per_iter_default,
    _prepare_loop_state,
    _print_run_header,
    release_run_lock,
)
from meta_agent.core.paths import get_workspace_root


def build_arg_parser() -> argparse.ArgumentParser:
    """Return the `meta-agent loop` argument parser."""
    parser = argparse.ArgumentParser(description="Run the harness optimization loop")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark YAML path (or path:split, or dir:split)")
    parser.add_argument("--split", default=None,
                        help="Search-split name when the family YAML has multiple splits")
    parser.add_argument("--iterations", type=int, default=5, help="Number of evolution iterations")
    parser.add_argument("--model", default=None,
                        help="Model for evaluation (default: $META_AGENT_MODEL, "
                        "or the provider default; falls back to gpt-5.4)")
    parser.add_argument("--fast", action="store_true", help="Use benchmark's fast_tasks subset")
    parser.add_argument("--concurrency", type=int, default=4, help="Parallel task count")
    parser.add_argument("--start-from", type=int, default=1, help="Starting iteration number (for resuming)")
    parser.add_argument("--proposer-model", default=None,
                        help="Model for the proposer agent (default: "
                        "$META_AGENT_PROPOSER_MODEL / $META_AGENT_MODEL, or the "
                        "eval model; falls back to gpt-5.4)")
    parser.add_argument(
        "--proposer-max-turns",
        type=int,
        default=DEFAULT_PROPOSER_MAX_TURNS,
        help=f"Maximum turns for each proposer session (default: {DEFAULT_PROPOSER_MAX_TURNS})",
    )
    parser.add_argument(
        "--max-proposer-failures",
        type=int,
        default=5,
        help=(
            "Maximum consecutive proposer/infrastructure failures before "
            "aborting the run. These failures do not consume evolution epochs."
        ),
    )
    parser.add_argument(
        "--baseline",
        default=None,
        nargs="?",
        help="Run a baseline harness before the loop (pass a harness path explicitly)",
    )
    parser.add_argument("--run-name", required=True,
                        help="Experience-store dir name for this run. REQUIRED. "
                        "Identifies the run independently of the benchmark, so you "
                        "can rerun the same family:split with different flags "
                        "(e.g. --candidates-per-iter 3) without colliding.")
    parser.add_argument("--fresh", action="store_true",
                        help="Wipe experience/<run-name>/ and its derived holdout "
                        "dir at startup, then start over from scratch. 3-second "
                        "countdown before deletion so you can Ctrl-C out.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a prior run that stopped with history on "
                        "disk. Also bypasses the run lockfile's live-PID refusal "
                        "when you're genuinely taking over from a stale writer.")
    parser.add_argument(
        "--resume-from-proposal",
        action="store_true",
        help=(
            "For k>1 runs, reuse the already-proposed candidates for the first "
            "resumed epoch instead of invoking the proposer again. Requires "
            "--resume and an on-disk proposal checkpoint for --start-from."
        ),
    )
    parser.add_argument("--evolve-skill", action="store_true",
                        help="Enable skill co-evolution (meta-proposer rewrites the active target skill periodically)")
    parser.add_argument("--skill-evolve-every", type=int, default=5,
                        help="Run skill evolution every N iterations (requires --evolve-skill)")
    parser.add_argument("--holdout", default=None, dest="holdout_benchmark",
                        help="Held-out benchmark ref for per-epoch validation "
                        "(path, path:split, or dir:split)")
    parser.add_argument("--holdout-split", default=None,
                        help="Split name for the holdout family YAML (if it defines multiple)")
    parser.add_argument(
        "--accept-on-holdout",
        action="store_true",
        help=(
            "Gate 'new best' acceptance on the holdout benchmark's reward "
            "instead of the search benchmark's reward. Requires "
            "--holdout. Matches paper 1's 'keep only if it improves "
            "holdout accuracy' policy and prevents search-set overfitting."
        ),
    )
    parser.add_argument("--proposer-cli", default=None,
                        choices=["claude", "codex", "inprocess", "api"],
                        help="Proposer backend. 'inprocess'/'api' drive the "
                        "configured LLM provider directly (no external CLI). "
                        "Default: 'inprocess' when META_AGENT_LLM_PROVIDER is "
                        "openrouter/anthropic, else 'codex'.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Number of search tasks per epoch (samples from full pool)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for task batching (reproducible shuffling)")
    parser.add_argument(
        "--surface-lock",
        default=None,
        choices=list(SURFACE_LOCK_CHOICES),
        help=(
            "Restrict proposer edits to one codex harness surface "
            "(agents, hooks, config, skills, subagents)."
        ),
    )
    parser.add_argument(
        "--candidates-per-iter",
        type=int,
        default=None,
        help=(
            "Candidates produced per proposer call. Defaults to benchmark "
            "optimizer.candidates_per_iter if set, else 1. When >1, the "
            "proposer writes each candidate to staging/<name>/harness.py and "
            "the loop evaluates all of them on the same task batch."
        ),
    )
    parser.add_argument(
        "--final-test",
        default=None,
        dest="final_test_benchmark",
        help=(
            "Benchmark ref for a post-search final evaluation. The proposer "
            "does not see these results during search."
        ),
    )
    parser.add_argument("--final-test-split", default=None)
    parser.add_argument(
        "--final-test-frontier",
        action="store_true",
        help="Evaluate Pareto/frontier candidates in the final test phase.",
    )
    parser.add_argument(
        "--final-test-current-best",
        action="store_true",
        help="Evaluate only the current accepted best in the final test phase.",
    )
    parser.add_argument(
        "--final-test-baseline",
        action="store_true",
        help="Include baseline in the final test phase.",
    )
    return parser


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse loop CLI args. Used by both `main()` and the unified CLI."""
    return build_arg_parser().parse_args(argv)


def run(args: argparse.Namespace) -> int:
    """Execute the optimizer loop. Returns process exit code."""
    from meta_agent.commands.inspect import list_candidates

    if getattr(args, "accept_on_holdout", False) and not args.holdout_benchmark:
        print(
            "ERROR: --accept-on-holdout requires --holdout",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "resume_from_proposal", False) and not getattr(args, "resume", False):
        print(
            "ERROR: --resume-from-proposal requires --resume",
            file=sys.stderr,
        )
        return 2
    # Resolve env-driven defaults for model / proposer model / proposer backend
    # so a non-Bedrock provider (openrouter/anthropic) works with no extra flags.
    from meta_agent.services.llm import (
        default_eval_model,
        default_proposer_cli,
        default_proposer_model,
    )
    if not getattr(args, "model", None):
        args.model = default_eval_model()
    if not getattr(args, "proposer_model", None):
        args.proposer_model = default_proposer_model(args.model)
    if not getattr(args, "proposer_cli", None):
        args.proposer_cli = default_proposer_cli()

    if getattr(args, "candidates_per_iter", None) is None:
        default_k = _benchmark_candidates_per_iter_default(
            args.benchmark, get_workspace_root(),
        )
        args.candidates_per_iter = default_k or 1

    requested_candidates_per_iter = getattr(args, "candidates_per_iter", None) or 1
    if getattr(args, "resume_from_proposal", False) and requested_candidates_per_iter <= 1:
        print(
            "ERROR: --resume-from-proposal currently supports only --candidates-per-iter > 1",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "final_test_split", None) and not getattr(args, "final_test_benchmark", None):
        print(
            "ERROR: --final-test-split requires --final-test",
            file=sys.stderr,
        )
        return 2
    if getattr(args, "max_proposer_failures", 5) < 1:
        print(
            "ERROR: --max-proposer-failures must be >= 1",
            file=sys.stderr,
        )
        return 2

    state = _prepare_loop_state(args)

    try:
        _print_run_header(state)
        list_candidates(state.experience_dir)
        print()

        _maybe_run_baseline(state)
        _seed_history_from_baseline_on_disk(state)

        epoch_idx = state.effective_start_from
        final_epoch_exclusive = (
            state.effective_start_from + state.effective_iterations
        )
        consecutive_proposer_failures = 0
        max_proposer_failures = getattr(args, "max_proposer_failures", 5)

        while epoch_idx < final_epoch_exclusive:
            consumed_epoch = _run_one_epoch(state, epoch_idx)
            if not consumed_epoch:
                consecutive_proposer_failures += 1
                print()
                list_candidates(state.experience_dir)
                if consecutive_proposer_failures >= max_proposer_failures:
                    print(
                        "ERROR: proposer failed "
                        f"{consecutive_proposer_failures} consecutive time(s) "
                        f"at epoch {epoch_idx}; aborting without consuming "
                        "the epoch. Restart after fixing proposer/runtime issues.",
                        file=sys.stderr,
                    )
                    return 1
                continue

            consecutive_proposer_failures = 0
            _maybe_evolve_skill(state)
            print()
            list_candidates(state.experience_dir)
            epoch_idx += 1

        print(f"\n{'='*60}")
        print(f"  Evolution complete — {len(state.history)} iterations")
        print(f"  Best: {state.best_rate:.0%}")
        print(f"{'='*60}\n")
        list_candidates(state.experience_dir)
        run_final_eval(state)
        return 0
    finally:
        release_run_lock(state.run_name)


def main() -> None:
    configure_logging()
    sys.exit(run(parse_args()))


if __name__ == "__main__":
    main()
