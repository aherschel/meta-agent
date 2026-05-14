"""Integration smoke-check for tau2.run — R7 de-risk gate for pool building.

Before committing to a full multi-actor τ³-airline trajectory pool (see
`docs/internal/EXPERIMENT_A_TASK_SUCCESS_PLAN.md` §9 Phase 1), this script
runs one airline task end-to-end for each requested provider and asserts:

    1. `tau2.run.run_tasks` is importable and callable from this env
    2. A `TextRunConfig` with the chosen `llm_agent` produces a valid Results
    3. `SimulationRun.reward_info.reward` is populated (float, not None)
       — this is the τ² pass/fail label the pool will key on
    4. `SimulationRun.get_messages()` returns a non-empty conversation
    5. Both Anthropic (via LiteLLM → Anthropic API or Bedrock) and OpenAI
       paths work with whatever auth is configured in the current environment

This is a *live* script: it calls production LLM APIs and costs real money
(~$0.15 per full run across both providers). It is NOT a pytest — tests in
`tests/` must stay <2s and hermetic per repo convention.

Typical output on success::

    [smoke] claude-haiku-4-5:  task=0  reward=1.0  turns=14  cost=$0.078  PASS
    [smoke] gpt-4o-mini:        task=0  reward=0.0  turns=22  cost=$0.015  PASS
    [smoke] Integration ready — tau2.run is plumbed for multi-provider pool build.

Required env vars::

    TAU2_DATA_DIR               # path to a `data/` dir from a tau2-bench repo clone
                                # (typically <parent>/tau2-bench/data)
    AWS_BEARER_TOKEN_BEDROCK    # for Bedrock-routed Anthropic models (default)
    AWS_REGION                  # Bedrock region (e.g. us-east-1)
    OPENAI_API_KEY              # for OpenAI-routed models

Usage::

    # Both providers (default; Bedrock for Anthropic, OpenAI direct for OpenAI)
    TAU2_DATA_DIR=../tau2-bench/data \\
        python -m benchmarks.tau3_trajectory_judge.smoke

    # One provider only
    python -m benchmarks.tau3_trajectory_judge.smoke --only anthropic
    python -m benchmarks.tau3_trajectory_judge.smoke --only openai

    # Override models / task
    python -m benchmarks.tau3_trajectory_judge.smoke \\
        --anthropic-model claude-sonnet-4-6 \\
        --openai-model gpt-5.4-mini \\
        --task-id 1 --max-steps 40

Exit codes::

    0 — every requested provider produced a valid trajectory with populated reward
    1 — any provider failed (import error, API error, missing reward, empty messages)
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional


@dataclass
class _ProviderSpec:
    label: str             # "anthropic" | "openai"
    model: str             # LiteLLM model name (e.g. "claude-haiku-4-5", "gpt-4o-mini")


@dataclass
class _SmokeResult:
    provider: str
    model: str
    task_id: str
    reward: Optional[float]
    num_messages: int
    cost_usd: Optional[float]
    duration_s: float
    ok: bool
    error: Optional[str] = None


def _run_one(
    provider: _ProviderSpec,
    task_id: str,
    user_model: str,
    max_steps: int,
    max_concurrency: int,
) -> _SmokeResult:
    """Run one airline task through tau2.run and validate the returned schema."""
    start = time.time()
    try:
        # Import from `tau2.runner` (not `tau2.run`). The top-level `tau2.run`
        # module shadows `run_tasks` with a deprecated flat-args wrapper; the
        # config-style version we want lives in `tau2.runner.batch.run_tasks`.
        from tau2.data_model.simulation import TextRunConfig
        from tau2.runner import get_tasks, run_tasks
    except ImportError as exc:
        return _SmokeResult(
            provider=provider.label, model=provider.model, task_id=task_id,
            reward=None, num_messages=0, cost_usd=None,
            duration_s=time.time() - start, ok=False,
            error=f"import failed: {type(exc).__name__}: {exc}",
        )

    try:
        tasks = get_tasks("airline", task_ids=[task_id])
        if not tasks:
            return _SmokeResult(
                provider=provider.label, model=provider.model, task_id=task_id,
                reward=None, num_messages=0, cost_usd=None,
                duration_s=time.time() - start, ok=False,
                error=f"get_tasks returned empty list for task_id={task_id!r}",
            )

        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user="user_simulator",
            llm_agent=provider.model,
            llm_user=user_model,
            num_trials=1,
            max_steps=max_steps,
            max_concurrency=max_concurrency,
            seed=42,
            log_level="WARNING",
        )

        results = run_tasks(
            config,
            tasks,
            save_path=None,          # in-memory only
            console_display=False,
        )

        if not results.simulations:
            return _SmokeResult(
                provider=provider.label, model=provider.model, task_id=task_id,
                reward=None, num_messages=0, cost_usd=None,
                duration_s=time.time() - start, ok=False,
                error="run_tasks returned Results with zero simulations",
            )

        sim = results.simulations[0]
        messages = sim.get_messages()
        reward_info = sim.reward_info
        reward = reward_info.reward if reward_info is not None else None

        if reward is None:
            return _SmokeResult(
                provider=provider.label, model=provider.model, task_id=task_id,
                reward=None, num_messages=len(messages),
                cost_usd=sim.agent_cost, duration_s=sim.duration, ok=False,
                error="reward_info missing or reward=None — τ² evaluator did not run",
            )

        if not messages:
            return _SmokeResult(
                provider=provider.label, model=provider.model, task_id=task_id,
                reward=reward, num_messages=0,
                cost_usd=sim.agent_cost, duration_s=sim.duration, ok=False,
                error="SimulationRun.get_messages() returned empty list",
            )

        return _SmokeResult(
            provider=provider.label, model=provider.model, task_id=task_id,
            reward=float(reward), num_messages=len(messages),
            cost_usd=sim.agent_cost, duration_s=sim.duration, ok=True,
        )

    except Exception as exc:
        return _SmokeResult(
            provider=provider.label, model=provider.model, task_id=task_id,
            reward=None, num_messages=0, cost_usd=None,
            duration_s=time.time() - start, ok=False,
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _format_row(r: _SmokeResult) -> str:
    tag = "PASS" if r.ok else "FAIL"
    reward_str = f"{r.reward:.1f}" if r.reward is not None else "—"
    cost_str = f"${r.cost_usd:.3f}" if r.cost_usd is not None else "—"
    dur_str = f"{r.duration_s:.0f}s"
    return (
        f"[smoke] {r.model:<24} task={r.task_id:<4} reward={reward_str:<4} "
        f"msgs={r.num_messages:<3} cost={cost_str:<8} {dur_str:<6} {tag}"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="tau2.run integration smoke-check for trajectory-judge pool building.",
    )
    parser.add_argument(
        "--only",
        choices=("anthropic", "openai", "both"),
        default="both",
        help="Which provider(s) to smoke. Default: both.",
    )
    parser.add_argument(
        "--anthropic-model",
        default="bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
        help=(
            "LiteLLM model name for the Anthropic smoke. Default routes Haiku 4.5 "
            "via Bedrock cross-region inference (matches meta_agent.services.llm."
            "BEDROCK_MODEL_MAP). Requires AWS_BEARER_TOKEN_BEDROCK + AWS_REGION."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="LiteLLM model name for the OpenAI smoke. Default: gpt-4o-mini. Requires OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--user-model",
        default="gpt-4.1",
        help=(
            "LLM used by the τ² user simulator. Default: gpt-4.1 (tau2's default "
            "user-sim model). Override to match your production setup."
        ),
    )
    parser.add_argument(
        "--task-id",
        default="0",
        help="Airline task_id to run. Default: 0 (a simple task).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Max conversation turns per task. Default: 30.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Max concurrent tasks (smoke runs one task so 1 is fine). Default: 1.",
    )
    args = parser.parse_args(argv)

    providers: list[_ProviderSpec] = []
    if args.only in ("anthropic", "both"):
        providers.append(_ProviderSpec(label="anthropic", model=args.anthropic_model))
    if args.only in ("openai", "both"):
        providers.append(_ProviderSpec(label="openai", model=args.openai_model))

    print(
        f"[smoke] Running {len(providers)} provider(s) on airline task={args.task_id} "
        f"via tau2.runner.run_tasks ...",
        flush=True,
    )

    results: list[_SmokeResult] = []
    for spec in providers:
        r = _run_one(
            provider=spec,
            task_id=args.task_id,
            user_model=args.user_model,
            max_steps=args.max_steps,
            max_concurrency=args.max_concurrency,
        )
        print(_format_row(r), flush=True)
        if not r.ok and r.error:
            print(f"        error: {r.error.splitlines()[0]}", flush=True)
        results.append(r)

    all_ok = all(r.ok for r in results)
    if all_ok:
        print(
            "[smoke] Integration ready — tau2.run is plumbed for multi-provider pool build.",
            flush=True,
        )
        return 0

    failed = [r for r in results if not r.ok]
    print(
        f"[smoke] FAIL — {len(failed)}/{len(results)} provider(s) did not produce "
        "a valid trajectory with populated τ² reward.",
        flush=True,
    )
    for r in failed:
        print(f"[smoke] {r.provider} failure detail:", flush=True)
        print((r.error or "(no error captured)"), flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
