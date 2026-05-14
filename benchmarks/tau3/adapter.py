"""tau-bench v3 adapter.

Owns the `TauBackend` pydantic schema, the per-task orchestration loop, and
the `register(...)` call that wires this adapter into `meta_agent.core.adapters`.
The Claude-SDK-specific per-task implementation lives in `sdk_adapter.py`.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, List, Optional

from pydantic import BaseModel, Field

from benchmarks.tau3.stage2 import (
    BaselineTrajectory,
    JudgeAsRewardConfig,
    compute_reward as _stage2_compute_reward,
    judge_candidate as _stage2_judge_candidate,
    load_baseline_pool as _stage2_load_baseline_pool,
)
from benchmarks.tau3.pointwise_reward import (
    PointwiseJudgeRewardConfig,
    judge_trajectory_pointwise as _pointwise_judge_trajectory,
    validate_pointwise_reward_config as _validate_pointwise_reward_config,
)
from meta_agent.core import adapters
from meta_agent.core.benchmark import Benchmark
from meta_agent.core.targets import detect_target
from meta_agent.task_runner import TaskResult


_TAU2_LOGGING_CONFIGURED = False


class _LiteLLMNoiseFilter:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._drop_blank_chunks = 0
        self._patterns = [
            re.compile(
                r"(?:\x1b\[[0-9;]*m)*Provider List: https://docs\.litellm\.ai/docs/providers(?:\x1b\[[0-9;]*m)*"
            ),
            re.compile(
                r"(?:\x1b\[[0-9;]*m)*Give Feedback / Get Help: https://github\.com/BerriAI/litellm/issues/new(?:\x1b\[[0-9;]*m)*"
            ),
            re.compile(
                r"LiteLLM\.Info: If you need to debug this error, use `litellm\._turn_on_debug\(\)'\\?\."
            ),
        ]

    def write(self, data: str) -> int:
        filtered = str(data)
        removed_noise = False
        for pattern in self._patterns:
            updated = pattern.sub("", filtered)
            if updated != filtered:
                removed_noise = True
            filtered = updated
        if filtered.strip():
            self._stream.write(filtered)
            self._drop_blank_chunks = 0
        elif removed_noise:
            # LiteLLM writes the boilerplate and surrounding blank lines as
            # separate chunks. Drop only the immediate blank aftermath, not all
            # whitespace globally; normal loggers often emit newlines separately.
            self._drop_blank_chunks = 3
        elif self._drop_blank_chunks > 0:
            self._drop_blank_chunks -= 1
        else:
            self._stream.write(filtered)
        return len(data)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._stream, "isatty", lambda: False)())

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _configure_tau2_logging() -> None:
    """Keep real tau2 errors, but hide conversation/debug chatter."""
    global _TAU2_LOGGING_CONFIGURED
    if _TAU2_LOGGING_CONFIGURED:
        return
    sys.stderr = _LiteLLMNoiseFilter(sys.stderr)  # type: ignore[assignment]
    sys.stdout = _LiteLLMNoiseFilter(sys.stdout)  # type: ignore[assignment]
    try:
        from loguru import logger as loguru_logger

        loguru_logger.remove()
        noisy_modules = (
            "tau2.user.user_simulator",
            "tau2.environment.environment",
            "tau2.domains.airline.tools",
        )

        def keep_record(record: dict[str, Any]) -> bool:
            module_name = str(record.get("name") or "")
            message = str(record.get("message") or "")
            if module_name.startswith(noisy_modules):
                return False
            if module_name == "tau2.utils.llm_utils" and "This model isn't mapped yet" in message:
                return False
            return True

        loguru_logger.add(sys.stderr, level="WARNING", filter=keep_record)
    except Exception:  # noqa: BLE001 - logging suppression should never break eval
        pass
    _TAU2_LOGGING_CONFIGURED = True


class TauBackend(BaseModel):
    tau_repo: str = ""
    domains: List[str] = Field(default_factory=lambda: ["airline", "retail"])
    user_model: str = "gpt-4o"
    user_model_provider: str = "openai"
    task_ids: Optional[List[str]] = None
    judge_model: Optional[str] = None
    judge_strategy: str = "binary"
    sample_size: Optional[int] = None
    # Stage-2 only (Experiment A §6): when present, every rollout's reward
    # is computed pairwise against the cached baseline trajectory via a
    # frozen judge harness. See benchmarks/tau3/stage2.py.
    judge_as_reward: Optional[JudgeAsRewardConfig] = None
    # Closed-loop pointwise reward path: every rollout's reward is the
    # normalized scalar emitted by a frozen pointwise evaluator harness.
    # This intentionally differs from judge_as_reward: no baseline comparison,
    # no official labels in the reward, and no official labels in search traces.
    pointwise_judge_reward: Optional[PointwiseJudgeRewardConfig] = None


def _resolve_tau_user_model(model: str | None, provider: str | None) -> str | None:
    """Resolve benchmark user-model config to the model string tau2/LiteLLM expects."""
    if not model:
        return None
    provider_name = (provider or "").strip().lower().replace("_", "-")
    model_name = model.strip()
    if provider_name in {"foundry", "azure-foundry", "openai-v1", "azure-openai-v1"}:
        base_url = (
            os.environ.get("AZURE_OPENAI_V1_BASE")
            or os.environ.get("AZURE_FOUNDRY_OPENAI_BASE")
            or ""
        ).strip()
        if not base_url:
            api_base = os.environ.get("AZURE_API_BASE", "").strip()
            if api_base:
                base_url = api_base.rstrip("/") + "/openai/v1"
        api_key = (
            os.environ.get("AZURE_FOUNDRY_API_KEY")
            or os.environ.get("AZURE_API_KEY")
            or ""
        ).strip()
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url.rstrip("/")
            os.environ["OPENAI_BASE_URL"] = base_url.rstrip("/")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if model_name.startswith("openai/"):
            return model_name
        return f"openai/{model_name}"
    if provider_name in {"azure", "azure-openai"}:
        for source, target in (
            ("TAU3_AZURE_API_BASE", "AZURE_API_BASE"),
            ("TAU3_AZURE_API_KEY", "AZURE_API_KEY"),
            ("TAU3_AZURE_API_VERSION", "AZURE_API_VERSION"),
        ):
            if os.environ.get(source):
                os.environ[target] = os.environ[source]
        compact_model = (
            model_name.lower()
            .replace("azure/", "")
            .replace("azure:", "")
            .replace("-", "")
            .replace(".", "")
            .replace("_", "")
        )
        env_candidates = [
            f"TAU3_AZURE_{compact_model.upper()}_DEPLOYMENT",
            f"AZURE_FOUNDRY_{compact_model.upper()}_DEPLOYMENT",
            f"AZURE_{compact_model.upper()}_DEPLOYMENT",
            f"AZURE_OPENAI_{compact_model.upper()}_DEPLOYMENT",
        ]
        deployment = next(
            (os.environ[key].strip() for key in env_candidates if os.environ.get(key)),
            "",
        )
        if deployment:
            return f"azure/{deployment}"
        if model_name.startswith("azure/"):
            return model_name
        if model_name.startswith("azure:"):
            return "azure/" + model_name.split(":", 1)[1]
        return f"azure/{model_name}"
    return model_name


def parse_backend(bench: Benchmark) -> TauBackend:
    """Validate `bench.backend` against TauBackend (defaults if missing)."""
    return TauBackend.model_validate(bench.backend or {})


def task_pool(bench: Benchmark) -> List[str]:
    """Tau-bench candidate task pool (explicit task_ids if set)."""
    return list(parse_backend(bench).task_ids or [])


async def run(
    *,
    benchmark: Benchmark,
    config_path: str,
    model: str,
    concurrency: int,
    task_filter: Optional[List[str]] = None,
    **_unused: Any,
) -> List[TaskResult]:
    """Run tau-bench tasks in parallel through the detected tau3 agent runtime."""
    _configure_tau2_logging()
    sdk = importlib.import_module("benchmarks.tau3.sdk_adapter")
    target_name = detect_target(Path(config_path)).name
    program = (
        importlib.import_module("benchmarks.tau3.program_adapter")
        if target_name == "program_harness"
        else None
    )

    tau_backend = parse_backend(benchmark)
    domains = tau_backend.domains
    if task_filter:
        filtered = [d for d in domains if d in task_filter]
        if filtered:
            domains = filtered

    user_model: Optional[str] = _resolve_tau_user_model(
        tau_backend.user_model or None,
        tau_backend.user_model_provider,
    )

    from tau2.runner import get_tasks as _get_tasks

    trace_dir = Path(tempfile.mkdtemp(prefix="tau_traces_"))

    task_list: list[tuple[str, Any]] = []
    for domain in domains:
        tasks = _get_tasks(domain)
        if tau_backend.task_ids:
            id_set = set(tau_backend.task_ids)
            tasks = [t for t in tasks if str(t.id) in id_set]
        for task in tasks:
            task_list.append((domain, task))

    if tau_backend.sample_size and tau_backend.sample_size < len(task_list):
        full_count = len(task_list)
        task_list = random.sample(task_list, tau_backend.sample_size)
        print(f"  [TAU] Sampled {len(task_list)} from {full_count} tasks")

    if tau_backend.task_ids and not task_list:
        sample_ids = [str(t.id) for t in _get_tasks(domains[0])[:3]]
        raise ValueError(
            f"task_ids filter matched 0 tasks. Actual IDs look like: {sample_ids}"
        )

    n_total = len(task_list)
    print(f"  [TAU] Running {n_total} tasks across {len(domains)} domain(s), concurrency={concurrency}")

    # Stage-2 reward path (Experiment A §6): load the cached baseline pool
    # once so every task's judge call reuses the same flattened text.
    stage2_cfg: Optional[JudgeAsRewardConfig] = tau_backend.judge_as_reward
    pointwise_cfg: Optional[PointwiseJudgeRewardConfig] = tau_backend.pointwise_judge_reward
    if stage2_cfg is not None and pointwise_cfg is not None:
        raise ValueError("Set only one of judge_as_reward or pointwise_judge_reward")
    stage2_baseline_pool: dict[str, BaselineTrajectory] = {}
    if stage2_cfg is not None:
        stage2_baseline_pool = _stage2_load_baseline_pool(
            Path(stage2_cfg.baseline_pool_path)
        )
        missing = [
            str(t.id) for (_dom, t) in task_list
            if str(t.id) not in stage2_baseline_pool
        ]
        if missing:
            raise ValueError(
                f"judge_as_reward: baseline cache missing {len(missing)} task(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Regenerate via benchmarks/tau3/cache_baseline_trajectories.py."
            )
        print(
            f"  [TAU-STAGE2] judge={stage2_cfg.config_path} "
            f"baseline={stage2_cfg.baseline_pool_path} "
            f"n_baseline={len(stage2_baseline_pool)}"
        )
    if pointwise_cfg is not None:
        _validate_pointwise_reward_config(pointwise_cfg)
        print(
            f"  [TAU-POINTWISE-REWARD] judge={pointwise_cfg.config_path} "
            f"model={pointwise_cfg.model or model} "
            f"score_range=[{pointwise_cfg.min_score}, {pointwise_cfg.max_score}]"
        )

    sem = asyncio.Semaphore(concurrency)
    completed = 0
    running_passed = 0
    _lock = asyncio.Lock()

    TASK_TIMEOUT_S = 600
    MAX_RETRIES = 10
    RETRY_DELAY_S = 15

    async def _run_one(domain: str, task: Any) -> TaskResult:
        nonlocal completed, running_passed
        task_id = str(task.id)
        task_name = f"{domain}_{task_id}"

        last_err: Exception | None = None
        r = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with sem:
                    if target_name == "program_harness":
                        r = await asyncio.wait_for(
                            program.run_tau_task_program(
                                domain=domain,
                                task_id=task_id,
                                config_path=config_path,
                                model=model,
                                user_model=user_model,
                                judge_model=tau_backend.judge_model,
                                judge_strategy=tau_backend.judge_strategy,
                                timeout_s=TASK_TIMEOUT_S,
                            ),
                            timeout=TASK_TIMEOUT_S,
                        )
                    else:
                        r = await asyncio.wait_for(
                            sdk.run_tau_task_sdk(
                                domain=domain,
                                task_id=task_id,
                                config_path=config_path,
                                model=model,
                                user_model=user_model,
                                judge_model=tau_backend.judge_model,
                                judge_strategy=tau_backend.judge_strategy,
                            ),
                            timeout=TASK_TIMEOUT_S,
                        )
                break
            except asyncio.TimeoutError:
                last_err = asyncio.TimeoutError()
                break
            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    print(
                        f"  [{task_name}] Attempt {attempt}/{MAX_RETRIES} failed "
                        f"({type(e).__name__}), retrying in {RETRY_DELAY_S}s...",
                        flush=True,
                    )
                    await asyncio.sleep(RETRY_DELAY_S)
                    continue

        if r is None:
            async with _lock:
                completed += 1
                rate = running_passed / completed
                print(
                    f"  [{completed:>3}/{n_total}] ERROR  {task_name:<20}  "
                    f"{type(last_err).__name__}: {last_err}  pass_rate={rate:.0%}",
                    flush=True,
                )
            return TaskResult(
                task_name=task_name,
                passed=False,
                reward=0.0,
                cost_usd=None,
                num_turns=None,
                duration_ms=0,
                wall_time_s=0.0,
                input_tokens=None,
                output_tokens=None,
                cache_tokens=None,
                session_id=None,
                work_dir="",
                verify_exit_code=1,
                verify_output=f"ERROR: {type(last_err).__name__}: {last_err}",
            )

        # Stage-2 reward override: replace the raw tau2 gold reward with the
        # pairwise-judge verdict (1 if judge picked cand over cached baseline,
        # else 0). Gold reward is preserved in the trace for audit.
        judge_cost: Optional[float] = None
        judge_trace_record: Optional[dict[str, Any]] = None
        if stage2_cfg is not None:
            baseline = stage2_baseline_pool[task_id]
            try:
                judge_reward, judge_outcome, reward_pair = await _stage2_judge_candidate(
                    config=stage2_cfg,
                    task_id=task_id,
                    cand_conversation=r.tau2_conversation,
                    baseline=baseline,
                    model=model,
                )
            except Exception as judge_err:  # noqa: BLE001 — surface judge errors
                judge_reward = 0.0
                judge_outcome = None
                judge_trace_record = {
                    "type": "stage2_judge_error",
                    "error": f"{type(judge_err).__name__}: {judge_err}",
                }
                reward_pair = None
            r.reward = judge_reward
            r.passed = judge_reward > 0
            if judge_outcome is not None:
                judge_cost = sum(
                    (o.cost_usd or 0.0)
                    for o in judge_outcome.orderings
                    if o.cost_usd is not None
                ) or None
                judge_trace_record = {
                    "type": "stage2_judge",
                    "task_id": task_id,
                    "pair_id": reward_pair.pair.pair_id if reward_pair else None,
                    "cand_slot": "A" if (reward_pair and reward_pair.cand_is_a) else "B",
                    "gold": judge_outcome.gold,
                    "decisions": list(judge_outcome.decisions),
                    "reward": judge_reward,
                    "baseline_gold_reward": baseline.gold_reward,
                    "error": judge_outcome.error,
                    "wall_time_s": judge_outcome.wall_time_s,
                    "cost_usd": judge_cost,
                }
        elif pointwise_cfg is not None:
            pointwise_result = await _pointwise_judge_trajectory(
                config=pointwise_cfg,
                task_id=task_id,
                cand_conversation=r.tau2_conversation,
                model=model,
            )
            r.reward = pointwise_result.reward
            r.passed = pointwise_result.passed
            judge_cost = pointwise_result.cost_usd
            judge_trace_record = pointwise_result.trace_record

        task_trace_dir = trace_dir / task_name
        task_trace_dir.mkdir(parents=True, exist_ok=True)
        with open(task_trace_dir / "trace.jsonl", "w") as f:
            for msg in r.messages:
                f.write(json.dumps(msg) + "\n")
            for tool_call in r.tool_calls:
                f.write(json.dumps({"type": "tool_call", **tool_call}) + "\n")
            if getattr(r, "error", None):
                f.write(json.dumps({"type": "agent_error", "error": r.error}) + "\n")
            grading: dict[str, Any] = {"type": "grading", "reward": r.reward}
            if pointwise_cfg is None:
                grading["gold_reward"] = r.gold_reward
            else:
                grading["reward_source"] = "pointwise_judge_reward"
                grading["official_gold_reward_hidden"] = True
            if tau_backend.judge_model:
                grading["judge_model"] = tau_backend.judge_model
            if stage2_cfg is not None:
                grading["stage2_reward"] = r.reward
                grading["stage2_judge_config"] = stage2_cfg.config_path
            if pointwise_cfg is not None:
                grading["pointwise_judge_config"] = pointwise_cfg.config_path
            f.write(json.dumps(grading) + "\n")
            if judge_trace_record is not None:
                f.write(json.dumps(judge_trace_record) + "\n")

        if getattr(r, "events", None):
            with open(task_trace_dir / "events.jsonl", "w") as f:
                for event in r.events:
                    f.write(json.dumps(event) + "\n")
            with open(task_trace_dir / "action_sequence.jsonl", "w") as f:
                for event in r.events:
                    if event.get("type") in {"action", "observation", "grading"}:
                        f.write(json.dumps(event) + "\n")
        if getattr(r, "tau2_conversation", None):
            with open(task_trace_dir / "tau2_conversation.jsonl", "w") as f:
                for message in r.tau2_conversation:
                    f.write(json.dumps(message) + "\n")

        total_cost = r.cost_usd
        if judge_cost is not None:
            total_cost = (total_cost or 0.0) + judge_cost

        async with _lock:
            completed += 1
            if r.passed:
                running_passed += 1
            mark = "PASS" if r.passed else "FAIL"
            rate = running_passed / completed
            gold_tag = ""
            if tau_backend.judge_model:
                gm = "G+" if r.gold_reward > 0 else "G-"
                jm = "J+" if r.passed else "J-"
                gold_tag = f"  {jm} {gm}"
            elif stage2_cfg is not None:
                gm = "G+" if r.gold_reward > 0 else "G-"
                jm = "J+" if r.passed else "J-"
                gold_tag = f"  {jm} {gm}"
            elif pointwise_cfg is not None:
                gold_tag = f"  J={r.reward:.2f}"
            print(
                f"  [{completed:>3}/{n_total}] {mark}  {task_name:<20} "
                f"turns={r.num_turns:<3} cost=${total_cost or 0:.3f}  "
                f"{r.duration_s:.0f}s  pass_rate={rate:.0%}{gold_tag}",
                flush=True,
            )

        return TaskResult(
            task_name=task_name,
            passed=r.passed,
            reward=r.reward,
            cost_usd=total_cost,
            num_turns=r.num_turns,
            duration_ms=int(r.duration_s * 1000),
            wall_time_s=r.duration_s,
            input_tokens=None,
            output_tokens=None,
            cache_tokens=None,
            session_id=r.session_id,
            work_dir=str(task_trace_dir),
            verify_exit_code=0 if r.passed else 1,
            verify_output=getattr(r, "error", None) or "",
        )

    return list(await asyncio.gather(*[_run_one(domain, task) for domain, task in task_list]))


# tau-bench owns the user simulator and environment tools. Claude Agent SDK
# harnesses receive them as MCP tools; program harnesses receive a safe
# ctx.task interface with async talk_to_customer/call_tool methods.
_SUPPORTED_TARGETS = frozenset({"claude_agent_sdk", "program_harness"})

# Both "tau" and "tau3" YAML types route to this adapter.
adapters.register(adapters.BenchmarkAdapter(
    name="tau", run=run, task_pool=task_pool, supported_targets=_SUPPORTED_TARGETS,
))
adapters.register(adapters.BenchmarkAdapter(
    name="tau3", run=run, task_pool=task_pool, supported_targets=_SUPPORTED_TARGETS,
))
