"""Stage-2 reward-via-judge glue for the tau3 adapter.

Experiment A Stage-2 (see docs/internal/EXPERIMENT_A_TASK_SUCCESS_PLAN.md
§6): harness-tune a frozen actor on τ³-airline search tasks using a frozen
Stage-1 champion judge as the reward signal. Each task's reward = 1 if the
judge picks the candidate's trajectory over the cached baseline
trajectory, else 0.

Isolating the Stage-2 pieces here keeps `benchmarks/tau3/adapter.py`'s
Stage-1 / regular-tau3 code paths byte-for-byte unchanged; Stage-2
activates only when ``TauBackend.judge_as_reward`` is set.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, cast

from pydantic import BaseModel

from meta_agent.task_runner.judge_runner import (
    JudgeDecision,
    JudgePair,
    _PairOutcome,
    judge_pair,
)
from benchmarks.tau3_trajectory_judge.adapter import (
    _PAIR_FRAMING_QUESTION,
)
from benchmarks.tau3_trajectory_judge.formatting import flatten_conversation


class JudgeAsRewardConfig(BaseModel):
    """Backend sub-config that activates the Stage-2 reward path.

    When present on ``TauBackend``, the adapter wraps each actor rollout
    with a pairwise judge call against a cached baseline trajectory.
    Absent ⇒ vanilla tau3 behavior (gold τ² evaluator, optional scalar judge).

    Fields
    ------
    config_path
        Path to the frozen judge harness directory (must contain
        ``harness.py``) or to the harness.py file itself. The judge's
        ``build_options(ctx)`` signature is invoked verbatim via
        ``sdk_judge_runner.judge_pair``.
    baseline_pool_path
        JSONL with one record per ``task_id`` holding the cached
        baseline rollout. Schema: ``{task_id, domain, conversation,
        ...}`` where ``conversation`` is the tau2 message-dump list
        (same shape as Stage-1 pools).
    model
        Model name for the judge SDK calls. Defaults to the actor model
        when None (keeps Stage-2 single-model by default).
    timeout_s
        Per-ordering judge timeout. Matches Stage-1 default (180s).
    position_seed
        RNG seed for the per-task coin flip that assigns cand to slot A
        or B. Deterministic across re-runs of the same adapter call.
    """

    config_path: str
    baseline_pool_path: str
    model: Optional[str] = None
    timeout_s: int = 180
    position_seed: int = 42


@dataclass(frozen=True)
class BaselineTrajectory:
    """Cached baseline rollout, pre-flattened for judge consumption."""

    task_id: str
    domain: str
    conversation_text: str
    gold_reward: Optional[float]
    actor_model: Optional[str]


def load_baseline_pool(path: Path) -> dict[str, BaselineTrajectory]:
    """Load a baseline-pool JSONL and pre-flatten conversations.

    Returns a dict keyed by ``task_id`` so the per-task lookup in
    ``build_reward_pair`` is O(1). Malformed JSONL lines raise —
    Stage-2 runs should fail loud if the cache is inconsistent.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Baseline trajectory pool not found: {path}")

    out: dict[str, BaselineTrajectory] = {}
    with path.open() as f:
        for line_num, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_num} of {path}: {exc}"
                ) from exc
            task_id = str(rec.get("task_id", ""))
            if not task_id:
                raise ValueError(
                    f"Line {line_num} of {path} has no task_id"
                )
            if task_id in out:
                raise ValueError(
                    f"Duplicate task_id={task_id!r} in {path}"
                )
            out[task_id] = BaselineTrajectory(
                task_id=task_id,
                domain=str(rec.get("domain", "")),
                conversation_text=flatten_conversation(
                    rec.get("conversation") or []
                ),
                gold_reward=(
                    float(rec["gold_reward"])
                    if rec.get("gold_reward") is not None
                    else None
                ),
                actor_model=rec.get("actor_model"),
            )
    return out


@dataclass(frozen=True)
class RewardPair:
    """Materialized Stage-2 reward pair with the coin-flip recorded."""

    pair: JudgePair
    cand_is_a: bool  # True ⇒ cand in slot A, gold = "A>B"; else slot B, gold="B>A"


def build_reward_pair(
    *,
    task_id: str,
    cand_conversation: list[dict[str, Any]],
    baseline: BaselineTrajectory,
    position_seed: int,
) -> RewardPair:
    """Assemble a JudgePair with cand randomized to slot A or B.

    The coin flip is seeded by ``(position_seed, task_id)`` so re-running
    the same (cache, task) yields the same placement — this matters for
    reproducibility and for comparing consecutive proposer candidates.

    Gold semantics mirror Stage 1's ``run_judge_benchmark`` contract:
    the pair passes iff the judge's decision equals ``gold``. We set
    ``gold`` to the slot cand landed in, so "passed" ⟺ "judge picked
    cand" regardless of position. Downstream, ``compute_reward`` maps
    the _PairOutcome to a 0/1 scalar.
    """
    rng = random.Random(f"{position_seed}::{task_id}")
    cand_is_a = rng.random() < 0.5
    cand_text = flatten_conversation(cand_conversation)

    if cand_is_a:
        response_a, response_b = cand_text, baseline.conversation_text
        gold = cast(JudgeDecision, "A>B")
    else:
        response_a, response_b = baseline.conversation_text, cand_text
        gold = cast(JudgeDecision, "B>A")

    pair = JudgePair(
        pair_id=f"stage2_task{task_id}",
        question=_PAIR_FRAMING_QUESTION,
        response_a=response_a,
        response_b=response_b,
        gold=gold,
        source=f"cand|baseline:{baseline.actor_model or '?'}",
        category=f"task{task_id}",
    )
    return RewardPair(pair=pair, cand_is_a=cand_is_a)


def compute_reward(outcome: _PairOutcome) -> float:
    """Map a single-ordering ``_PairOutcome`` to a 0/1 Stage-2 reward.

    With ``judge_pair(..., position_swap=False)`` there is exactly one
    decision. Reward = 1.0 when the decision equals gold (i.e. the
    judge picked cand, given ``gold`` was set to cand's slot). An
    errored / format-failed pair yields 0.0 — same policy as Stage-1
    ``_position_consistent_correct``.
    """
    if outcome.error is not None:
        return 0.0
    decisions = outcome.decisions
    if not decisions:
        return 0.0
    return 1.0 if decisions[0] == outcome.gold else 0.0


async def judge_candidate(
    *,
    config: JudgeAsRewardConfig,
    task_id: str,
    cand_conversation: list[dict[str, Any]],
    baseline: BaselineTrajectory,
    model: str,
) -> tuple[float, _PairOutcome, RewardPair]:
    """End-to-end Stage-2 reward for one (cand, cached-base) rollout pair.

    ``model`` is the fallback used when ``config.model`` is unset — the
    actor model. Returns (reward, outcome, reward_pair); the adapter
    writes the outcome to its per-task trace so proposers see WHY a
    cand won or lost.
    """
    reward_pair = build_reward_pair(
        task_id=task_id,
        cand_conversation=cand_conversation,
        baseline=baseline,
        position_seed=config.position_seed,
    )
    outcome = await judge_pair(
        harness_path=_resolve_judge_harness_path(config.config_path),
        pair=reward_pair.pair,
        model=config.model or model,
        timeout=config.timeout_s,
        position_swap=False,
    )
    return compute_reward(outcome), outcome, reward_pair


def _resolve_judge_harness_path(config_path: str) -> Path:
    """Normalize a judge harness ref to the ``harness.py`` file path."""
    p = Path(config_path)
    if p.is_file() and p.name == "harness.py":
        return p
    if p.is_dir() and (p / "harness.py").is_file():
        return p / "harness.py"
    raise ValueError(
        f"judge_as_reward.config_path must contain harness.py; got {config_path!r}"
    )
