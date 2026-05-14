from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable, cast

from meta_agent.core.run_context import RunContext


class ResearchHarnessError(Exception):
    """Raised when a research harness module is invalid."""


@dataclass(frozen=True)
class ResearchRuntimeSettings:
    """Small set of runtime-facing knobs the research harness may control."""

    approval_policy: str = "never"
    sandbox: str = "workspace-write"


@dataclass(frozen=True)
class ResearchExample:
    """One input/response example the harness can provide to the model."""

    user: str
    assistant: str


@dataclass(frozen=True)
class ResearchHarnessSpec:
    """Typed contract for the single-file research harness.

    The research harness owns prompt/context/examples and a small set of
    runtime-facing knobs, while the adapter owns the per-task loop.
    """

    system_instructions: str
    task_context: str = ""
    examples: tuple[ResearchExample, ...] = ()
    max_attempts: int = 1
    runtime_settings: ResearchRuntimeSettings = field(default_factory=ResearchRuntimeSettings)


def _load_module_from_path(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ResearchHarnessError(f"cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _coerce_examples(raw_examples: object) -> tuple[ResearchExample, ...]:
    if raw_examples is None:
        return ()
    if not isinstance(raw_examples, (list, tuple)):
        raise ResearchHarnessError("examples must be a list or tuple of ResearchExample")
    normalized: list[ResearchExample] = []
    for idx, example in enumerate(raw_examples):
        if not isinstance(example, ResearchExample):
            raise ResearchHarnessError(
                f"examples[{idx}] must be ResearchExample, got {type(example).__name__}"
            )
        if not example.user.strip() or not example.assistant.strip():
            raise ResearchHarnessError(
                f"examples[{idx}] user and assistant must both be non-empty"
            )
        normalized.append(example)
    return tuple(normalized)


def _coerce_research_harness_spec(value: object) -> ResearchHarnessSpec:
    if not isinstance(value, ResearchHarnessSpec):
        raise ResearchHarnessError(
            f"build_harness() must return ResearchHarnessSpec, got {type(value).__name__}"
        )
    system_instructions = value.system_instructions.strip()
    if not system_instructions:
        raise ResearchHarnessError("system_instructions must be non-empty")
    task_context = value.task_context.strip()
    examples = _coerce_examples(value.examples)
    if value.max_attempts not in (1, 2):
        raise ResearchHarnessError("max_attempts must be 1 or 2")
    runtime_settings = value.runtime_settings
    if not runtime_settings.approval_policy.strip():
        raise ResearchHarnessError("runtime approval_policy must be non-empty")
    if not runtime_settings.sandbox.strip():
        raise ResearchHarnessError("runtime sandbox must be non-empty")
    return ResearchHarnessSpec(
        system_instructions=system_instructions,
        task_context=task_context,
        examples=examples,
        max_attempts=value.max_attempts,
        runtime_settings=runtime_settings,
    )


def compose_initial_prompt(spec: ResearchHarnessSpec, task_instruction: str) -> str:
    """Compose the first-pass prompt from a validated spec and the task text."""

    parts: list[str] = [spec.system_instructions.strip()]
    if spec.task_context:
        parts.append(f"Context:\n{spec.task_context}")
    if spec.examples:
        example_blocks: list[str] = []
        for i, example in enumerate(spec.examples, start=1):
            example_blocks.append(
                f"Example {i} input:\n{example.user.strip()}\n\n"
                f"Example {i} response:\n{example.assistant.strip()}"
            )
        parts.append("\n\n".join(example_blocks))
    parts.append(f"Task:\n{task_instruction.strip()}")
    return "\n\n".join(parts)


def build_research_harness_spec(
    harness_path: Path,
    ctx: RunContext,
) -> ResearchHarnessSpec:
    """Import harness.py and build a validated research harness spec."""

    module = _load_module_from_path(harness_path, "candidate_research_harness")
    builder_obj = getattr(module, "build_harness", None)
    if not callable(builder_obj):
        raise ResearchHarnessError("No callable build_harness(ctx) found")
    builder = cast(Callable[[RunContext], object], builder_obj)
    return _coerce_research_harness_spec(builder(ctx))


def validate_research_harness(harness_path: Path) -> ResearchHarnessSpec:
    """Validate that harness.py implements the expected contract."""

    if not harness_path.exists():
        raise ResearchHarnessError(f"{harness_path} does not exist")
    ctx = RunContext(cwd="/app", model="gpt-5.4", task_instruction="test task")
    return build_research_harness_spec(harness_path, ctx)
