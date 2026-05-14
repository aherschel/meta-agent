from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, cast


class ProgramHarnessError(Exception):
    """Raised when a program harness module is invalid or fails at runtime."""


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): _jsonable(v) for k, v in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [_jsonable(item) for item in value]
        return str(value)


@dataclass(frozen=True)
class ModelCallResult:
    text: str
    raw: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)
    cost_usd: Optional[float] = None


@dataclass(frozen=True)
class CommandResult:
    command: str | Sequence[str]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int


@dataclass
class HarnessResult:
    final_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    duration_ms: Optional[int] = None
    wall_time_s: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_tokens: Optional[int] = None
    session_id: Optional[str] = None


class HarnessContext:
    """Safe context passed to a candidate-owned program harness.

    The benchmark adapter decides what `task` contains. The context provides
    small helper APIs and event logging, but it must not expose labels or
    scorer internals.
    """

    def __init__(
        self,
        *,
        task: Any,
        model: str,
        cwd: str | Path,
        timeout: int = 300,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.task = task
        self.model = model
        self.cwd = str(cwd)
        self.timeout = timeout
        self.metadata = dict(metadata or {})
        self._events: list[dict[str, Any]] = []

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self._events)

    def log_event(self, event_type: str, **payload: Any) -> None:
        if not event_type.strip():
            raise ProgramHarnessError("event_type must be non-empty")
        self._events.append({
            "type": event_type,
            "timestamp": time.time(),
            **payload,
        })

    async def call_model(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: Optional[float] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> ModelCallResult:
        from meta_agent.services.llm import extract_text, invoke_model

        start = time.time()
        self.log_event(
            "model_input",
            model=self.model,
            system=system,
            messages=_jsonable([dict(message) for message in messages]),
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=_jsonable(dict(extra_body)) if extra_body is not None else None,
        )
        response = await invoke_model(
            model=self.model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        duration_ms = int((time.time() - start) * 1000)
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        text = extract_text(response)
        self.log_event(
            "raw_model_output",
            model=self.model,
            content=text,
            usage=_jsonable(dict(usage)),
            duration_ms=duration_ms,
        )
        self.log_event(
            "model_call",
            duration_ms=duration_ms,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )
        return ModelCallResult(text=text, raw=dict(response), usage=dict(usage))

    async def run_command(
        self,
        command: str | Sequence[str],
        *,
        timeout: Optional[int] = None,
        shell: Optional[bool] = None,
    ) -> CommandResult:
        actual_shell = isinstance(command, str) if shell is None else shell
        start = time.time()

        def _run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
                shell=actual_shell,
            )

        try:
            completed = await asyncio.to_thread(_run)
            duration_ms = int((time.time() - start) * 1000)
            result = CommandResult(
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                duration_ms=duration_ms,
            )
        except subprocess.TimeoutExpired as exc:
            duration_ms = int((time.time() - start) * 1000)
            result = CommandResult(
                command=command,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or f"TIMEOUT after {timeout or self.timeout}s",
                duration_ms=duration_ms,
            )

        self.log_event(
            "command",
            command=command if isinstance(command, str) else list(command),
            returncode=result.returncode,
            duration_ms=result.duration_ms,
        )
        return result

    def finish(self, final_output: Any, **metadata: Any) -> HarnessResult:
        return HarnessResult(
            final_output=final_output,
            metadata=metadata,
            events=self.events,
        )


ProgramRunFn = Callable[[HarnessContext], Awaitable[object]]


def _load_module_from_path(module_path: Path) -> ModuleType:
    module_path = module_path.resolve()
    module_name = f"candidate_program_harness_{abs(hash(module_path))}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ProgramHarnessError(f"cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(module_path.parent)
    sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(parent)
        except ValueError:
            pass
    return module


def _harness_file(path: Path) -> Path:
    return path if path.is_file() else path / "harness.py"


def load_program_harness(path: Path) -> ProgramRunFn:
    """Load a program harness and return its async run callable."""
    harness_path = _harness_file(path)
    if not harness_path.is_file():
        raise ProgramHarnessError(f"program harness not found: {harness_path}")

    module = _load_module_from_path(harness_path)
    run_obj = getattr(module, "run", None)
    if callable(run_obj):
        if not inspect.iscoroutinefunction(run_obj):
            raise ProgramHarnessError("top-level run(ctx) must be async")
        return cast(ProgramRunFn, run_obj)

    cls = getattr(module, "Harness", None)
    if cls is None:
        raise ProgramHarnessError("program harness must define async run(ctx) or class Harness")
    run_attr = getattr(cls, "run", None)
    if not inspect.iscoroutinefunction(run_attr):
        raise ProgramHarnessError("Harness.run(ctx) must be async")

    def _make_runner(ctx: HarnessContext) -> Awaitable[object]:
        try:
            instance = cls()
        except TypeError as exc:
            raise ProgramHarnessError("Harness must be constructible with no arguments") from exc
        return cast(Awaitable[object], instance.run(ctx))

    return _make_runner


def _coerce_harness_result(value: object, ctx: HarnessContext) -> HarnessResult:
    if isinstance(value, HarnessResult):
        if not value.events:
            value.events = ctx.events
        return value
    return HarnessResult(final_output=value, events=ctx.events)


async def run_program_harness(
    path: Path,
    ctx: HarnessContext,
    *,
    timeout: Optional[int] = None,
) -> HarnessResult:
    run = load_program_harness(path)
    start = time.time()
    value = await asyncio.wait_for(run(ctx), timeout=timeout or ctx.timeout)
    result = _coerce_harness_result(value, ctx)
    wall_time_s = time.time() - start
    if result.wall_time_s is None:
        result.wall_time_s = wall_time_s
    if result.duration_ms is None:
        result.duration_ms = int(wall_time_s * 1000)
    if not result.events:
        result.events = ctx.events
    return result


def validate_program_harness(path: Path) -> None:
    """Validate contract shape without executing the harness run."""
    load_program_harness(path)


def events_to_jsonl(events: Sequence[Mapping[str, Any]]) -> str:
    lines = [json.dumps(dict(event), default=str) for event in events]
    return "\n".join(lines) + ("\n" if lines else "")
