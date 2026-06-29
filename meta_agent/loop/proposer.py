"""Proposer invocation: subprocess plumbing + the public `invoke_proposer` entry.

Owns:
- the SURFACE_LOCK constants
- file-signature helpers used to detect what the proposer actually changed
- `_run_proposer_cli` (codex/claude subprocess + stream-json parsing)
- `invoke_proposer` (the loop's "go ask the proposer to write a new harness" call)
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from meta_agent.utils.logging import get_logger
from meta_agent.core.paths import PACKAGE_ROOT, get_workspace_root, rel_to_workspace
from meta_agent.loop.state import PROPOSER_INSTRUCTIONS_DIR
from meta_agent.loop.proposer_session_log import write_proposer_session_log
from meta_agent.core.targets import AgentTarget, get_target


@dataclass
class ProposerRunResult:
    """Outcome of one `_run_proposer_cli` invocation.

    Captures the cost-tracking surface we need for the candidate's
    ``proposer_cost.json`` sidecar. ``cost_usd`` comes from the final
    ``result`` stream-json event (both codex and claude CLIs emit it); it's
    the *proposer* call's cost, independent of eval-time model calls.
    """

    exit_code: int
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    model: Optional[str] = None
    cli: str = ""
    stderr: str = ""


@dataclass
class ProposerInvocationResult:
    """What `invoke_proposer` returns to the epoch caller."""

    success: bool
    run: Optional[ProposerRunResult] = None

logger = get_logger("loop")


DEFAULT_PROPOSER_MAX_TURNS = 200


SURFACE_LOCK_CHOICES = ("agents", "hooks", "config", "skills", "subagents")


def write_proposer_cost_sidecar(
    candidate_dir: Path,
    run: ProposerRunResult,
    shared_across: int = 1,
) -> None:
    """Persist a `proposer_cost.json` sidecar under `candidate_dir`.

    When one proposer call produces ``k`` candidates (k>1 iterations), the
    total cost is shared across them — pass ``shared_across=k`` and we
    record each candidate's attributable share as ``cost_usd / k`` alongside
    the raw ``cost_usd_total``. ``scores.json`` consumes the share (so
    summing `proposer_cost_usd` across siblings equals the real outlay).
    """
    if run.cost_usd is None and run.num_turns is None:
        return
    candidate_dir.mkdir(parents=True, exist_ok=True)
    total = run.cost_usd
    share = None if total is None else float(total) / max(1, shared_across)
    payload = {
        "cost_usd": share,
        "cost_usd_total": total,
        "shared_across": shared_across,
        "num_turns": run.num_turns,
        "cli": run.cli,
        "model": run.model,
        "input_tokens": run.input_tokens,
        "output_tokens": run.output_tokens,
        "cache_read_tokens": run.cache_read_tokens,
    }
    (candidate_dir / "proposer_cost.json").write_text(json.dumps(payload, indent=2))
SURFACE_LOCK_DESCRIPTION: dict[str, str] = {
    "agents": "`AGENTS.md`",
    "hooks": "`.codex/hooks.json` and `.codex/hooks/*`",
    "config": "`.codex/config.toml`",
    "skills": "`.codex/skills/*`",
    "subagents": "`.codex/agents/*`",
}


# --- File-signature + surface-lock helpers -------------------------------

def _complete_staged_candidate_count(
    staging_dir: Path,
    target: AgentTarget,
    candidates_per_iter: int,
) -> int:
    """Return the number of complete proposer outputs in staging.

    A proposer can fail at the transport layer after writing candidate files.
    This is especially common with streaming CLIs: the model may have already
    written and sanity-checked harnesses, then the stream can terminate with a
    non-zero process exit. Validation is the right layer to judge those files;
    the CLI exit code should only make the iteration fail when staging lacks a
    complete slate to validate.
    """
    required = target.required_written_files
    if not required:
        return 1 if staging_dir.exists() else 0

    if candidates_per_iter == 1:
        return 1 if any((staging_dir / filename).exists() for filename in required) else 0

    if not staging_dir.exists():
        return 0
    return sum(
        1
        for sub in staging_dir.iterdir()
        if sub.is_dir() and any((sub / filename).exists() for filename in required)
    )


def _staging_contains_required_files(
    staging_dir: Path,
    target: AgentTarget,
    candidates_per_iter: int,
) -> bool:
    """True iff staging contains at least one complete proposer output."""
    return _complete_staged_candidate_count(
        staging_dir, target, candidates_per_iter,
    ) > 0

def _copy_seed_harness_files(src_dir: Path, dst_dir: Path, target: AgentTarget) -> None:
    """Copy the target's harness-relevant files from a prior candidate into staging."""
    if not src_dir.is_dir():
        return
    allowed_files = set(target.harness_files) | {target.module_filename}
    allowed_dirs = set(target.harness_dirs)
    for item in src_dir.iterdir():
        is_allowed = (
            (item.is_file() and item.name in allowed_files) or
            (item.is_dir() and item.name in allowed_dirs)
        )
        if not is_allowed:
            continue
        dest = dst_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(item, dest)


def _collect_file_signatures(root: Path) -> dict[str, str]:
    """Return relative-path -> sha256 hash for every file under `root`."""
    signatures: dict[str, str] = {}
    if not root.exists():
        return signatures
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(root).as_posix()
        signatures[rel_path] = hashlib.sha256(path.read_bytes()).hexdigest()
    return signatures


def _diff_file_signatures(before: dict[str, str], after: dict[str, str]) -> list[str]:
    """Paths added/modified/deleted between two signature snapshots."""
    before_paths = set(before)
    after_paths = set(after)
    added = after_paths - before_paths
    deleted = before_paths - after_paths
    modified = {p for p in before_paths & after_paths if before[p] != after[p]}
    return sorted(added | deleted | modified)


def _is_surface_lock_allowed_path(target: AgentTarget, path: str, surface_lock: str) -> bool:
    if path == "proposal_notes.json":
        return True
    return target.is_surface_target_path(path, surface_lock)


def _surface_lock_violations(
    target: AgentTarget, changed_files: list[str], surface_lock: str,
) -> list[str]:
    return [
        path for path in changed_files
        if not _is_surface_lock_allowed_path(target, path, surface_lock)
    ]


def _experience_volume_rel(path: Path) -> str:
    """Return a path relative to the mounted experience-volume workspace.

    Modal may resolve ``/repo/experience`` to an internal ``/__modal/volumes``
    path. The isolated Codex proposer only sees the same volume mounted at
    ``/work/experience``, so never use ``Path.resolve()`` here.
    """
    parts = path.parts
    if "experience" in parts:
        idx = parts.index("experience")
        return Path(*parts[idx:]).as_posix()
    run_markers = {
        "candidates",
        "staging",
        "_internal",
        "proposer_briefs",
        "reports",
        "history.json",
        "frontier.json",
        "candidate_index.json",
    }
    for idx, part in enumerate(parts):
        if part in run_markers and idx > 0:
            return (Path("experience") / Path(*parts[idx - 1:])).as_posix()
    if path.is_absolute() and len(parts) >= 2:
        return (Path("experience") / parts[-1]).as_posix()
    return rel_to_workspace(path)


def _codex_visible_path(path: Path) -> str:
    """Path string visible to the Codex proposer."""
    if os.environ.get("META_AGENT_CODEX_PROPOSER_ISOLATED", "").strip() == "1":
        return f"/work/{_experience_volume_rel(path)}"
    return rel_to_workspace(path)


def _codex_fair_proposer_enabled() -> bool:
    """True when Codex should use the same broad proposer surface as Claude/Opus."""
    return os.environ.get("META_AGENT_CODEX_FAIR_PROPOSER", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


# --- CLI subprocess plumbing ---------------------------------------------

def _build_codex_cmd(prompt: str, model: Optional[str]) -> list[str]:
    """Compatibility shim for tests and older imports.

    Real Codex proposer sessions now go through `meta_agent.loop.codex_wrapper`
    and supply the prompt on stdin. The prompt argument is intentionally ignored
    here; callers should not rely on prompt-in-argv behavior.
    """
    from meta_agent.loop.codex_wrapper import build_command

    return build_command(model, cwd=get_workspace_root())


def _build_claude_cmd(
    prompt: str, system_append: str, model: Optional[str], max_turns: int,
) -> list[str]:
    """Build the `claude` CLI command.

    Routing (Bedrock vs Anthropic vs OpenRouter) is governed by the env applied
    in `_run_proposer_cli` via `agent_sdk_subprocess_env()`; the model id is
    resolved for the active provider so a non-Bedrock provider isn't handed a
    Bedrock inference-profile id.
    """
    from meta_agent.services.llm import resolve_model_for_provider

    permission_mode = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions").strip()
    if model:
        model = resolve_model_for_provider(model)
    cmd = [
        "claude",
        "--print", "--verbose",
        "--output-format", "stream-json",
        "--append-system-prompt", system_append,
        "--allowedTools", "Read,Write,Edit,Bash,Glob,Grep",
        "--max-turns", str(max_turns),
        "-p", prompt,
    ]
    if model:
        cmd.extend(["--model", model])
    if permission_mode:
        cmd.extend(["--permission-mode", permission_mode])
    return cmd


_DEFAULT_PROPOSER_STALL_TIMEOUT_S = 600
"""How long the proposer subprocess may sit without emitting any stdout line
before we consider it hung and force-kill it. 10 min is generous — a healthy
proposer emits a `command_execution`/`agent_message` event every few seconds.
Override via the `META_AGENT_PROPOSER_STALL_TIMEOUT_S` env var."""

_DEFAULT_PROPOSER_TOTAL_TIMEOUT_S = 3600
"""Hard wall-clock timeout for proposer sessions."""


def _print_proposer_stream_event(
    event: dict,
    *,
    label: str,
    captured: dict,
    log_all_commands: bool,
) -> None:
    """Print compact progress for a Claude/Codex stream-json event."""
    event_type = event.get("type", "")
    if event_type == "assistant":
        content = event.get("message", {}).get("content", [])
        for block in content:
            if block.get("type") == "text":
                text = block["text"].strip()
                if text:
                    print(f"  [{label.upper()}] {text[:300]}")
    elif event_type == "result":
        turns = event.get("num_turns", 0)
        captured["num_turns"] = turns
        cost = event.get("total_cost_usd")
        if isinstance(cost, (int, float)):
            captured["cost_usd"] = float(cost)
        usage = event.get("usage") or {}
        if isinstance(usage, dict):
            captured["input_tokens"] = usage.get("input_tokens")
            captured["output_tokens"] = usage.get("output_tokens")
            captured["cache_read_tokens"] = usage.get("cache_read_input_tokens")
        cost_fmt = (
            f"${captured['cost_usd']:.4f}"
            if captured["cost_usd"] is not None else "$?.????"
        )
        print(
            f"  [{label.upper()}] Done — {turns} turns  "
            f"cost={cost_fmt}"
        )
    elif event_type == "item.completed":
        item = event.get("item", {})
        item_type = item.get("type", "")
        if item_type == "command_execution":
            cmd = item.get("command", "")[:120]
            ec = item.get("exit_code", "?")
            command_failed = (
                (isinstance(ec, int) and ec != 0)
                or (not isinstance(ec, int) and str(ec) != "0")
            )
            if log_all_commands or command_failed:
                print(f"  [{label.upper()} CMD({ec})] {cmd}")
        elif item_type == "file_change":
            paths = [
                c.get("path", "?").rsplit("/", 1)[-1]
                for c in item.get("changes", [])
            ]
            print(f"  [{label.upper()} FILE] {', '.join(paths)}")
        elif item_type == "agent_message":
            text = str(item.get("text", "")).strip()
            if text:
                print(f"  [{label.upper()}] {text[:300]}")
    elif event_type == "error":
        message = str(event.get("message", "")).strip()
        if message:
            print(f"  [{label.upper()} ERROR] {message[:500]}")
    elif event_type == "turn.failed":
        error = event.get("error") or {}
        if isinstance(error, dict):
            message = str(error.get("message", "")).strip()
        else:
            message = str(error).strip()
        if message:
            print(f"  [{label.upper()} FAILED] {message[:500]}")
    sys.stdout.flush()


def _run_proposer_cli(
    prompt: str,
    system_append: str,
    label: str,
    cli: str = "claude",
    trace_path: Optional[Path] = None,
    session_dir: Optional[Path] = None,
    staging_dir: Optional[Path] = None,
    model_instructions: str = "",
    max_turns: int = DEFAULT_PROPOSER_MAX_TURNS,
    model: Optional[str] = None,
    target: Optional[AgentTarget] = None,
) -> ProposerRunResult:
    """Run a proposer CLI with stream-json, print summaries, optionally save trace.

    Returns a :class:`ProposerRunResult` with the exit code and any cost /
    token / turn numbers captured from the final ``result`` stream-json event.
    If the subprocess stops emitting output for longer than the stall timeout,
    we kill it and return a non-zero exit code — the outer loop treats that
    like a failed proposer run. Any partial content the proposer already wrote
    to disk (e.g. `staging/harness.py`) stays intact; the loop will either
    promote or discard it based on validator success.
    """
    import threading
    import time

    # CLI-free proposer: drive the configured LLM provider directly. Needs no
    # `claude`/`codex` binary on PATH — the path used on the ASP fleet.
    if cli in {"inprocess", "api"}:
        if staging_dir is None or target is None:
            return ProposerRunResult(
                exit_code=1, cli=cli, model=model,
                stderr="in-process proposer requires staging_dir and target",
            )
        from meta_agent.loop.inprocess_proposer import run_inprocess_proposer

        return run_inprocess_proposer(
            prompt=prompt,
            system_append=system_append,
            staging_dir=staging_dir,
            target=target,
            model=model,
            trace_path=trace_path,
        )

    stall_timeout = float(
        os.environ.get(
            "META_AGENT_PROPOSER_STALL_TIMEOUT_S",
            str(_DEFAULT_PROPOSER_STALL_TIMEOUT_S),
        )
    )
    total_timeout = float(
        os.environ.get(
            "META_AGENT_PROPOSER_TOTAL_TIMEOUT_S",
            str(_DEFAULT_PROPOSER_TOTAL_TIMEOUT_S),
        )
    )

    logger.info(f"Invoking {label}...")
    sys.stdout.flush()

    # Captured from the final `result` stream-json event inside the reader
    # thread. Both codex and claude CLIs emit `total_cost_usd` and usage data
    # there; we propagate it back to the caller for proposer_cost.json.
    captured: dict = {
        "cost_usd": None,
        "num_turns": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
    }

    log_all_commands = os.environ.get(
        "META_AGENT_PROPOSER_LOG_COMMANDS", ""
    ).strip().lower() in {"1", "true", "yes", "all"}

    if cli == "codex":
        if os.environ.get("META_AGENT_CODEX_PROPOSER_ISOLATED", "").strip() == "1":
            run_result = _run_codex_proposer_isolated(
                prompt=prompt,
                model=model,
                trace_path=trace_path,
                session_dir=session_dir,
                staging_dir=staging_dir,
                model_instructions=model_instructions,
            )
            if run_result.stderr and run_result.exit_code != 0:
                logger.info(f"isolated proposer stderr: {run_result.stderr[:1000]}")
            return run_result

        from meta_agent.loop import codex_wrapper

        result = codex_wrapper.run(
            prompt=prompt,
            model=model,
            cwd=get_workspace_root(),
            trace_path=trace_path,
            stall_timeout_seconds=stall_timeout,
            timeout_seconds=total_timeout,
            on_event=lambda event: _print_proposer_stream_event(
                event,
                label=label,
                captured=captured,
                log_all_commands=log_all_commands,
            ),
        )
        run_result = ProposerRunResult(
            exit_code=result.exit_code,
            cost_usd=result.cost_usd,
            num_turns=result.num_turns,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cache_read_tokens=result.cache_read_tokens,
            model=model,
            cli=cli,
            stderr=result.stderr,
        )
        if result.stderr and result.exit_code != 0:
            logger.info(f"{label} stderr: {result.stderr[:1000]}")
        if trace_path is not None and session_dir is not None:
            try:
                write_proposer_session_log(
                    trace_path=trace_path,
                    session_dir=session_dir,
                    prompt=prompt,
                    cli=cli,
                    model=model,
                    cwd=get_workspace_root(),
                    exit_code=run_result.exit_code,
                    cost_usd=run_result.cost_usd,
                    num_turns=run_result.num_turns,
                    input_tokens=run_result.input_tokens,
                    output_tokens=run_result.output_tokens,
                    cache_read_tokens=run_result.cache_read_tokens,
                    command=result.command,
                    stderr=result.stderr,
                )
            except Exception as exc:  # noqa: BLE001 - logging must not break proposal flow
                logger.warning(f"Could not write proposer session log: {type(exc).__name__}: {exc}")
        return run_result

    cmd = _build_claude_cmd(prompt, system_append, model, max_turns)

    def _finalize(exit_code: int) -> ProposerRunResult:
        result = ProposerRunResult(exit_code=exit_code, cli=cli, model=model, **captured)
        if trace_path is not None and session_dir is not None:
            try:
                write_proposer_session_log(
                    trace_path=trace_path,
                    session_dir=session_dir,
                    prompt=prompt,
                    cli=cli,
                    model=model,
                    cwd=get_workspace_root(),
                    exit_code=result.exit_code,
                    cost_usd=result.cost_usd,
                    num_turns=result.num_turns,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_read_tokens=result.cache_read_tokens,
                    command=cmd,
                )
            except Exception as exc:  # noqa: BLE001 - logging must not break proposal flow
                logger.warning(f"Could not write proposer session log: {type(exc).__name__}: {exc}")
        return result

    trace_file = open(trace_path, "w") if trace_path else None
    try:
        from meta_agent.services.llm import agent_sdk_subprocess_env
        subprocess_env = agent_sdk_subprocess_env() if cli != "codex" else None
        process = subprocess.Popen(
            cmd,
            cwd=str(get_workspace_root()),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=subprocess_env,
        )

        last_activity = [time.time()]
        reader_done = [False]
        trace_write_failed = [False]

        def _consume_stdout() -> None:
            # NOTE: This runs in a daemon thread. Don't raise — we just drain.
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    line = line.rstrip()
                    last_activity[0] = time.time()
                    if not line:
                        continue
                    if trace_file and not trace_file.closed and not trace_write_failed[0]:
                        try:
                            trace_file.write(line + "\n")
                            trace_file.flush()
                        except (OSError, ValueError) as exc:
                            trace_write_failed[0] = True
                            logger.warning(
                                f"Could not write proposer trace {trace_path}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                    try:
                        event = json.loads(line)
                        _print_proposer_stream_event(
                            event,
                            label=label,
                            captured=captured,
                            log_all_commands=log_all_commands,
                        )
                    except json.JSONDecodeError:
                        pass
            finally:
                reader_done[0] = True

        reader = threading.Thread(target=_consume_stdout, daemon=True)
        reader.start()

        while not reader_done[0]:
            idle = time.time() - last_activity[0]
            if idle > stall_timeout:
                logger.warning(
                    f"{label} stalled for {idle:.0f}s "
                    f"(> {stall_timeout:.0f}s timeout), killing subprocess."
                )
                process.kill()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    logger.warning(f"{label} did not exit after SIGKILL; abandoning.")
                # Let the reader thread drain whatever remains, then exit.
                reader.join(timeout=10)
                exit_code = process.returncode if process.returncode is not None else 1
                return _finalize(exit_code)
            time.sleep(5)

        reader.join(timeout=30)
        rc = process.wait()
        return _finalize(rc)
    finally:
        if trace_file:
            trace_file.close()


def _run_codex_proposer_isolated(
    *,
    prompt: str,
    model: Optional[str],
    trace_path: Optional[Path],
    session_dir: Optional[Path],
    staging_dir: Optional[Path],
    model_instructions: str = "",
) -> ProposerRunResult:
    """Delegate Codex proposing to a Modal function with only the volume mounted."""
    try:
        import modal
    except ImportError as exc:
        return ProposerRunResult(
            exit_code=1,
            stderr=f"Modal is not available for isolated Codex proposer: {exc}",
            model=model,
            cli="codex",
        )

    app_name = os.environ.get("META_AGENT_APP_NAME", "meta-agent").strip() or "meta-agent"
    trace_rel = _experience_volume_rel(trace_path) if trace_path is not None else None
    staging_rel = _experience_volume_rel(staging_dir) if staging_dir is not None else None
    kwargs = {
        "prompt": prompt,
        "model": model,
        "trace_rel": trace_rel,
        "staging_rel": staging_rel,
        "model_instructions": model_instructions or None,
        "launch_id": os.environ.get("META_AGENT_MODAL_LAUNCH_ID"),
        "tool_output_token_limit": os.environ.get("CODEX_TOOL_OUTPUT_TOKEN_LIMIT"),
    }
    try:
        direct_exc: Optional[BaseException] = None
        try:
            from meta_agent.cloud import modal_runner

            payload = modal_runner._run_codex_proposer_isolated.remote(**kwargs)
        except Exception as exc:  # noqa: BLE001 - fall back to deployed lookup
            direct_exc = exc
            fn = modal.Function.from_name(app_name, "_run_codex_proposer_isolated")
            payload = fn.remote(**kwargs)
        if direct_exc is not None:
            logger.info(
                "Direct isolated Codex function call failed, but deployed lookup "
                f"succeeded: {type(direct_exc).__name__}: {direct_exc}"
            )
    except Exception as exc:  # noqa: BLE001 - infrastructure failure should not crash loop
        return ProposerRunResult(
            exit_code=1,
            stderr=f"Isolated Codex proposer failed: {type(exc).__name__}: {exc}",
            model=model,
            cli="codex",
        )

    run_result = ProposerRunResult(
        exit_code=int(payload.get("exit_code", 1)),
        cost_usd=payload.get("cost_usd"),
        num_turns=payload.get("num_turns"),
        input_tokens=payload.get("input_tokens"),
        output_tokens=payload.get("output_tokens"),
        cache_read_tokens=payload.get("cache_read_tokens"),
        model=model,
        cli="codex",
        stderr=str(payload.get("stderr") or ""),
    )
    stderr = run_result.stderr
    if stderr and run_result.exit_code != 0:
        logger.info(f"isolated proposer stderr: {stderr[:1000]}")
    staged_files = payload.get("staged_files") or {}
    if staging_dir is not None and isinstance(staged_files, dict):
        for rel_path, content in staged_files.items():
            if not isinstance(rel_path, str) or not isinstance(content, str):
                continue
            dest = staging_dir / rel_path
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content)
            except OSError as exc:
                logger.warning(
                    f"Could not materialize isolated staged file {rel_path}: "
                    f"{type(exc).__name__}: {exc}"
                )
    trace_text = payload.get("trace_text")
    if trace_path is not None and isinstance(trace_text, str):
        try:
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(trace_text)
        except OSError as exc:
            logger.warning(
                f"Could not materialize isolated proposer trace: "
                f"{type(exc).__name__}: {exc}"
            )
    if trace_path is not None and session_dir is not None:
        try:
            write_proposer_session_log(
                trace_path=trace_path,
                session_dir=session_dir,
                prompt=prompt,
                cli="codex",
                model=model,
                cwd=get_workspace_root(),
                exit_code=run_result.exit_code,
                cost_usd=run_result.cost_usd,
                num_turns=run_result.num_turns,
                input_tokens=run_result.input_tokens,
                output_tokens=run_result.output_tokens,
                cache_read_tokens=run_result.cache_read_tokens,
                command=list(payload.get("command") or []),
                stderr=stderr,
            )
        except Exception as exc:  # noqa: BLE001 - logging must not break proposal flow
            logger.warning(f"Could not write proposer session log: {type(exc).__name__}: {exc}")
    return run_result


def _proposer_session_dir(experience_dir: Path, trace_path: Optional[Path]) -> Optional[Path]:
    if trace_path is None:
        return None
    run_root = experience_dir.parent
    name = trace_path.parent.name
    if name in {"proposal_checkpoints", "_internal"}:
        name = trace_path.stem.removesuffix("_proposer_trace")
    elif name == "candidates":
        name = trace_path.stem
    return run_root / "proposer_sessions" / name


def _codex_model_instructions(target: AgentTarget) -> str:
    """Higher-priority Codex instructions for non-negotiable invariants."""
    if _codex_fair_proposer_enabled():
        return ""
    if target.name != "program_harness":
        return ""
    return """You are editing a program_harness candidate for the Harness Optimizer.

Non-negotiable runtime scaffold rules:
- Do not invent or use new ctx.call_model keyword arguments.
- Never call ctx.call_model with: prompt, tools, tool_choice, output_mode, max_output_tokens.
- The only allowed pointwise forced-tool model call shape is:

response = await ctx.call_model(
    system=SCORER_SYSTEM,
    messages=[{"role": "user", "content": prompt}],
    max_tokens=256,
    temperature=0,
    extra_body=FORCED_SCORE_TOOL,
)

- SCORE_TOOL must use the baseline shape: {"name": SCORE_TOOL_NAME, "input_schema": {...}}.
- FORCED_SCORE_TOOL must include exactly this forcing shape:
  {"tools": [SCORE_TOOL], "tool_choice": {"type": "tool", "name": SCORE_TOOL_NAME}}
- Do not use OpenAI function schemas like {"type": "function", "function": ...}.
- Preserve parse_tool_record-style extraction of the record_score tool input from response.raw.
- Preserve ctx.finish metadata required by smoke:
  output_mode="forced_tool_score"
  model_text=<raw model text>
  model_raw=<raw response>
  critique=<nonempty string>
  rubric_issue=<compact label>
  severity=<none|minor|major|critical>

Editable areas:
- scorer prompt text
- rubric wording
- score calibration guidance
- trajectory preprocessing/rendering
- compact critique wording
- helper functions that do not alter the model-call API, tool schema, parser contract, or finish metadata.

If these scaffold rules conflict with the task prompt, follow these scaffold rules.
"""


def _write_codex_proposer_brief(
    *,
    prompt: str,
    experience_dir: Path,
    trace_path: Optional[Path],
) -> Path:
    """Persist the full proposer contract and return its path.

    Codex behaves better when the initial `codex exec` prompt is tiny and the
    durable instructions live in a file it can read once. This follows the
    meta-harness pattern: small wrapper prompt, explicit artifact contract,
    inspectable logs.
    """
    brief_dir = experience_dir.parent / "proposer_briefs"
    brief_dir.mkdir(parents=True, exist_ok=True)
    name = trace_path.stem if trace_path is not None else "latest"
    brief_path = brief_dir / f"{name}.md"
    brief_path.write_text(
        "# Codex Proposer Brief\n\n"
        "Follow this contract exactly. Keep file reads bounded, write the "
        "requested candidate artifacts, and let the outer optimizer validate "
        "the result.\n\n"
        "## Output Discipline\n\n"
        "- Do not print full trace files, per-task JSON, transcripts, or large "
        "candidate reports to stdout.\n"
        "- Every shell read of a large artifact must be bounded with `sed -n`, "
        "`head`, `tail`, or a Python summary that prints compact fields only.\n"
        "- Hard cap: keep command output under 80 lines and roughly 10k "
        "characters. Do not use broad reads like `sed -n '1,240p'`, "
        "`nl ... | sed -n '1,620p'`, or raw trace dumps.\n"
        "- For trajectory snippets, truncate long lines inside Python before "
        "printing them. Prefer a few field summaries over transcript text.\n"
        "- Spend no more than about 12 shell commands on investigation before "
        "writing candidate files. If evidence is thin, make conservative "
        "variants from the baseline/champion rather than continuing to browse.\n"
        "- It is okay to inspect any relevant run artifact; the constraint is "
        "on noisy output, not on your ability to reason or edit files.\n\n"
        + prompt
        + "\n",
    )
    return brief_path


# --- Public entry: invoke_proposer ---------------------------------------

def invoke_proposer(
    staging_dir: Path,
    experience_dir: Path,
    bench_name: str,
    trace_path: Optional[Path] = None,
    model: Optional[str] = None,
    harness: str = "claude_agent_sdk",
    proposer_cli: str = "claude",
    max_turns: int = DEFAULT_PROPOSER_MAX_TURNS,
    parent_seed_dir: Optional[Path] = None,
    surface_lock: Optional[str] = None,
    candidates_per_iter: int = 1,
    benchmark_path: Optional[str] = None,
) -> ProposerInvocationResult:
    """Ask the proposer CLI to write a new harness into `staging_dir`.

    When `candidates_per_iter > 1`, the prompt instructs the proposer to write
    each candidate to `staging/<descriptive_name>/harness.py` (rather than
    `staging/harness.py` directly). The loop accepts either a structured
    `staging/pending_eval.json` manifest or falls back to listing subdirectories.

    Returns a :class:`ProposerInvocationResult` with ``success=True`` iff
    the proposer wrote at least one required file (and any surface-lock
    constraint was satisfied). The ``run`` attribute carries the proposer
    CLI's exit code + captured cost/turn telemetry for sidecar persistence.
    """
    target = get_target(harness)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for item in staging_dir.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()

    if parent_seed_dir:
        if parent_seed_dir.is_dir():
            _copy_seed_harness_files(parent_seed_dir, staging_dir, target)
            logger.info(f"Seeded staging from {rel_to_workspace(parent_seed_dir)}")
        else:
            print(
                f"[LOOP] WARNING: parent seed dir missing: "
                f"{rel_to_workspace(parent_seed_dir)}"
            )

    if target.name == "codex" and surface_lock and surface_lock != "agents":
        if not (staging_dir / "AGENTS.md").exists():
            print(
                f"[LOOP] Surface lock '{surface_lock}' requires a seeded AGENTS.md. "
                "Run with --baseline <baseline-harness-path> (or resume from prior candidates)."
            )
            return ProposerInvocationResult(success=False, run=None)

    before_signatures = _collect_file_signatures(staging_dir)

    codex_restricted_mode = proposer_cli == "codex" and not _codex_fair_proposer_enabled()
    exp_rel = _codex_visible_path(experience_dir) if codex_restricted_mode else rel_to_workspace(experience_dir)
    run_root_rel = (
        _codex_visible_path(experience_dir.parent)
        if codex_restricted_mode
        else rel_to_workspace(experience_dir.parent)
    )
    staging_rel = _codex_visible_path(staging_dir) if codex_restricted_mode else rel_to_workspace(staging_dir)
    skill_path = PROPOSER_INSTRUCTIONS_DIR / target.skill_filename
    skill_rel = rel_to_workspace(skill_path)
    output_instruction = target.proposer_output_instruction.format(staging=staging_rel)

    benchmark_context_rel = ""
    benchmark_context_excerpt = ""
    proposer_context_instruction = ""
    if benchmark_path:
        from meta_agent.core.benchmark import parse_benchmark_ref
        yaml_str, _ = parse_benchmark_ref(benchmark_path)
        context_path = Path(yaml_str).parent / "proposer_context.md"
        if context_path.is_file():
            ctx_rel = rel_to_workspace(context_path)
            benchmark_context_rel = ctx_rel
            benchmark_context_excerpt = "\n".join(
                context_path.read_text().splitlines()[:160]
            )
            proposer_context_instruction = (
                f" Read `{ctx_rel}` immediately after `{skill_rel}` — "
                f"it contains benchmark-specific guidance."
            )

    surface_lock_instruction = ""
    if surface_lock:
        lock_desc = SURFACE_LOCK_DESCRIPTION.get(surface_lock, surface_lock)
        surface_lock_instruction = (
            f" Surface lock is active: only modify {lock_desc}. "
            "You may also update proposal_notes.json. "
            "Any other changed file path invalidates this iteration."
        )

    multi_candidate_instruction = ""
    if candidates_per_iter > 1:
        portfolio_instruction = (
            " Portfolio policy: reserve the slate for complementary risk. "
            "Include a mix of conservative mutations, evidence-backed "
            "recombinations, and exploratory candidates when the available "
            "slate size and prior evidence support them. Do not spend the "
            "whole slate on parallel rewrites of the same prompt idea. For "
            "each candidate, write down the expected gains and likely "
            "regressions by bucket or task family in proposal_notes.json."
        )
        if candidates_per_iter >= 5:
            portfolio_instruction += (
                " For a 5-candidate slate, use this concrete allocation: "
                "two small champion mutations, two evidence-backed recombinations "
                "from complementary prior candidates, and one radical candidate "
                "that changes a different mechanism from the champion."
            )
        multi_candidate_instruction = (
            f" IMPORTANT: produce {candidates_per_iter} distinct candidate harnesses "
            f"this iteration. Write each to `{staging_rel}/<descriptive_name>/harness.py` "
            f"(pick a short descriptive subdir name per candidate, e.g. "
            f"`ablate_prompt`, `add_hooks`, `combined`). Do NOT write "
            f"`{staging_rel}/harness.py` directly. Plan complementary hypotheses "
            f"across DIFFERENT lever axes — do not propose N variants of the "
            f"same change. Write `proposal_notes.json` inside each candidate "
            f"subdir (not at the staging root) with that candidate's "
            f"hypothesis, axis, rationale, risks. Also write "
            f"`{staging_rel}/pending_eval.json` with a `candidates` list; each "
            f"entry should include `name`, `subdir`, `hypothesis`, and "
            f"`risk_level` for one staged candidate.{portfolio_instruction}"
        )

    proposal_notes_instruction = ""
    if candidates_per_iter == 1:
        proposal_notes_instruction = (
            f" If possible, also write {staging_rel}/proposal_notes.json with keys: "
            f"hypothesis, lever, inspected_tasks, rationale, risks."
        )

    runtime_api_instruction = ""
    if target.name == "program_harness":
        runtime_api_instruction = (
            "## Do Not Change This Runtime API Shape\n\n"
            "For program pointwise reward harnesses, preserve the baseline "
            "`ctx.call_model` call shape exactly. Tool forcing is passed via "
            "`extra_body`, not as top-level OpenAI-style kwargs:\n\n"
            "```python\n"
            "FORCED_SCORE_TOOL = {\n"
            "    \"tools\": [SCORE_TOOL],\n"
            "    \"tool_choice\": {\"type\": \"tool\", \"name\": SCORE_TOOL_NAME},\n"
            "}\n\n"
            "response = await ctx.call_model(\n"
            "    system=SCORER_SYSTEM,\n"
            "    messages=[{\"role\": \"user\", \"content\": prompt}],\n"
            "    max_tokens=256,\n"
            "    temperature=0,\n"
            "    extra_body=FORCED_SCORE_TOOL,\n"
            ")\n"
            "```\n\n"
            "Forbidden `ctx.call_model` kwargs: `prompt`, `tools`, "
            "`tool_choice`, `output_mode`, `max_output_tokens`. Do not use "
            "OpenAI-style function tool schemas like "
            "`{\"type\": \"function\", \"function\": ...}`. Use the baseline "
            "tool schema shape: `{\"name\": SCORE_TOOL_NAME, "
            "\"input_schema\": {...}}` inside `FORCED_SCORE_TOOL`.\n\n"
            "Safe edit areas: scorer prompt text, rubric wording, score "
            "calibration, trajectory preprocessing/rendering, critique wording, "
            "constants, and helper functions that do not change the runtime API. "
            "Risky edit areas: `ctx.call_model`, `SCORE_TOOL`, "
            "`FORCED_SCORE_TOOL`, `parse_tool_record`, and `ctx.finish` "
            "metadata. Only touch risky areas when preserving the baseline API "
            "shape exactly.\n\n"
        )

    if codex_restricted_mode:
        experience_navigation = (
            f"Read `{run_root_rel}/candidate_index.json` first if it exists. "
            "Then inspect a small bounded set of files directly: scores, "
            "category summaries, proposal notes, the current accepted best "
            "harness, one or two nearby regressions only if they already exist, "
            "and only a few compact per-task/trace snippets. If only baseline "
            "exists, inspect only baseline scores/harness plus the benchmark "
            "context and then write candidates. Never dump full trace files or full "
            "per-task JSON to stdout; use Python summaries or bounded "
            "`sed -n` slices instead. Do not run "
            "`meta-agent list`, `meta-agent show`, `meta-agent failures`, or "
            "`meta-agent diff` inside the Modal proposer; those helper commands "
            "can become long-running in this environment and waste the epoch. "
            "Do not inspect, kill, or manage system processes. Keep shell "
            "exploration brief and bounded, then write the candidate files."
        )
    else:
        experience_navigation = (
            f"Use `meta-agent list --dir {exp_rel}` to see prior candidates. "
            f"Use `meta-agent show <name> --dir {exp_rel}` or "
            f"`meta-agent failures <name> --dir {exp_rel}` for details. "
            f"Use `meta-agent diff <name1> <name2> --dir {exp_rel}` to see "
            f"which tasks flipped between two candidates — this is the fastest "
            f"way to isolate what a change actually affected."
        )

    if codex_restricted_mode:
        context_hint = (
            "Benchmark context excerpt:\n"
            "```markdown\n"
            f"{benchmark_context_excerpt}\n"
            "```\n"
            if benchmark_context_rel else
            "No benchmark-specific proposer_context.md was found. "
        )
        prompt = (
            f"You are in the workspace root. You are optimizing benchmark `{bench_name}`. "
            f"The harness target is `{target.name}` and required candidate "
            f"file(s) are: {', '.join(target.required_written_files)}. "
            "Use this message as the primary contract. Do not read "
            f"`{skill_rel}`, `meta_agent/proposer_instructions/shared.md`, "
            "or other long instruction files unless you are blocked; if you "
            "must read them, read at most 80 lines. "
            f"{context_hint}"
            "\n\n## Program Harness Contract Summary\n\n"
            "- Candidate harnesses live in `harness.py` and should be readable, "
            "single-file programs.\n"
            "- Export `async def run(ctx)` or `class Harness` with "
            "`async def run(self, ctx)`.\n"
            "- The benchmark provides safe `ctx.task`, `ctx.call_model(...)`, "
            "`ctx.log_event(...)`, and `ctx.finish(...)` APIs.\n"
            "- Do not modify adapters, scorers, labels, split manifests, eval "
            "runners, Modal/runtime files, hidden holdout plumbing, or "
            "`_internal/` state.\n"
            "- Do not branch on task IDs, trajectory IDs, split membership, "
            "actor model names, hidden labels, or validation/test examples.\n"
            "- For pointwise reward harnesses, keep the pointwise scalar "
            "interface: score one trajectory at a time and return a scalar "
            "score through `ctx.finish(score, score=score, ...)`. Do not build "
            "a direct A/B judge unless the benchmark context explicitly says "
            "the target is pairwise.\n\n"
            f"{runtime_api_instruction}"
            f"Experience store: `{exp_rel}/`. {experience_navigation} "
            "Only inspect candidates in THIS benchmark's experience store. "
            "Do not inspect source files, tuned scorers, harnesses, benchmark "
            "directories, or prior run roots outside this experience store. "
            "Decide whether to start from scratch, copy-and-modify a prior "
            "candidate, or fuse ideas from several. "
            f"Then {output_instruction}.{surface_lock_instruction}"
            f"{multi_candidate_instruction}"
            f"{proposal_notes_instruction}"
            " The compact validation command you run is syntax/layout only; "
            "do not assume `py_compile` proves runtime API compatibility. "
            "The outer optimizer owns runtime smoke testing."
        )
    else:
        prompt = (
            f"You are in /repo. "
            f"Read `{skill_rel}` first, then follow its instructions."
            f"{proposer_context_instruction} "
            f"You are optimizing for the '{bench_name}' benchmark. "
            f"The experience store for this benchmark is at '{exp_rel}/'. "
            f"{experience_navigation} "
            f"Only inspect candidates in THIS benchmark's experience store — "
            f"do not list or read other benchmarks. "
            f"Inspect source code and execution traces across multiple prior candidates — "
            f"including ones that regressed. Comparing siblings is how you isolate "
            f"confounded edits and avoid anchoring on a noisy champion. "
            f"Decide whether to start from scratch, copy-and-modify a prior candidate, "
            f"or fuse ideas from several. "
            f"Then {output_instruction}.{surface_lock_instruction}"
            f"{multi_candidate_instruction}"
            f"{proposal_notes_instruction}"
        )
    system_append = f"Read {skill_path} for your full instructions."
    if proposer_cli == "codex":
        brief_path = _write_codex_proposer_brief(
            prompt=prompt,
            experience_dir=experience_dir,
            trace_path=trace_path,
        )
        brief_rel = rel_to_workspace(brief_path)
        prompt = (
            f"{prompt}\n\n"
            "A copy of this contract has already been saved for audit at "
            f"`{brief_rel}`. Do not read or cat that file during this proposer "
            "session; follow the contract above directly. After writing the "
            "requested staged files and running one compact validation command, "
            "stop with a short final summary."
        )
        system_append = f"Read {brief_rel} for the full proposer contract."

    run = _run_proposer_cli(
        prompt=prompt,
        system_append=system_append,
        label="proposer",
        cli=proposer_cli,
        trace_path=trace_path,
        session_dir=_proposer_session_dir(experience_dir, trace_path),
        staging_dir=staging_dir,
        model_instructions=_codex_model_instructions(target),
        max_turns=max_turns,
        model=model,
        target=target,
    )
    complete_staged_count = _complete_staged_candidate_count(
        staging_dir, target, candidates_per_iter,
    )
    wrote_any_required = complete_staged_count > 0
    if run.exit_code != 0:
        logger.info(f"Proposer exited with code {run.exit_code}")
        if not wrote_any_required:
            return ProposerInvocationResult(success=False, run=run)
        if candidates_per_iter > 1 and complete_staged_count < candidates_per_iter:
            logger.warning(
                "Proposer exited non-zero after writing only "
                f"{complete_staged_count}/{candidates_per_iter} complete "
                "candidate(s); refusing to salvage a partial slate."
            )
            return ProposerInvocationResult(success=False, run=run)
        logger.warning(
            "Proposer exited non-zero after writing complete candidate files; "
            "salvaging staging output and continuing to validation."
        )

    if not wrote_any_required:
        required_list = ", ".join(target.required_written_files)
        logger.info(f"Proposer did not write any of: {required_list} in {staging_dir}")
        return ProposerInvocationResult(success=False, run=run)
    logger.info(f"Proposer wrote {target.name} harness to {staging_dir}")

    if surface_lock:
        after_signatures = _collect_file_signatures(staging_dir)
        changed_files = _diff_file_signatures(before_signatures, after_signatures)
        violations = _surface_lock_violations(target, changed_files, surface_lock)
        if violations:
            print(
                f"[LOOP] Surface lock '{surface_lock}' violated by files: "
                + ", ".join(violations)
            )
            return ProposerInvocationResult(success=False, run=run)
        touched_target = any(
            target.is_surface_target_path(path, surface_lock) for path in changed_files
        )
        if not touched_target:
            logger.info(f"Surface lock '{surface_lock}' made no target-surface change.")
            return ProposerInvocationResult(success=False, run=run)
        print(
            f"[LOOP] Surface lock '{surface_lock}' satisfied "
            f"({len(changed_files)} changed file(s))"
        )

    return ProposerInvocationResult(success=True, run=run)
