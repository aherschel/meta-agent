"""Validation: is this artifact (config, evolved skill) safe to use?

Two functions:
- `validate_config` checks a proposer-written harness config can be loaded
  according to its harness target's contract.
- `validate_skill` sanity-checks evolved proposer instructions before they
  overwrite the on-disk version.

`validate_config` returns a `ValidationResult` (truthy on success). The
`error` field captures the reason for rejection so failed candidates can
be persisted with a human-readable diagnostic that the proposer reads
next iteration.
"""
from __future__ import annotations

import json
import os
import re
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Optional

from meta_agent.utils.logging import get_logger
from meta_agent.loop.state import SHARED_PROPOSER_INSTRUCTIONS_PATH

logger = get_logger("loop")


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of `validate_config`. Truthy iff `ok`."""

    ok: bool
    error: Optional[str] = None
    traceback: Optional[str] = None

    def __bool__(self) -> bool:  # backward-compat with `if not validate_config(...)`
        return self.ok


def validate_config(
    config_path: Path, bench_type: str = "local", harness: Optional[str] = None,
) -> ValidationResult:
    """Validate a config. If `harness` is None, sniff it from `config_path`."""
    from meta_agent.core.targets import FILE_BASED_HARNESSES, TargetDetectionError, detect_target

    if harness is None:
        try:
            harness = detect_target(Path(config_path)).name
        except TargetDetectionError as exc:
            logger.info(f"FAIL: {exc}")
            return ValidationResult(ok=False, error=str(exc))

    logger.info(f"Validating {config_path} (type={bench_type}, harness={harness})...")

    if harness in FILE_BASED_HARNESSES:
        return _validate_file_based(config_path, harness)
    if harness == "claude_agent_sdk":
        return _validate_claude_agent_sdk(config_path)
    if harness == "research_single_file":
        return _validate_research_single_file(config_path)
    if harness == "program_harness":
        return _validate_program_harness(config_path, bench_type=bench_type)
    msg = f"unknown harness {harness!r}"
    logger.info(f"FAIL: {msg}")
    return ValidationResult(ok=False, error=msg)


def _validate_file_based(config_path: Path, harness: str) -> ValidationResult:
    config_dir = config_path if config_path.is_dir() else config_path.parent

    if harness == "codex":
        if not (config_dir / "AGENTS.md").exists():
            msg = f"No AGENTS.md found in {config_dir}"
            logger.info(f"FAIL: {msg}")
            return ValidationResult(ok=False, error=msg)
    elif harness == "claude_code":
        if not (config_dir / "CLAUDE.md").exists() and not (config_dir / "AGENTS.md").exists():
            msg = f"No CLAUDE.md (or AGENTS.md fallback) found in {config_dir}"
            logger.info(f"FAIL: {msg}")
            return ValidationResult(ok=False, error=msg)

    hooks_json = config_dir / ".codex" / "hooks.json"
    if hooks_json.exists():
        try:
            json.loads(hooks_json.read_text())
        except json.JSONDecodeError as e:
            msg = f".codex/hooks.json is not valid JSON: {e}"
            logger.info(f"FAIL: {msg}")
            return ValidationResult(ok=False, error=msg)

    codex_toml = config_dir / ".codex" / "config.toml"
    if codex_toml.exists():
        try:
            import tomllib as _tomllib  # type: ignore
            _tomllib.loads(codex_toml.read_text())
        except ModuleNotFoundError:
            pass
        except Exception as e:
            msg = f".codex/config.toml is not valid TOML: {e}"
            logger.info(f"FAIL: {msg}")
            return ValidationResult(ok=False, error=msg)

    logger.info(f"PASS: {harness} config is valid")
    return ValidationResult(ok=True)


def _validate_research_single_file(config_path: Path) -> ValidationResult:
    try:
        from meta_agent.harness_contracts.research import validate_research_harness
        validate_research_harness(config_path)
    except Exception as e:
        msg = f"research_single_file validation failed: {e}"
        logger.info(f"FAIL: {msg}")
        return ValidationResult(ok=False, error=msg, traceback=traceback.format_exc())
    logger.info("PASS: research_single_file harness is valid")
    return ValidationResult(ok=True)


def _validate_claude_agent_sdk(config_path: Path) -> ValidationResult:
    """Validate a `build_options(ctx) -> ClaudeAgentOptions` harness module."""
    try:
        from meta_agent.harness_contracts.claude_agent_sdk import validate_claude_agent_harness
        harness_file = config_path if config_path.is_file() else config_path / "harness.py"
        validate_claude_agent_harness(harness_file)
    except Exception as e:
        msg = f"claude_agent_sdk harness validation failed: {e}"
        logger.info(f"FAIL: {msg}")
        return ValidationResult(ok=False, error=msg, traceback=traceback.format_exc())
    logger.info("PASS: claude_agent_sdk harness is valid")
    return ValidationResult(ok=True)


def _validate_tau3_pointwise_program_contract(config_path: Path) -> Optional[str]:
    """Static guard for tau3 pointwise scorer scaffold invariants."""
    harness_file = config_path if config_path.is_file() else config_path / "harness.py"
    try:
        source = harness_file.read_text()
    except OSError as exc:
        return f"could not read program harness for static contract check: {exc}"

    forbidden_call = re.search(
        r"ctx\.call_model\s*\([^)]*\b(prompt|tools|tool_choice|output_mode|max_output_tokens)\s*=",
        source,
        flags=re.DOTALL,
    )
    if forbidden_call:
        return (
            "pointwise program harness must not pass "
            f"{forbidden_call.group(1)!r} directly to ctx.call_model; "
            "use system/messages/max_tokens/temperature/extra_body only"
        )

    if re.search(r"['\"]type['\"]\s*:\s*['\"]function['\"]", source):
        return (
            "pointwise program harness must not use OpenAI-style function tool "
            "schemas; use {'name': SCORE_TOOL_NAME, 'input_schema': {...}}"
        )

    required_snippets = [
        ("extra_body=FORCED_SCORE_TOOL", "ctx.call_model must pass extra_body=FORCED_SCORE_TOOL"),
        ("SCORE_TOOL_NAME", "missing SCORE_TOOL_NAME scaffold"),
        ("SCORE_TOOL", "missing SCORE_TOOL scaffold"),
        ("input_schema", "SCORE_TOOL must use input_schema"),
        ("FORCED_SCORE_TOOL", "missing FORCED_SCORE_TOOL scaffold"),
        ("parse_tool_record", "missing parse_tool_record scaffold"),
        ("model_raw=", "ctx.finish must expose model_raw for smoke validation"),
        ("model_text=", "ctx.finish must expose model_text for smoke validation"),
    ]
    for snippet, message in required_snippets:
        if snippet not in source:
            return message

    if (
        'output_mode="forced_tool_score"' not in source
        and "output_mode='forced_tool_score'" not in source
    ):
        return "ctx.finish must expose output_mode='forced_tool_score'"

    return None


def _validate_program_harness(config_path: Path, bench_type: str) -> ValidationResult:
    disable_tau3_guard = os.environ.get(
        "META_AGENT_DISABLE_TAU3_STATIC_GUARD", ""
    ).strip().lower() in {"1", "true", "yes"}
    if bench_type == "tau3_trajectory_judge" and not disable_tau3_guard:
        contract_error = _validate_tau3_pointwise_program_contract(config_path)
        if contract_error is not None:
            msg = f"program_harness static contract failed: {contract_error}"
            logger.info(f"FAIL: {msg}")
            return ValidationResult(ok=False, error=msg)
    try:
        from meta_agent.harness_contracts.program import validate_program_harness
        validate_program_harness(config_path)
    except Exception as e:
        msg = f"program_harness validation failed: {e}"
        logger.info(f"FAIL: {msg}")
        return ValidationResult(ok=False, error=msg, traceback=traceback.format_exc())
    logger.info("PASS: program_harness harness is valid")
    return ValidationResult(ok=True)


def validate_skill(skill_path: Path, reference_path: Optional[Path] = None) -> bool:
    """Sanity-check an evolved skill doc before promoting it."""
    if not skill_path.exists():
        logger.info("FAIL: Evolved skill file not found")
        return False

    content = skill_path.read_text()

    if len(content) < 200:
        logger.info("FAIL: Evolved skill is suspiciously short")
        return False

    required = ["harness", "proposal_notes"]
    for token in required:
        if token not in content:
            logger.info(f"FAIL: Evolved skill is missing required reference: {token}")
            return False

    comparison_path = reference_path or SHARED_PROPOSER_INSTRUCTIONS_PATH
    if comparison_path.exists():
        original_len = len(comparison_path.read_text())
        if original_len > 0 and len(content) > original_len * 2:
            logger.info(
                f"FAIL: Evolved skill is >2x the original size "
                f"({len(content)} vs {original_len} chars)"
            )
            return False

    logger.info("PASS: Evolved skill is valid")
    return True
