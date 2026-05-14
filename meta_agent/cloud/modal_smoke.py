"""Isolated smoke to pinpoint which `@anthropic-ai/claude-code` version broke
the old-style `{"decision": "block", ...}` hook-callback output shape.

This file is an ISOLATED diagnostic: it defines its own Modal app name
(`meta-agent-hook-smoke`), its own image, its own functions. It touches
nothing in the deployed `meta-agent` app or the `meta-agent-experience`
volume, so it is safe to run alongside any in-flight loop/eval.

Scope:
    For each candidate CLI pin (2.1.114 known-good, 2.1.116, 2.1.117 broken),
    spawn a container with that CLI installed, run a tiny claude-agent-sdk
    session that triggers a `PreToolUse` hook under two output shapes:

      * "old_decision_block" — {"decision": "block", "reason": "..."}
        (the shape currently baked into our baseline + every proposer
        harness on the volume)

      * "new_hookspecific" — {"hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "..."
        }}
        (the shape the 2.1.117 Zod schema appears to require)

    Each (version × shape) returns a single row telling us whether the hook
    fired, whether a ZodError appeared anywhere in the CLI subprocess stderr
    or in the parsed message stream, and the exact CLI version string.

Output: structured JSON printed by the `probe_all` local_entrypoint. Pin
selection is obvious from the table: the last version where
`old_decision_block` does not ZodError is the pin.

Usage:
    modal run meta_agent/cloud/modal_smoke.py::probe_all

Cost: <$1 total. Each probe is a single 2-message SDK session, bounded to
120s. Image builds are cached after first run.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import modal

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Deliberately different app name so this cannot collide with the deployed
# `meta-agent` app. Modal treats them as fully separate.
_SMOKE_APP = modal.App("meta-agent-hook-smoke")
app = _SMOKE_APP

# Same SDK-bundled-CLI strip-out as the main runner, so the SDK falls
# through to the Node-based `claude` binary on PATH.
_SDK_BUNDLED_CLAUDE = (
    "/usr/local/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"
)


def _image(cli_pin: str) -> modal.Image:
    """Image with exactly one pinned `@anthropic-ai/claude-code` version."""
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git", "curl", "ca-certificates", "gnupg")
        .run_commands(
            "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
            "apt-get install -y nodejs",
            f"npm install -g @anthropic-ai/claude-code@{cli_pin}",
        )
        .pip_install("claude-agent-sdk>=0.1.53", "pydantic>=2.0")
        .run_commands(f"rm -f {_SDK_BUNDLED_CLAUDE}")
        .env(
            {
                "CLAUDE_CODE_USE_BEDROCK": "1",
                # `bypassPermissions` requires --dangerously-skip-permissions
                # which is blocked when running as root (Modal default). The
                # main runner uses `acceptEdits` for this exact reason.
                "CLAUDE_PERMISSION_MODE": "acceptEdits",
            }
        )
    )


_SECRETS = [modal.Secret.from_name("bedrock-creds")]


# Probe body runs in-container. Tests the SDK callback-hook path — the path
# the live k10/k5/k2 harnesses use (Python function returned via SDK's
# in-process hooks={...} option). A separate command-type probe (below)
# sanity-checks the settings.json path for comparison, because a previous
# iteration showed command-type hooks are UNAFFECTED — only the callback
# control-protocol schema is strict in 2.1.117.
_PROBE_SCRIPT = r"""
import asyncio, json, os, subprocess, sys, traceback

# Force full traceback on subprocess errors so we see any captured stderr.
os.environ.setdefault("CLAUDE_CODE_LOG_LEVEL", "debug")

from claude_agent_sdk import (
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    create_sdk_mcp_server,
    tool,
)

CLI_VER = subprocess.check_output(["claude", "--version"], text=True).strip()
BEDROCK_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


# Minimal MCP tool that mirrors the live `submit_verdict` contract — this
# is the tool our judge harnesses gate on in production. Live harnesses use
# hooks fired around MCP-backed tool calls; my previous Bash-only probe
# missed that axis entirely.
@tool(
    "submit_verdict",
    "Submit the pairwise verdict",
    {"choice": str, "rationale": str},
)
async def _submit_verdict(args):
    return {"content": [{"type": "text", "text": "ack"}]}


_MCP_SERVER = create_sdk_mcp_server(
    name="judge", version="0.0.1", tools=[_submit_verdict]
)
_MCP_TOOL = "mcp__judge__submit_verdict"

# Each test is a (PreToolUse_shape, PostToolUse_shape) pair. The live
# sdk_judge_runner attaches BOTH kinds of hooks: user's PreToolUse gate
# (returning {"decision": "block", ...}) AND runner's PostToolUse stop
# hook (returning {"continue_": False, "stopReason": "..."}). My earlier
# PreToolUse-only probe did NOT reproduce the live ZodError, so we test
# the combination here.
TESTS = {
    # Baseline: what the old probe tested — works in all versions.
    "pre_only_old": {
        "pre_hook": {"decision": "block", "reason": "pre-test"},
        "post_hook": None,
    },
    # sdk_judge_runner pattern: PostToolUse returning continue_/stopReason.
    "post_only_stopreason": {
        "pre_hook": None,
        "post_hook": {"continue_": False, "stopReason": "stop-test"},
    },
    # Combination mirroring the live crash surface most closely.
    "pre_old_plus_post_stopreason": {
        "pre_hook": {"decision": "block", "reason": "pre-test"},
        "post_hook": {"continue_": False, "stopReason": "stop-test"},
    },
    # Forward-compatible shape proposers should be moving toward.
    "pre_new_plus_post_stopreason": {
        "pre_hook": {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": "pre-test-new",
            },
        },
        "post_hook": {"continue_": False, "stopReason": "stop-test"},
    },
    # The EXACT broken shape from evo_003_post_verdict_review:
    # hookSpecificOutput with additionalContext but NO hookEventName.
    # Hypothesis: this shape fails ALL CLI versions — bug is in the
    # proposer-written harness, not in the CLI.
    "post_verdict_review_broken_shape": {
        "pre_hook": None,
        "post_hook": {
            "hookSpecificOutput": {
                "additionalContext": "reproducing the live bug shape"
            }
        },
    },
}


async def _run_one(test_name, test_cfg):
    row = {
        "test": test_name,
        "pre_shape": test_cfg["pre_hook"],
        "post_shape": test_cfg["post_hook"],
        "pre_fired": False,
        "post_fired": False,
        "zod_error_seen": False,
        "hook_stderr_seen": False,
        "messages_seen": 0,
        "message_types": [],
        "error": None,
        "error_snippet": None,
        "last_message_blob": None,
        "stderr_lines": 0,
        "stderr_snippet": None,
    }

    # CRITICAL: sdk_judge_runner captures CLI subprocess stderr via
    # options.stderr (see meta_agent.task_runner.judge_runner).
    # The ZodError from hook validation lands HERE, not in the message
    # stream. Our earlier probe missed it.
    stderr_lines = []
    def _stderr_capture(line: str):
        stderr_lines.append(line)
        if "ZodError" in line and not row["zod_error_seen"]:
            row["zod_error_seen"] = True
        if "Error in hook callback" in line and not row["hook_stderr_seen"]:
            row["hook_stderr_seen"] = True

    hooks_cfg = {}

    if test_cfg["pre_hook"] is not None:
        async def _pre_hook(input_data, tool_use_id, context):
            row["pre_fired"] = True
            return test_cfg["pre_hook"]
        hooks_cfg["PreToolUse"] = [HookMatcher(matcher=None, hooks=[_pre_hook])]

    if test_cfg["post_hook"] is not None:
        async def _post_hook(input_data, tool_use_id, context):
            row["post_fired"] = True
            return test_cfg["post_hook"]
        hooks_cfg["PostToolUse"] = [HookMatcher(matcher=None, hooks=[_post_hook])]

    opts = ClaudeAgentOptions(
        model=BEDROCK_MODEL,
        permission_mode="acceptEdits",
        allowed_tools=[_MCP_TOOL],
        mcp_servers={"judge": _MCP_SERVER},
        max_turns=5,
        hooks=hooks_cfg,
        stderr=_stderr_capture,
    )
    prompt = (
        "Call the submit_verdict tool with choice='A>B' and rationale='probe'. "
        "Do it immediately."
    )

    last_blob = None
    try:
        async with ClaudeSDKClient(options=opts) as client:
            await client.query(prompt)
            async for msg in client.receive_response():
                row["messages_seen"] += 1
                blob = json.dumps(msg, default=str)
                last_blob = blob
                mtype = type(msg).__name__
                if mtype not in row["message_types"]:
                    row["message_types"].append(mtype)
                if "ZodError" in blob and not row["zod_error_seen"]:
                    row["zod_error_seen"] = True
                    idx = blob.find("ZodError")
                    row["error_snippet"] = blob[max(0, idx-100) : idx + 600]
                if "hook_stderr" in blob and not row["hook_stderr_seen"]:
                    row["hook_stderr_seen"] = True
                    if not row["error_snippet"]:
                        idx = blob.find("hook_stderr")
                        row["error_snippet"] = blob[max(0, idx-100) : idx + 600]
                if row["messages_seen"] > 50:
                    break
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        if "ZodError" in tb and not row["zod_error_seen"]:
            row["zod_error_seen"] = True
        if row["error_snippet"] is None:
            row["error_snippet"] = tb[-1000:]
    finally:
        if last_blob:
            row["last_message_blob"] = last_blob[:1500]
        row["stderr_lines"] = len(stderr_lines)
        if stderr_lines:
            # Prefer an interesting line (ZodError/Error) over the first one.
            interesting = next(
                (l for l in stderr_lines if "ZodError" in l or "Error in hook" in l),
                stderr_lines[0],
            )
            row["stderr_snippet"] = interesting[:1500]
    return row


async def _main():
    import claude_agent_sdk as _sdk
    out = {
        "cli_version": CLI_VER,
        "sdk_version": getattr(_sdk, "__version__", "?"),
        "tests": {},
    }
    for name, cfg in TESTS.items():
        out["tests"][name] = await _run_one(name, cfg)
    print("PROBE_RESULT_JSON:" + json.dumps(out))


asyncio.run(_main())
"""


def _run_probe() -> dict:
    """Run the probe script in-container; extract the single JSON result line."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT],
        capture_output=True,
        text=True,
        timeout=300,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    for line in stdout.splitlines():
        if line.startswith("PROBE_RESULT_JSON:"):
            data = json.loads(line[len("PROBE_RESULT_JSON:") :])
            # Also attach any CLI-subprocess stderr we captured at the Python
            # level (ZodError often lands here in broken versions).
            if "ZodError" in stderr:
                data["subprocess_stderr_had_zoderror"] = True
                idx = stderr.find("ZodError")
                data["subprocess_stderr_snippet"] = stderr[idx : idx + 800]
            return data
    return {
        "cli_version": "unknown",
        "shapes": {},
        "error": "no PROBE_RESULT_JSON line",
        "stdout_tail": stdout[-1500:],
        "stderr_tail": stderr[-1500:],
    }


# One function per pinned CLI — the image is baked at decoration time so
# each has its own distinct, reproducible image.

@_SMOKE_APP.function(image=_image("2.1.114"), secrets=_SECRETS, timeout=600)
def probe_2_1_114() -> dict:
    return _run_probe()


@_SMOKE_APP.function(image=_image("2.1.116"), secrets=_SECRETS, timeout=600)
def probe_2_1_116() -> dict:
    return _run_probe()


@_SMOKE_APP.function(image=_image("2.1.117"), secrets=_SECRETS, timeout=600)
def probe_2_1_117() -> dict:
    return _run_probe()


@_SMOKE_APP.local_entrypoint()
def probe_all():
    """Fan out all version probes in parallel, print a summary table."""
    probes = [
        ("2.1.114", probe_2_1_114),
        ("2.1.116", probe_2_1_116),
        ("2.1.117", probe_2_1_117),
    ]
    calls = [(name, fn.spawn()) for name, fn in probes]
    print(f"Spawned {len(calls)} probes. Waiting...", flush=True)

    rows = []
    for name, call in calls:
        try:
            result = call.get()
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}", "shapes": {}}
        rows.append((name, result))

    print("\n" + "=" * 90)
    print("HOOK SHAPE COMPATIBILITY MATRIX")
    print("=" * 90)
    print(
        f"{'pin':<8} {'cli':<22} {'sdk':<10} {'test':<32} {'status'}"
    )
    print("-" * 90)
    for pin, result in rows:
        cli_ver = result.get("cli_version", "?")[:20]
        sdk_ver = str(result.get("sdk_version", "?"))[:8]
        tests = result.get("tests") or {}
        if not tests:
            err = result.get("error", "probe failed")
            print(f"{pin:<8} {cli_ver:<22} {sdk_ver:<10} {'-':<32} ERROR: {err}")
            continue
        for test_name, row_data in tests.items():
            pre_f = row_data.get("pre_fired")
            post_f = row_data.get("post_fired")
            zod = row_data.get("zod_error_seen")
            hstderr = row_data.get("hook_stderr_seen")
            err = row_data.get("error")
            if zod or hstderr:
                status = "BROKEN (ZodError)"
            elif err:
                status = f"ERROR: {str(err)[:28]}"
            elif not (pre_f or post_f):
                status = "?? no hook fired"
            else:
                parts = []
                if pre_f:
                    parts.append("pre")
                if post_f:
                    parts.append("post")
                status = f"WORKS ({'+'.join(parts)})"
            print(f"{pin:<8} {cli_ver:<22} {sdk_ver:<10} {test_name:<32} {status}")

    # Dump full JSON at the end for anything the table truncated.
    print("\n" + "=" * 78)
    print("RAW RESULTS (for debugging)")
    print("=" * 78)
    for pin, result in rows:
        print(f"\n---- pin={pin} ----")
        print(json.dumps(result, indent=2, default=str)[:3000])
