"""Run `meta-agent loop` / `meta-agent eval` on Modal in one long-running container.

This is the generic cloud runner — benchmark-agnostic. It bakes the full repo
into a Modal image with Python + Node.js + the `claude` / `codex` CLIs, then
invokes the standard `python -m meta_agent` entrypoint inside the container.
The loop's own semantics (adapters, experience store layout, proposer flow)
are unchanged — we're just hosting the same command in a bigger box.

Why this exists:

- Apple-Silicon Macs run the SDK's bundled x86_64 Claude CLI under Rosetta,
  which caps concurrency and cooks the laptop.
- Multi-hour loop runs shouldn't depend on your laptop being awake and
  thermally happy.
- Holdout evals on larger benchmark splits benefit from a dedicated machine
  with fast network to Bedrock.

Usage (short one-shot runs — ephemeral app per invocation):

    modal run meta_agent/cloud/modal_runner.py::loop \\
        --benchmark benchmarks/tau3/benchmark.yaml:search \\
        --holdout benchmarks/tau3/benchmark.yaml:holdout \\
        --baseline harnesses/claude_vanilla \\
        --run-name tau3-agent-run \\
        --iterations 10 \\
        --model claude-sonnet-4-6 \\
        --proposer-cli codex

    modal run meta_agent/cloud/modal_runner.py::eval \\
        --benchmark benchmarks/tau3/benchmark.yaml:holdout \\
        --config experience/tau3-agent-run/candidates/evo_007 \\
        --name transfer_eval \\
        --model claude-sonnet-4-6

Usage (long multi-hour runs — deploy once, then dispatch to persistent app):

    # One-time: deploy the app. Functions register with Modal's control plane
    # and stay visible across runs.
    modal deploy meta_agent/cloud/modal_runner.py

    # Each launch dispatches onto the deployed app with preemption-retry
    # semantics baked into the function decorator. Survives local client
    # disconnects / DNS blips cleanly, stays in `modal app list`.
    python -m meta_agent.cloud.modal_runner launch-loop \\
        --benchmark benchmarks/tau3_trajectory_judge/benchmark.yaml:judge-train \\
        --holdout   benchmarks/tau3_trajectory_judge/benchmark.yaml:judge-val \\
        --baseline  harnesses/claude_vanilla_tau3_trajectory_judge \\
        --run-name  tau3-judge-v1-k10 --iterations 20 \\
        --candidates-per-iter 10 --concurrency 5 --accept-on-holdout \\
        --model claude-haiku-4-5 --proposer-model claude-opus-4-6 --proposer-cli claude

After the run, pull results locally:

    modal volume get meta-agent-experience tau3-agent-run ./experience/

See `meta_agent/cloud/modal_runner.py::setup_codex_secret` for the one-time codex
auth upload.

Requires three Modal Secrets (create once, per account):

    modal secret create bedrock-creds \\
        AWS_BEARER_TOKEN_BEDROCK=<your-bedrock-bearer-token> \\
        AWS_REGION=us-east-1

    modal secret create openai-key OPENAI_API_KEY=<sk-...>

    python -m meta_agent.cloud.modal_runner upload-codex-auth   # reads ~/.codex/
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Optional

import modal


_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "cloud" and _THIS_FILE.parent.parent.name == "meta_agent":
    _REPO_ROOT = _THIS_FILE.parents[2]
else:
    # Modal copies this runner file to a flat path such as /root/modal_runner.py
    # when importing the deployed function. The full repository is mounted at
    # /repo in the image, so avoid indexing nonexistent parents there.
    _REPO_ROOT = Path(os.environ.get("META_AGENT_REPO_ROOT", "/repo"))

_MODAL_APP_NAME = os.environ.get("META_AGENT_APP_NAME", "meta-agent")
_EXPERIENCE_VOLUME_NAME = os.environ.get(
    "META_AGENT_EXPERIENCE_VOLUME", "meta-agent-experience"
)
_HF_CACHE_VOLUME_NAME = os.environ.get("META_AGENT_HF_CACHE_VOLUME", "meta-agent-hf-cache")

_app = modal.App(_MODAL_APP_NAME)
# `modal deploy` looks for a top-level `app` symbol by default. Keep the
# canonical private `_app` and expose a public alias so both deploy and
# run entry points (`modal deploy meta_agent/cloud/modal_runner.py` /
# `modal run meta_agent/cloud/modal_runner.py::loop`) work.
app = _app


def _new_launch_id(kind: str) -> str:
    """Stable id for one Modal input, preserved across Modal retries."""
    return f"{kind}-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Image: Python + Node.js + claude CLI + codex CLI + project deps
# ---------------------------------------------------------------------------
#
# We deliberately drop the claude-agent-sdk's bundled x86_64 Claude CLI so
# the SDK falls through to `shutil.which("claude")`, which resolves to the
# globally-installed `@anthropic-ai/claude-code` npm package we just put on
# PATH. That binary is a Node script — runs native on ARM64 Linux without
# Rosetta shenanigans, so we can push concurrency much higher than a Mac.

_SDK_BUNDLED_CLAUDE = (
    "/usr/local/lib/python3.12/site-packages/claude_agent_sdk/_bundled/claude"
)

_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "curl",
        "ca-certificates",
        "gnupg",
        "ripgrep",
        "jq",
        "patch",
        "diffutils",
    )
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @anthropic-ai/claude-code @openai/codex",
        # tau2-bench: both the dataset (airline/retail/telecom policy, tasks.json,
        # db.json, split_tasks.json) AND the Python package. The PyPI package
        # named `tau2` is a completely unrelated "magnetic relaxation rates"
        # calculator — we must install from this research clone instead.
        # Shallow clone keeps the image small; pin upstream by editing the
        # ref below if reproducibility against a specific tau2-bench SHA
        # matters more than picking up fixes.
        "git clone --depth 1 https://github.com/sierra-research/tau2-bench.git /opt/tau2-bench",
    )
    .pip_install(
        "claude-agent-sdk>=0.1.53",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "httpx>=0.27",
        "codex-app-server-sdk>=0.2.0",
        "datasets>=2.0",
        "boto3>=1.35",
        "six>=1.16",
    )
    # tau-bench v3 runtime: install from the research clone so `tau2.runner`,
    # `tau2.data_model`, `tau2.evaluator`, etc. all resolve correctly. The
    # PyPI `tau2` is a different project and does NOT work here.
    .run_commands("pip install /opt/tau2-bench")
    .run_commands(f"rm -f {_SDK_BUNDLED_CLAUDE}")
    .env(
        {
            "CLAUDE_CODE_USE_BEDROCK": "1",
            "CLAUDE_CODE_STREAM_CLOSE_TIMEOUT": "300000",
            "HF_HOME": "/root/.cache/huggingface",
            # Modal containers run as root, and `claude`'s
            # `--dangerously-skip-permissions` flag is blocked under root for
            # safety. `acceptEdits` covers every workflow in this repo (judge
            # benchmarks never write files; agentic benchmarks accept edits
            # freely on an ephemeral container). The vanilla harness + every
            # proposer-written harness that reads `CLAUDE_PERMISSION_MODE`
            # will pick this up automatically.
            "CLAUDE_PERMISSION_MODE": "acceptEdits",
            # Codex's default `--full-auto` mode wraps every command in bwrap
            # (a Linux namespace sandbox) which fails inside Modal containers
            # with `bwrap: loopback: Failed RTM_NEWADDR: No child process`.
            # The failure is silent — codex returns exit 0 but every file
            # write inside the tool-call router fails, so the proposer emits
            # a no-op harness identical to baseline. The Modal container is
            # already a disposable sandbox, so bypassing codex's inner one is
            # safe. Read by `loop.codex_wrapper.build_command`; flips codex to
            # `--dangerously-bypass-approvals-and-sandbox`.
            "CODEX_DANGEROUS_BYPASS": "1",
            "CODEX_TOOL_OUTPUT_TOKEN_LIMIT": os.environ.get(
                "CODEX_TOOL_OUTPUT_TOKEN_LIMIT", "4000"
            ),
            # Read by `tau2.utils.utils` at module import time to locate
            # airline/retail/telecom task + policy + db JSON. Point at the
            # tau2-bench clone we just baked in.
            "TAU2_DATA_DIR": "/opt/tau2-bench/data",
        }
    )
    .add_local_dir(
        str(_REPO_ROOT),
        "/repo",
        copy=True,
        ignore=[
            "**/__pycache__/**",
            "**/*.pyc",
            ".pytest_cache/**",
            ".git/**",
            "experience/**",
            ".venv/**",
            "venv/**",
            "ui/node_modules/**",
            "ui/.next/**",
            # Root-anchored: excludes only a top-level data/ dir if it exists.
            # Crucially, does NOT match nested data dirs like
            # `benchmarks/tau3_trajectory_judge/data/` (13 MB pool JSONL that
            # must ship into the image for trajectory-judge evals).
            "/data/**",
            "*.bak",
        ],
    )
    .run_commands("pip install -e /repo")
)


_CODEX_PROPOSER_IMAGE = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "gnupg", "ripgrep")
    .run_commands(
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash -",
        "apt-get install -y nodejs",
        "npm install -g @openai/codex",
    )
    .env(
        {
            "CODEX_DANGEROUS_BYPASS": "1",
            "CODEX_TOOL_OUTPUT_TOKEN_LIMIT": os.environ.get(
                "CODEX_TOOL_OUTPUT_TOKEN_LIMIT", "4000"
            ),
        }
    )
)


_EXPERIENCE_VOLUME = modal.Volume.from_name(
    _EXPERIENCE_VOLUME_NAME, create_if_missing=True
)
_HF_CACHE_VOLUME = modal.Volume.from_name(
    _HF_CACHE_VOLUME_NAME, create_if_missing=True
)


_SECRETS = [
    modal.Secret.from_name("bedrock-creds"),
    modal.Secret.from_name("openai-key"),
    modal.Secret.from_name("azure-openai"),
    modal.Secret.from_name("tau3-azure-openai"),
    modal.Secret.from_name("codex-auth"),
    modal.Secret.from_name("tinker-api-key"),
]


def _toml_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _maybe_configure_azure_provider(model: Optional[str]) -> tuple[Optional[str], list[str]]:
    if not model:
        return model, []

    requested = model
    deployment = model
    if model.startswith("azure:"):
        deployment = model.split(":", 1)[1]
    elif model.startswith("azure/"):
        deployment = model.split("/", 1)[1]
    else:
        azure_deployment = os.environ.get("AZURE_GPT55_DEPLOYMENT", "").strip()
        if not azure_deployment or model != azure_deployment:
            return model, []

    base_url = (
        os.environ.get("AZURE_OPENAI_V1_BASE", "").strip()
        or os.environ.get("AZURE_FOUNDRY_OPENAI_BASE", "").strip()
    )
    if not base_url:
        api_base = os.environ.get("AZURE_API_BASE", "").strip()
        if api_base:
            base_url = api_base.rstrip("/") + "/openai/v1"
    if not base_url:
        raise RuntimeError(
            f"Azure Codex proposer requested ({requested}) but no Azure OpenAI "
            "v1 base URL is configured. Set AZURE_OPENAI_V1_BASE or "
            "AZURE_FOUNDRY_OPENAI_BASE in the azure-openai secret."
        )

    env_key = "AZURE_API_KEY" if os.environ.get("AZURE_API_KEY") else "AZURE_FOUNDRY_API_KEY"
    return deployment, [
        "-c", 'model_provider="azure"',
        "-c", "model_providers.azure.name=" + _toml_string("Azure OpenAI"),
        "-c", "model_providers.azure.base_url=" + _toml_string(base_url.rstrip("/")),
        "-c", 'model_providers.azure.wire_api="responses"',
        "-c", "model_providers.azure.env_key=" + _toml_string(env_key),
        "-c", "model_providers.azure.requires_openai_auth=false",
        "-c", "model_providers.azure.supports_websockets=false",
    ]


def _reasoning_effort_args(model: Optional[str]) -> list[str]:
    effort = os.environ.get("CODEX_MODEL_REASONING_EFFORT", "").strip()
    if effort.lower() in {"none", "unset", "off"}:
        return []
    if not effort and model and model.lower().startswith("gpt-5.5"):
        effort = "xhigh"
    if not effort:
        return []
    return ["-c", "model_reasoning_effort=" + _toml_string(effort)]


def _build_isolated_codex_cmd(
    model: Optional[str],
    cwd: Path,
    model_instructions_path: Optional[Path] = None,
) -> list[str]:
    model, provider_args = _maybe_configure_azure_provider(model)
    cmd = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--cd",
        str(cwd),
        "--ephemeral",
    ]
    cmd.extend(provider_args)
    cmd.extend(_reasoning_effort_args(model))
    tool_output_token_limit = os.environ.get(
        "CODEX_TOOL_OUTPUT_TOKEN_LIMIT", "4000",
    ).strip()
    if tool_output_token_limit.lower() in {"none", "unset", "off"}:
        tool_output_token_limit = ""
    if tool_output_token_limit:
        cmd.extend(["-c", f"tool_output_token_limit={tool_output_token_limit}"])
    if model_instructions_path is not None:
        cmd.extend([
            "-c",
            "model_instructions_file=" + _toml_string(str(model_instructions_path)),
        ])
    if model:
        cmd.extend(["--model", model])
    cmd.append("--dangerously-bypass-approvals-and-sandbox")
    cmd.append("-")
    return cmd


def _commit_experience_volume_barrier(label: str = "") -> bool:
    """Commit the attached experience volume from the active Modal container."""
    try:
        _EXPERIENCE_VOLUME.commit()
    except Exception as exc:  # noqa: BLE001
        print(
            f"[modal] experience volume commit failed"
            f"{f' ({label})' if label else ''}: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return False
    print(
        f"[modal] experience volume committed"
        f"{f' ({label})' if label else ''}",
        flush=True,
    )
    return True


# ---------------------------------------------------------------------------
# Core remote function
# ---------------------------------------------------------------------------


@_app.function(
    image=_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/repo/experience": _EXPERIENCE_VOLUME,
        "/root/.cache/huggingface": _HF_CACHE_VOLUME,
    },
    timeout=6 * 3600,  # 6h per eval — plenty for PPE-HP 500-pair holdout at concurrency 10
    cpu=8.0,
    memory=32768,
    # Preemption tolerance: 2 retries is enough to survive a single spot-worker
    # eviction. Higher counts amplify cancellation cascades (cancel → retry →
    # cancel again → retry …) and stomp volume state when new containers race
    # the previous container's cleanup. Each retry restarts with the same argv;
    # scores.json is rewritten from scratch so partial writes are safe.
    retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.5),
)
def _run_eval_remote(
    *,
    config_path: str,
    name: str,
    model: str,
    benchmark_path: str,
    fast: bool,
    tasks: Optional[str],
    concurrency: int,
    experience_dir: Optional[str],
    split: Optional[str] = None,
) -> Optional[dict]:
    """Run ONE evaluation inside its own Modal container.

    This is the fanout unit for k>1 per-epoch evaluation. Each sibling
    candidate gets its own container with 8 CPU / 32GB RAM, so three siblings
    fan out to 3x the effective resources instead of sharing a single
    orchestrator's memory — the root cause of the k=3 OOM crashes.

    Each call:
    1. Reloads the experience volume so it sees the orchestrator's freshly
       persisted candidate directory.
    2. Runs the standard `eval_runner.run()` with the same contract as the
       in-process code path. Scores.json and per-task artifacts land on the
       shared volume under the candidate dir.
    3. Commits the volume so the orchestrator (and other fanout callers) see
       the new files.
    4. Returns the scores dict back to the orchestrator, same shape as the
       in-process version.
    """
    import os
    from pathlib import Path as _Path

    os.chdir("/repo")
    _materialize_codex_auth()
    _EXPERIENCE_VOLUME.reload()

    from meta_agent.utils.logging import configure_logging
    from meta_agent.loop.epoch import run_evaluation
    from meta_agent.services.llm import bedrock_runtime_client

    configure_logging()
    bedrock_runtime_client()
    print(f"[eval-remote] {name}: starting", flush=True)
    scores = run_evaluation(
        config_path=_Path(config_path),
        name=name,
        model=model,
        benchmark_path=benchmark_path,
        split=split,
        fast=fast,
        tasks=tasks,
        concurrency=concurrency,
        experience_dir=_Path(experience_dir) if experience_dir else None,
    )
    try:
        _EXPERIENCE_VOLUME.commit()
    except Exception as exc:
        print(f"[eval-remote] {name}: volume commit failed: {exc}", flush=True)
    ok = scores is not None
    print(f"[eval-remote] {name}: done ok={ok}", flush=True)
    return scores


@_app.function(
    image=_CODEX_PROPOSER_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/work/experience": _EXPERIENCE_VOLUME,
    },
    timeout=2 * 3600,
    cpu=4.0,
    memory=16384,
    retries=0,
)
def _run_codex_proposer_isolated(
    *,
    prompt: str,
    model: Optional[str],
    trace_rel: Optional[str],
    staging_rel: Optional[str],
    model_instructions: Optional[str] = None,
    launch_id: Optional[str] = None,
    tool_output_token_limit: Optional[str] = None,
) -> dict:
    """Run Codex in a minimal image with only the experience volume mounted.

    This is intentionally separate from ``_run_meta_agent``. The optimizer and
    evaluator need the full repo; the proposer only needs the experience store
    and a prompt. Keeping Codex in ``/work`` prevents it from reading unrelated
    repo harnesses such as tuned scorers under ``/repo/harnesses``.
    """
    import json as _json
    import os as _os
    import queue as _queue
    import subprocess as _subprocess
    import threading as _threading
    import time as _time
    from pathlib import Path as _Path

    _os.chdir("/work")
    if launch_id:
        _os.environ["META_AGENT_MODAL_LAUNCH_ID"] = launch_id
    if str(tool_output_token_limit or "").strip().lower() in {"none", "unset", "off"}:
        _os.environ.pop("CODEX_TOOL_OUTPUT_TOKEN_LIMIT", None)
    elif tool_output_token_limit:
        _os.environ["CODEX_TOOL_OUTPUT_TOKEN_LIMIT"] = str(tool_output_token_limit)

    _materialize_codex_auth()
    _EXPERIENCE_VOLUME.reload()

    captured = {
        "cost_usd": None,
        "num_turns": None,
        "input_tokens": None,
        "output_tokens": None,
        "cache_read_tokens": None,
    }
    stderr_lines: list[str] = []
    cwd = _Path("/work")
    model_instructions_path = None
    if model_instructions:
        model_instructions_path = _Path("/work/model_instructions.md")
        model_instructions_path.write_text(model_instructions)
    try:
        cmd = _build_isolated_codex_cmd(model, cwd, model_instructions_path)
    except Exception as exc:  # noqa: BLE001
        return {
            "exit_code": 1,
            "command": [],
            "stderr": f"Failed to build codex command: {type(exc).__name__}: {exc}",
            **captured,
        }

    trace_file = None
    if trace_rel:
        try:
            trace_path = _Path("/work") / trace_rel
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_file = open(trace_path, "w")
        except OSError as exc:
            stderr_lines.append(
                f"Could not open codex trace {trace_rel}: {type(exc).__name__}: {exc}"
            )

    trace_write_failed = False
    last_activity = _time.time()
    started_at = last_activity
    try:
        try:
            proc = _subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=_subprocess.PIPE,
                stderr=_subprocess.PIPE,
                stdin=_subprocess.PIPE,
                text=True,
                bufsize=1,
                env=_os.environ.copy(),
            )
        except (OSError, _subprocess.SubprocessError) as exc:
            stderr_lines.append(
                f"Failed to start codex process: {type(exc).__name__}: {exc}"
            )
            return {
                "exit_code": 1,
                "command": cmd,
                "stderr": "\n".join(stderr_lines),
                **captured,
            }

        assert proc.stdin is not None
        try:
            proc.stdin.write(prompt)
            proc.stdin.close()
        except (BrokenPipeError, OSError) as exc:
            stderr_lines.append(
                f"Failed to write prompt to codex stdin: {type(exc).__name__}: {exc}"
            )

        q: _queue.Queue[tuple[str, str]] = _queue.Queue()

        def enqueue(stream, name: str) -> None:
            try:
                for raw in stream:
                    q.put((name, raw))
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        threads = [
            _threading.Thread(target=enqueue, args=(proc.stdout, "stdout"), daemon=True),
            _threading.Thread(target=enqueue, args=(proc.stderr, "stderr"), daemon=True),
        ]
        for thread in threads:
            thread.start()

        while True:
            now = _time.time()
            if now - started_at > 2 * 3600:
                proc.kill()
                stderr_lines.append("Process timed out after 7200s.")
                break
            if now - last_activity > 600:
                proc.kill()
                stderr_lines.append("Process stalled for 600s without output.")
                break
            try:
                stream_name, raw = q.get(timeout=0.1)
            except _queue.Empty:
                if proc.poll() is not None:
                    if all(not thread.is_alive() for thread in threads) and q.empty():
                        break
                continue

            line = raw.rstrip("\n")
            if not line:
                continue
            last_activity = _time.time()
            if stream_name == "stderr":
                stderr_lines.append(line)
                continue

            if trace_file and not trace_file.closed and not trace_write_failed:
                try:
                    trace_file.write(line + "\n")
                    trace_file.flush()
                except OSError as exc:
                    trace_write_failed = True
                    stderr_lines.append(
                        f"Could not write codex trace {trace_rel}: "
                        f"{type(exc).__name__}: {exc}"
                    )
            try:
                event = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            etype = event.get("type")
            if etype == "result":
                turns = event.get("num_turns")
                if isinstance(turns, int):
                    captured["num_turns"] = turns
                cost = event.get("total_cost_usd")
                if isinstance(cost, (int, float)):
                    captured["cost_usd"] = float(cost)
                usage = event.get("usage") or {}
                if isinstance(usage, dict):
                    captured["input_tokens"] = usage.get("input_tokens")
                    captured["output_tokens"] = usage.get("output_tokens")
                    captured["cache_read_tokens"] = usage.get("cache_read_input_tokens")
            elif etype == "item.completed":
                item = event.get("item") or {}
                if isinstance(item, dict):
                    if item.get("type") == "file_change":
                        paths = [
                            c.get("path", "?").rsplit("/", 1)[-1]
                            for c in item.get("changes", [])
                            if isinstance(c, dict)
                        ]
                        print(f"  [ISOLATED PROPOSER FILE] {', '.join(paths)}", flush=True)
                    elif item.get("type") == "agent_message":
                        text = str(item.get("text", "")).strip()
                        if text:
                            print(f"  [ISOLATED PROPOSER] {text[:300]}", flush=True)
            elif etype == "error":
                message = str(event.get("message", "")).strip()
                if message:
                    print(f"  [ISOLATED PROPOSER ERROR] {message[:500]}", flush=True)
            elif etype == "turn.failed":
                error = event.get("error") or {}
                if isinstance(error, dict):
                    message = str(error.get("message", "")).strip()
                else:
                    message = str(error).strip()
                if message:
                    print(f"  [ISOLATED PROPOSER FAILED] {message[:500]}", flush=True)

        for thread in threads:
            thread.join(timeout=2)
        try:
            exit_code = proc.wait(timeout=5)
        except _subprocess.TimeoutExpired:
            proc.kill()
            exit_code = proc.wait(timeout=5)
    finally:
        if trace_file:
            trace_file.close()

    staged_files: dict[str, str] = {}
    if staging_rel:
        staging_path = _Path("/work") / staging_rel
        if staging_path.is_dir():
            for path in sorted(staging_path.rglob("*")):
                if not path.is_file():
                    continue
                try:
                    staged_files[path.relative_to(staging_path).as_posix()] = path.read_text()
                except UnicodeDecodeError:
                    continue

    trace_text = None
    if trace_rel:
        trace_path = _Path("/work") / trace_rel
        if trace_path.is_file():
            try:
                trace_text = trace_path.read_text()
            except UnicodeDecodeError:
                trace_text = None

    try:
        _EXPERIENCE_VOLUME.commit()
    except Exception as exc:  # noqa: BLE001
        stderr_lines.append(f"Volume commit failed: {type(exc).__name__}: {exc}")

    return {
        "exit_code": exit_code,
        "command": cmd,
        "stderr": "\n".join(stderr_lines),
        "staged_files": staged_files,
        "trace_text": trace_text,
        **captured,
    }


@_app.function(
    image=_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/repo/experience": _EXPERIENCE_VOLUME,
        "/root/.cache/huggingface": _HF_CACHE_VOLUME,
    },
    timeout=24 * 3600,
    cpu=16.0,
    memory=131072,
    # Preemption tolerance: 2 retries is the sweet spot — covers a single
    # spot-worker eviction mid-run without amplifying cancellation cascades.
    # Each restart runs with the same argv; the meta-agent loop always passes
    # `--resume` so it picks up from the last completed epoch on the volume.
    # Higher retry counts cause problems when we explicitly cancel a run: the
    # cancellation bubbles up as an exception, modal.Retries schedules a new
    # container (even though the user meant to stop), and the new container
    # races the old container's volume cleanup → "pid still alive" warnings
    # and partial-write stomping.
    retries=modal.Retries(max_retries=2, initial_delay=30.0, backoff_coefficient=1.5),
)
def _run_meta_agent(
    argv: list[str],
    launch_id: Optional[str] = None,
    env_overrides: Optional[dict[str, str]] = None,
) -> int:
    """Run the meta-agent CLI inside this Modal container, in-process.

    The container has:
    - /repo mounted with the current codebase
    - /repo/experience backed by a persistent Modal Volume (candidates keep
      between runs; delete by `modal volume rm -r meta-agent-experience
      <benchmark_name>`)
    - /root/.cache/huggingface persistent for dataset downloads
    - claude-agent-sdk + native Node-based claude CLI + codex CLI on PATH
    - Bedrock + OpenAI + codex auth env wired in via secrets

    Execution model: we invoke the `meta-agent` CLI in-process via
    `meta_agent.__main__.main()` rather than as a subprocess. This matters
    because Modal's `.spawn()` / `.remote()` calls inside the loop (the k>1
    eval fanout) require an active Modal app context in the current process.
    A subprocess would lose that context, silently fall back to threads, and
    OOM the orchestrator at k>1 × concurrency × large-batch workloads.
    """
    import os
    import sys as _sys

    os.chdir("/repo")
    if launch_id:
        os.environ["META_AGENT_MODAL_LAUNCH_ID"] = launch_id
    os.environ.setdefault("META_AGENT_APP_NAME", _MODAL_APP_NAME)
    os.environ.setdefault("META_AGENT_CODEX_PROPOSER_ISOLATED", "1")
    os.environ.setdefault("META_AGENT_MODAL_RUNNER_MODULE", __name__)
    if env_overrides:
        for key, value in env_overrides.items():
            if value is not None:
                os.environ[str(key)] = str(value)

    _materialize_codex_auth()
    from meta_agent.services.llm import bedrock_runtime_client

    bedrock_runtime_client()

    # Simulate the CLI's argv so `meta_agent.__main__.main()` picks it up.
    print(f"[modal] meta-agent {' '.join(argv)}", flush=True)
    saved_argv = _sys.argv
    _sys.argv = ["meta-agent", *argv]
    try:
        from meta_agent.__main__ import main as _cli_main
        try:
            _cli_main()
            return 0
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
            return code
    finally:
        _sys.argv = saved_argv


def _materialize_codex_auth() -> None:
    """Write the codex-auth Modal Secret into ~/.codex/ so `codex exec` works.

    The secret bundles two values:
    - CODEX_AUTH_JSON  (content of ~/.codex/auth.json, required)
    - CODEX_CONFIG_TOML (content of ~/.codex/config.toml, optional)

    Use `python -m meta_agent.cloud.modal_runner upload-codex-auth` to create the
    secret from your local ~/.codex/ directory.
    """
    import os
    from pathlib import Path

    codex_home = Path.home() / ".codex"
    codex_home.mkdir(parents=True, exist_ok=True)

    auth = os.environ.get("CODEX_AUTH_JSON", "")
    if auth:
        auth_path = codex_home / "auth.json"
        auth_path.write_text(auth)
        auth_path.chmod(0o600)
        print(f"[modal] wrote codex auth -> {auth_path}", flush=True)
    else:
        print(
            "[modal] WARNING: CODEX_AUTH_JSON missing from codex-auth secret. "
            "The codex proposer will fail to authenticate.",
            flush=True,
        )

    config = os.environ.get("CODEX_CONFIG_TOML", "")
    if config:
        (codex_home / "config.toml").write_text(config)


# ---------------------------------------------------------------------------
# Local entrypoints: `modal run meta_agent/cloud/modal_runner.py::loop ...`
# ---------------------------------------------------------------------------


@_app.local_entrypoint()
def loop(
    benchmark: str,
    run_name: str,
    baseline: str = "harnesses/claude_vanilla",
    iterations: int = 10,
    model: str = "claude-sonnet-4-6",
    proposer_model: str = "gpt-5.3-codex",
    proposer_cli: str = "codex",
    proposer_max_turns: Optional[int] = None,
    concurrency: int = 100,
    fast: bool = False,
    split: Optional[str] = None,
    holdout: Optional[str] = None,
    holdout_split: Optional[str] = None,
    accept_on_holdout: bool = False,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None,
    start_from: int = 1,
    candidates_per_iter: int = 1,
    final_test: Optional[str] = None,
    final_test_split: Optional[str] = None,
    final_test_frontier: bool = False,
    final_test_current_best: bool = False,
    final_test_baseline: bool = False,
    fresh: bool = False,
    resume: bool = False,
    resume_from_proposal: bool = False,
) -> None:
    """Run `meta-agent loop` against `benchmark` on Modal.

    All flags mirror the local `meta-agent loop` CLI.

    `benchmark` and `holdout` accept any of:
      * ``path/to/family.yaml``
      * ``path/to/family.yaml:split-name``
      * ``path/to/dir:split-name``
    Or pass the file/dir and use `split` / `holdout_split` separately.

    `accept_on_holdout=True` gates 'new best' selection on the holdout
    (validation) benchmark instead of the search (train) benchmark —
    the standard remedy for search-set overfitting. Requires `holdout`
    to be set.

    `batch_size` samples a subset of the search benchmark per epoch
    instead of running the full pool. Use with `seed` for reproducible
    batch rotation. Strongly recommended for large train sets: cuts
    per-iteration cost linearly without changing acceptance semantics
    (the full holdout still gates).

    `run_name` overrides the experience-store dir name (default:
    ``<family>-<split>``). Use this to run two concurrent loops over
    the same family:split with different flags (e.g. k=1 vs k=3)
    without colliding on shared artifacts.
    """
    argv = [
        "loop",
        "--benchmark", benchmark,
        "--baseline", baseline,
        "--iterations", str(iterations),
        "--model", model,
        "--proposer-model", proposer_model,
        "--proposer-cli", proposer_cli,
        "--concurrency", str(concurrency),
    ]
    if proposer_max_turns is not None:
        argv.extend(["--proposer-max-turns", str(proposer_max_turns)])
    if split:
        argv.extend(["--split", split])
    if fast:
        argv.append("--fast")
    if holdout:
        argv.extend(["--holdout", holdout])
    if holdout_split:
        argv.extend(["--holdout-split", holdout_split])
    if accept_on_holdout:
        argv.append("--accept-on-holdout")
    if batch_size is not None:
        argv.extend(["--batch-size", str(batch_size)])
    if seed is not None:
        argv.extend(["--seed", str(seed)])
    if start_from != 1:
        argv.extend(["--start-from", str(start_from)])
    if candidates_per_iter != 1:
        argv.extend(["--candidates-per-iter", str(candidates_per_iter)])
    if final_test:
        argv.extend(["--final-test", final_test])
    if final_test_split:
        argv.extend(["--final-test-split", final_test_split])
    if final_test_frontier:
        argv.append("--final-test-frontier")
    if final_test_current_best:
        argv.append("--final-test-current-best")
    if final_test_baseline:
        argv.append("--final-test-baseline")
    # --run-name is required locally and by the inner CLI.
    argv.extend(["--run-name", run_name])
    if fresh:
        argv.append("--fresh")
    if resume:
        argv.append("--resume")
    if resume_from_proposal:
        argv.append("--resume-from-proposal")

    call = _run_meta_agent.spawn(argv, _new_launch_id("loop"))
    print("=" * 70)
    print("Launched Modal loop")
    print(f"  app:              {_MODAL_APP_NAME}")
    print(f"  volume:           {_EXPERIENCE_VOLUME_NAME}")
    print(f"  run_name:         {run_name}")
    print(f"  iterations:       {iterations} (start_from={start_from})")
    print(f"  k:                {candidates_per_iter}")
    print(f"  concurrency:      {concurrency}")
    print(f"  FunctionCall ID:  {call.object_id}")
    print()
    print("Monitor via:")
    print(f"  Dashboard:  https://modal.com/apps/canvas-org/main/{_MODAL_APP_NAME}")
    print(f"  Logs CLI:   modal app logs {_MODAL_APP_NAME} --function-call {call.object_id}")
    print("=" * 70)


@_app.local_entrypoint()
def eval(
    benchmark: str,
    config: str,
    name: str,
    model: str = "claude-sonnet-4-6",
    concurrency: int = 100,
    tasks: Optional[str] = None,
    fast: bool = False,
    split: Optional[str] = None,
) -> None:
    """Run `meta-agent eval` against `benchmark` on Modal (one-shot, no proposer).

    Typical use: transfer-eval an optimized harness on a held-out test set.
    `benchmark` accepts ``path:split`` or you can pass `split` separately.
    """
    argv = [
        "eval",
        "--benchmark", benchmark,
        "--config", config,
        "--name", name,
        "--model", model,
        "--concurrency", str(concurrency),
    ]
    if split:
        argv.extend(["--split", split])
    if tasks:
        argv.extend(["--tasks", tasks])
    if fast:
        argv.append("--fast")

    rc = _run_meta_agent.remote(argv, _new_launch_id("eval"))
    print(f"[modal] meta-agent exit code: {rc}")
    if rc != 0:
        sys.exit(rc)


@_app.function(
    image=_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/repo/experience": _EXPERIENCE_VOLUME,
        "/root/.cache/huggingface": _HF_CACHE_VOLUME,
    },
    timeout=3 * 3600,
    cpu=8.0,
    memory=32768,
)
def _cache_baseline_tau3_remote(
    benchmark: str,
    config_path: str,
    out_path: str,
    model: str,
    user_model: Optional[str],
    concurrency: int,
    timeout: int,
    tasks: Optional[str],
) -> dict:
    """Generate a baseline trajectory cache (Experiment A §6) inside Modal.

    Writes a JSONL pool to ``out_path`` (resolved inside ``/repo``, i.e.
    the mounted experience volume when the path starts with
    ``experience/``). The Stage-2 adapter reads this pool at eval time to
    pair every candidate rollout against the frozen baseline trajectory.
    """
    import asyncio
    import os
    from pathlib import Path as _Path

    os.chdir("/repo")
    _materialize_codex_auth()
    _EXPERIENCE_VOLUME.reload()

    from meta_agent.utils.logging import configure_logging

    from benchmarks.tau3.cache_baseline_trajectories import (
        _load_task_list,
        cache_baseline,
    )

    configure_logging()

    task_filter = (
        [t.strip() for t in tasks.split(",") if t.strip()] if tasks else None
    )
    task_list = _load_task_list(benchmark, task_filter=task_filter)
    if not task_list:
        print("[cache-baseline] no tasks resolved; aborting", flush=True)
        return {"n_rollouts": 0, "n_passed": 0, "n_failed": 0, "n_errors": 0}

    print(
        f"[cache-baseline] benchmark={benchmark} config={config_path} "
        f"out={out_path} model={model} n_tasks={len(task_list)}",
        flush=True,
    )

    stats = asyncio.run(cache_baseline(
        task_list=task_list,
        config_path=config_path,
        model=model,
        out_path=_Path(out_path),
        user_model=user_model,
        concurrency=concurrency,
        timeout_s=timeout,
    ))
    try:
        _EXPERIENCE_VOLUME.commit()
    except Exception as exc:
        print(f"[cache-baseline] volume commit failed: {exc}", flush=True)
    print(f"[cache-baseline] stats: {stats}", flush=True)
    return stats


@_app.local_entrypoint()
def cache_baseline_tau3(
    benchmark: str = "benchmarks/tau3/benchmark.yaml:search-judge-v1",
    config: str = "harnesses/claude_vanilla",
    out: str = "experience/.cache/tau3_baseline_cache_v1.jsonl",
    model: str = "claude-haiku-4-5",
    # Default MUST match the user_model used to build the Stage-1 judge
    # training pool (`build_pool.py --user-model gpt-4.1`) so cached
    # baseline trajectories are in-distribution with what the frozen judge
    # was trained on. Override only if the Stage-1 pool was regenerated
    # with a different user simulator.
    user_model: str = "gpt-4.1",
    concurrency: int = 20,
    timeout: int = 300,
    tasks: Optional[str] = None,
) -> None:
    """Generate the frozen baseline-trajectory cache for Stage 2 on Modal.

    Runs the vanilla actor on every task in ``benchmark`` and writes one
    JSONL record per task to ``out`` on the ``meta-agent-experience``
    volume. Safe to re-run — it overwrites ``out`` each time. Pass
    ``--tasks id1,id2,..`` to cache a subset for dev smoke.

    Example::

        modal run meta_agent/cloud/modal_runner.py::cache_baseline_tau3 \\
            --benchmark benchmarks/tau3/benchmark.yaml:search-judge-v1 \\
            --config harnesses/claude_vanilla \\
            --out experience/.cache/tau3_baseline_cache_v1.jsonl \\
            --model claude-haiku-4-5 \\
            --concurrency 20
    """
    stats = _cache_baseline_tau3_remote.remote(
        benchmark=benchmark,
        config_path=config,
        out_path=out,
        model=model,
        user_model=user_model,
        concurrency=concurrency,
        timeout=timeout,
        tasks=tasks,
    )
    print(f"[modal] cache-baseline stats: {stats}")
    if stats["n_rollouts"] == 0:
        sys.exit(1)


@_app.function(
    image=_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/repo/experience": _EXPERIENCE_VOLUME,
        "/root/.cache/huggingface": _HF_CACHE_VOLUME,
    },
    timeout=24 * 3600,
    cpu=16.0,
    memory=131072,
)
def _tau3_best_of_n_remote(
    *,
    pool: str,
    task_split: str,
    judges: list[str],
    pointwise_judges: list[str],
    n_values: str,
    samples_per_task: int,
    sampling_mode: str,
    cache_pointwise_scores: bool,
    model: str,
    timeout: int,
    concurrency: int,
    seed: int,
    task_limit: Optional[int],
    task_ids: Optional[str],
    out: str,
) -> dict:
    """Run the tau3 trajectory-judge best-of-N driver inside Modal."""
    import json
    import os
    import sys as _sys
    from pathlib import Path as _Path

    os.chdir("/repo")
    _materialize_codex_auth()
    _EXPERIENCE_VOLUME.reload()

    argv = [
        "best_of_n",
        "--pool", pool,
        "--task-split", task_split,
        "--n-values", n_values,
        "--samples-per-task", str(samples_per_task),
        "--sampling-mode", sampling_mode,
        "--model", model,
        "--timeout", str(timeout),
        "--concurrency", str(concurrency),
        "--seed", str(seed),
        "--out", out,
    ]
    for judge in judges:
        argv.extend(["--judge", judge])
    for judge in pointwise_judges:
        argv.extend(["--pointwise-judge", judge])
    if cache_pointwise_scores:
        argv.append("--cache-pointwise-scores")
    if task_limit is not None:
        argv.extend(["--task-limit", str(task_limit)])
    if task_ids:
        argv.extend(["--task-ids", task_ids])

    print(f"[modal] tau3 best-of-n {' '.join(argv[1:])}", flush=True)
    saved_argv = _sys.argv
    _sys.argv = argv
    try:
        from benchmarks.tau3_trajectory_judge.best_of_n import main as _bon_main

        _bon_main()
    finally:
        _sys.argv = saved_argv

    try:
        _EXPERIENCE_VOLUME.commit()
    except Exception as exc:
        print(f"[best-of-n] volume commit failed: {exc}", flush=True)

    out_path = _Path(out)
    if not out_path.is_absolute():
        out_path = _Path("/repo") / out_path
    payload = json.loads(out_path.read_text())
    return {
        "out": out,
        "summary": payload.get("summary"),
        "n_episodes": len(payload.get("episodes") or []),
    }


@_app.local_entrypoint()
def tau3_best_of_n(
    pool: str = "benchmarks/tau3_trajectory_judge/data/airline_pool_v1_test.jsonl",
    task_split: str = "judge-test",
    judges: str = "baseline=harnesses/reward_models/tau3_airline_trajectory/pairwise_judge",
    pointwise_judges: str = "",
    n_values: str = "2,4",
    samples_per_task: int = 5,
    sampling_mode: str = "controlled_mixed",
    cache_pointwise_scores: bool = False,
    model: str = "claude-haiku-4-5",
    timeout: int = 720,
    concurrency: int = 100,
    seed: int = 42,
    task_limit: Optional[int] = None,
    task_ids: Optional[str] = None,
    out: str = "experience/best_of_n/tau3_best_of_n.json",
) -> None:
    """Run tau3 trajectory-judge best-of-N on Modal.

    `judges` is a comma-separated list of NAME=CONFIG_OR_HARNESS_PATH entries,
    matching `benchmarks.tau3_trajectory_judge.best_of_n`. `pointwise_judges`
    is the same format for scalar-scoring program harnesses.
    """
    import json

    judge_list = [item.strip() for item in judges.split(",") if item.strip()]
    pointwise_judge_list = [
        item.strip() for item in pointwise_judges.split(",") if item.strip()
    ]
    result = _tau3_best_of_n_remote.remote(
        pool=pool,
        task_split=task_split,
        judges=judge_list,
        pointwise_judges=pointwise_judge_list,
        n_values=n_values,
        samples_per_task=samples_per_task,
        sampling_mode=sampling_mode,
        cache_pointwise_scores=cache_pointwise_scores,
        model=model,
        timeout=timeout,
        concurrency=concurrency,
        seed=seed,
        task_limit=task_limit,
        task_ids=task_ids,
        out=out,
    )
    print(json.dumps(result, indent=2))


@_app.local_entrypoint()
def raw(argv_str: str) -> None:
    """Escape hatch — pass any `meta-agent <...>` argv as a single string.

    Example:
        modal run meta_agent/cloud/modal_runner.py::raw --argv-str \\
            "propose --project foo --model gpt-5.3-codex"
    """
    import shlex

    argv = shlex.split(argv_str)
    rc = _run_meta_agent.remote(argv, _new_launch_id("raw"))
    print(f"[modal] meta-agent exit code: {rc}")
    if rc != 0:
        sys.exit(rc)


# ---------------------------------------------------------------------------
# Smoke-gate verification: run the production smoke-gate against existing
# volume candidates. Use this to confirm known-bad candidates are rejected
# and known-good candidates are accepted after shipping smoke_gate.py.
# ---------------------------------------------------------------------------


@_app.function(
    image=_IMAGE,
    secrets=_SECRETS,
    volumes={
        "/repo/experience": _EXPERIENCE_VOLUME,
        "/root/.cache/huggingface": _HF_CACHE_VOLUME,
    },
    timeout=15 * 60,  # smoke = 1 pair at ~60-180s each; 15 min for a handful
    cpu=4.0,
    memory=16384,
)
def _verify_smoke_gate_remote(
    run_name: str,
    candidate_names: list[str],
    benchmark: str,
    model: str,
) -> list[dict]:
    """Run ``smoke_candidate`` against existing candidates on the volume.

    For each name in ``candidate_names``, resolves
    ``/repo/experience/<run_name>/candidates/<name>/harness.py`` and
    runs the production smoke-gate. Returns a list of
    ``{name, ok, error, duration_s, pair_id}`` dicts — one per candidate.

    Use this to sanity-check the gate before committing to a rerun:
    point it at a known-bad (e.g., ``evo_003_post_verdict_review``) and a
    known-good (e.g., ``baseline``) from a completed run and confirm the
    gate rejects/accepts as expected.
    """
    import asyncio
    import os
    from pathlib import Path as _Path

    os.chdir("/repo")
    _materialize_codex_auth()
    _EXPERIENCE_VOLUME.reload()

    from meta_agent.core.benchmark import load_benchmark
    from meta_agent.utils.logging import configure_logging
    from meta_agent.loop.smoke_gate import smoke_candidate

    configure_logging()
    print(f"[smoke-verify] run={run_name} benchmark={benchmark}", flush=True)
    print(f"[smoke-verify] candidates={candidate_names}", flush=True)

    if ":" in benchmark:
        path, split = benchmark.split(":", 1)
    else:
        path, split = benchmark, None
    bench = load_benchmark(path, split=split)

    async def _run_all() -> list[dict]:
        results: list[dict] = []
        for name in candidate_names:
            harness_path = _Path(
                f"/repo/experience/{run_name}/candidates/{name}/harness.py"
            )
            if not harness_path.is_file():
                print(f"[smoke-verify] {name}: MISSING {harness_path}", flush=True)
                results.append({
                    "name": name, "ok": False, "error": f"missing harness: {harness_path}",
                    "duration_s": 0.0, "pair_id": None,
                })
                continue
            print(f"[smoke-verify] {name}: starting...", flush=True)
            sr = await smoke_candidate(
                harness_path=harness_path,
                benchmark=bench,
                model=model,
            )
            verdict = "PASS" if sr.ok else "REJECT"
            print(
                f"[smoke-verify] {name}: {verdict} pair={sr.pair_id} "
                f"duration={sr.duration_s:.1f}s"
                + (f" error={sr.error[:200]}" if sr.error else ""),
                flush=True,
            )
            results.append({
                "name": name,
                "ok": sr.ok,
                "error": sr.error,
                "duration_s": sr.duration_s,
                "pair_id": sr.pair_id,
            })
        return results

    return asyncio.run(_run_all())


@_app.local_entrypoint()
def verify_smoke_gate(
    run_name: str,
    candidate_names: str,  # comma-separated; e.g. "baseline,evo_003_post_verdict_review"
    benchmark: str = "benchmarks/tau3_trajectory_judge/benchmark.yaml:judge-train",
    model: str = "claude-haiku-4-5",
) -> None:
    """Smoke-test specific volume candidates to verify the smoke-gate works.

    Typical use: after shipping ``meta_agent/loop/smoke_gate.py``, run
    this against a known-bad candidate and a known-good candidate; confirm
    the gate rejects/accepts as expected.

    Example::

        modal run meta_agent/cloud/modal_runner.py::verify_smoke_gate \\
            --run-name tau3-judge-v1-k10 \\
            --candidate-names "baseline,evo_002_fused_best,evo_003_post_verdict_review"

    Expected output:
        [smoke-verify] baseline: PASS ...
        [smoke-verify] evo_002_fused_best: PASS ...
        [smoke-verify] evo_003_post_verdict_review: REJECT ... ZodError ...
    """
    names = [n.strip() for n in candidate_names.split(",") if n.strip()]
    if not names:
        raise SystemExit("no candidate-names provided")
    results = _verify_smoke_gate_remote.remote(
        run_name=run_name,
        candidate_names=names,
        benchmark=benchmark,
        model=model,
    )
    print()
    print("=" * 70)
    print("  Smoke-gate verification results")
    print("=" * 70)
    print(f"  {'candidate':<45}  {'verdict':>8}  {'pair_id':>12}  {'time':>6}")
    print(f"  {'-'*45}  {'-'*8}  {'-'*12}  {'-'*6}")
    for r in results:
        verdict = "PASS" if r["ok"] else "REJECT"
        pair_id = r.get("pair_id") or "-"
        dur = f"{r['duration_s']:.0f}s"
        print(f"  {r['name']:<45}  {verdict:>8}  {pair_id:>12}  {dur:>6}")
        if r.get("error"):
            print(f"    error: {r['error'][:200]}")
    n_rejected = sum(1 for r in results if not r["ok"])
    print(f"\n  {len(results)} candidate(s) checked — {n_rejected} rejected.")


# ---------------------------------------------------------------------------
# One-shot helper: upload local ~/.codex/ contents as a Modal secret
# ---------------------------------------------------------------------------


def upload_codex_auth() -> None:
    """Read the local `~/.codex/auth.json` + `~/.codex/config.toml` and push
    them to Modal as the `codex-auth` secret.

    Invoked as:

        python -m meta_agent.cloud.modal_runner upload-codex-auth

    Re-run this any time your codex OAuth token refreshes. Idempotent —
    overwrites whatever was there.
    """
    import subprocess
    from pathlib import Path

    codex_home = Path.home() / ".codex"
    auth_path = codex_home / "auth.json"
    config_path = codex_home / "config.toml"

    if not auth_path.is_file():
        raise SystemExit(
            f"codex auth not found at {auth_path}. Run `codex` locally once "
            "to authenticate, then retry."
        )

    auth_content = auth_path.read_text()
    config_content = config_path.read_text() if config_path.is_file() else ""

    cmd = [
        "modal", "secret", "create", "codex-auth",
        f"CODEX_AUTH_JSON={auth_content}",
        f"CODEX_CONFIG_TOML={config_content}",
        "--force",
    ]
    print(
        f"[modal] uploading codex-auth ({len(auth_content)}B auth + "
        f"{len(config_content)}B config)...",
        flush=True,
    )
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit("modal secret create failed; see error above")
    print("[modal] codex-auth secret updated.")


def _deploy_modal_app(
    *,
    app_name: str,
    experience_volume: str,
    hf_cache_volume: str,
    codex_tool_output_token_limit: Optional[int] = None,
    stream_logs: bool = False,
) -> None:
    import subprocess

    env = os.environ.copy()
    env["META_AGENT_APP_NAME"] = app_name
    env["META_AGENT_EXPERIENCE_VOLUME"] = experience_volume
    env["META_AGENT_HF_CACHE_VOLUME"] = hf_cache_volume
    if codex_tool_output_token_limit is not None:
        env["CODEX_TOOL_OUTPUT_TOKEN_LIMIT"] = str(codex_tool_output_token_limit)

    cmd = [
        "modal",
        "deploy",
        str(_THIS_FILE),
        "--name",
        app_name,
    ]
    if stream_logs:
        cmd.append("--stream-logs")

    print(
        "[modal] deploying "
        f"app={app_name} volume={experience_volume} hf_cache={hf_cache_volume}",
        flush=True,
    )
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def deploy_app() -> None:
    """Deploy this Modal app with command-line flags instead of env incantations.

    Modal binds the app name and mounted volumes when the app is deployed, so
    these cannot be changed by an already-running FunctionCall. This helper
    keeps that deployment-time nature explicit while giving callers a normal
    CLI surface:

        python -m meta_agent.cloud.modal_runner deploy-app \\
            --app-name meta-agent-gpt55 \\
            --experience-volume meta-agent-experience-gpt55
    """
    import argparse

    parser = argparse.ArgumentParser(prog="deploy-app", add_help=True)
    parser.add_argument("--app-name", default=_MODAL_APP_NAME)
    parser.add_argument("--experience-volume", default=_EXPERIENCE_VOLUME_NAME)
    parser.add_argument("--hf-cache-volume", default=_HF_CACHE_VOLUME_NAME)
    parser.add_argument(
        "--codex-tool-output-token-limit",
        type=str,
        default=None,
        help="Default Codex tool_output_token_limit baked into this deployment.",
    )
    parser.add_argument("--stream-logs", action="store_true")
    args = parser.parse_args(sys.argv[2:])

    _deploy_modal_app(
        app_name=args.app_name,
        experience_volume=args.experience_volume,
        hf_cache_volume=args.hf_cache_volume,
        codex_tool_output_token_limit=args.codex_tool_output_token_limit,
        stream_logs=args.stream_logs,
    )


def launch_loop() -> None:
    """Launch a `meta-agent loop` job on the *deployed* Modal app.

    This is the preferred entrypoint for multi-hour runs. Unlike
    ``modal run --detach meta_agent/cloud/modal_runner.py::loop`` (which creates a
    fresh ephemeral app per invocation and depends on the local client for
    control-plane continuity), this uses ``modal.Function.from_name`` to
    dispatch onto the pre-deployed app, so:

    - The app stays visible in ``modal app list`` after invocation
    - Preemption retries (configured in ``@_run_meta_agent``) take effect
    - Jobs survive local client disconnects / DNS blips cleanly
    - Each invocation returns a FunctionCall ID you can inspect/tail later

    One-time setup (before first use):
        modal deploy meta_agent/cloud/modal_runner.py

    Usage (same flags as the `loop` modal-run entrypoint):
        python -m meta_agent.cloud.modal_runner launch-loop \\
            --benchmark benchmarks/tau3_trajectory_judge/benchmark.yaml:judge-train \\
            --holdout   benchmarks/tau3_trajectory_judge/benchmark.yaml:judge-val \\
            --baseline  harnesses/claude_vanilla_tau3_trajectory_judge \\
            --run-name  tau3-judge-v1-k10 \\
            --iterations 17 --start-from 4 \\
            --candidates-per-iter 10 --concurrency 5 \\
            --model claude-haiku-4-5 --proposer-model claude-opus-4-6 --proposer-cli claude \\
            --accept-on-holdout --resume
    """
    import argparse

    parser = argparse.ArgumentParser(prog="launch-loop", add_help=True)
    parser.add_argument("--app-name", default=_MODAL_APP_NAME)
    parser.add_argument(
        "--deploy",
        action="store_true",
        help=(
            "Deploy/update the target app before launching. Required when "
            "using a new app name or volume from this one command."
        ),
    )
    parser.add_argument(
        "--experience-volume",
        default=None,
        help="Modal volume to mount at /repo/experience when --deploy is set.",
    )
    parser.add_argument(
        "--hf-cache-volume",
        default=_HF_CACHE_VOLUME_NAME,
        help="Modal volume for HuggingFace cache when --deploy is set.",
    )
    parser.add_argument("--stream-deploy-logs", action="store_true")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--baseline", default="harnesses/claude_vanilla")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--proposer-model", default="gpt-5.3-codex")
    parser.add_argument("--proposer-cli", default="codex", choices=["claude", "codex"])
    parser.add_argument("--proposer-max-turns", type=int, default=None)
    parser.add_argument("--max-proposer-failures", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--split", default=None)
    parser.add_argument("--holdout", default=None)
    parser.add_argument("--holdout-split", default=None)
    parser.add_argument("--accept-on-holdout", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--candidates-per-iter", type=int, default=1)
    parser.add_argument("--final-test", default=None)
    parser.add_argument("--final-test-split", default=None)
    parser.add_argument("--final-test-frontier", action="store_true")
    parser.add_argument("--final-test-current-best", action="store_true")
    parser.add_argument("--final-test-baseline", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-from-proposal", action="store_true")
    parser.add_argument(
        "--codex-tool-output-token-limit",
        type=str,
        default=None,
        help=(
            "Override Codex CLI tool_output_token_limit inside the Modal run "
            "(useful for A/B testing proposer context truncation, e.g. 8000; "
            "use 'none' to omit the override)."
        ),
    )
    parser.add_argument(
        "--codex-fair-proposer",
        action="store_true",
        help=(
            "Use the Opus-like full-repo Codex proposer surface: no isolated "
            "proposer, no tau3 scaffold static guard, and no locked Codex "
            "model instructions. Keeps infrastructure retry/salvage fixes."
        ),
    )
    args = parser.parse_args(sys.argv[2:])

    if args.experience_volume and not args.deploy:
        parser.error("--experience-volume only applies with --deploy")
    if args.deploy:
        if not args.experience_volume:
            parser.error("--deploy requires --experience-volume")
        _deploy_modal_app(
            app_name=args.app_name,
            experience_volume=args.experience_volume,
            hf_cache_volume=args.hf_cache_volume,
            codex_tool_output_token_limit=args.codex_tool_output_token_limit,
            stream_logs=args.stream_deploy_logs,
        )

    # Build the inner meta-agent argv (same shape as `loop` local_entrypoint).
    argv = [
        "loop",
        "--benchmark", args.benchmark,
        "--baseline", args.baseline,
        "--iterations", str(args.iterations),
        "--model", args.model,
        "--proposer-model", args.proposer_model,
        "--proposer-cli", args.proposer_cli,
        "--concurrency", str(args.concurrency),
    ]
    if args.proposer_max_turns is not None:
        argv.extend(["--proposer-max-turns", str(args.proposer_max_turns)])
    if args.max_proposer_failures is not None:
        argv.extend(["--max-proposer-failures", str(args.max_proposer_failures)])
    if args.split:
        argv.extend(["--split", args.split])
    if args.fast:
        argv.append("--fast")
    if args.holdout:
        argv.extend(["--holdout", args.holdout])
    if args.holdout_split:
        argv.extend(["--holdout-split", args.holdout_split])
    if args.accept_on_holdout:
        argv.append("--accept-on-holdout")
    if args.batch_size is not None:
        argv.extend(["--batch-size", str(args.batch_size)])
    if args.seed is not None:
        argv.extend(["--seed", str(args.seed)])
    if args.start_from != 1:
        argv.extend(["--start-from", str(args.start_from)])
    if args.candidates_per_iter != 1:
        argv.extend(["--candidates-per-iter", str(args.candidates_per_iter)])
    if args.final_test:
        argv.extend(["--final-test", args.final_test])
    if args.final_test_split:
        argv.extend(["--final-test-split", args.final_test_split])
    if args.final_test_frontier:
        argv.append("--final-test-frontier")
    if args.final_test_current_best:
        argv.append("--final-test-current-best")
    if args.final_test_baseline:
        argv.append("--final-test-baseline")
    argv.extend(["--run-name", args.run_name])
    if args.fresh:
        argv.append("--fresh")
    if args.resume:
        argv.append("--resume")
    if args.resume_from_proposal:
        argv.append("--resume-from-proposal")

    # Resolve the deployed function by name (NOT creating a fresh ephemeral
    # app via `modal run`). Requires `modal deploy meta_agent/cloud/modal_runner.py`
    # to have been run at least once on the current codebase.
    try:
        fn = modal.Function.from_name(args.app_name, "_run_meta_agent")
    except Exception as exc:
        raise SystemExit(
            f"Could not look up deployed function: {exc}\n\n"
            f"Deploy the app first:  META_AGENT_APP_NAME={args.app_name} "
            f"modal deploy meta_agent/cloud/modal_runner.py"
        )

    env_overrides = {"META_AGENT_APP_NAME": args.app_name}
    if args.codex_tool_output_token_limit is not None:
        env_overrides["CODEX_TOOL_OUTPUT_TOKEN_LIMIT"] = str(
            args.codex_tool_output_token_limit
        )
    if args.codex_fair_proposer:
        env_overrides["META_AGENT_CODEX_FAIR_PROPOSER"] = "1"
        env_overrides["META_AGENT_CODEX_PROPOSER_ISOLATED"] = "0"
        env_overrides["META_AGENT_DISABLE_TAU3_STATIC_GUARD"] = "1"

    call = fn.spawn(argv, _new_launch_id("launch-loop"), env_overrides)
    print("=" * 70)
    print(f"Launched on deployed app '{args.app_name}'")
    print(f"  run_name:         {args.run_name}")
    print(f"  iterations:       {args.iterations} (start_from={args.start_from})")
    print(f"  k:                {args.candidates_per_iter}")
    print(f"  concurrency:      {args.concurrency}")
    print(f"  FunctionCall ID:  {call.object_id}")
    print()
    print("Monitor via:")
    print(f"  Dashboard:  https://modal.com/apps/canvas-org/main/{args.app_name}")
    print(f"  Logs CLI:   modal app logs {args.app_name} --function-call {call.object_id}")
    print("=" * 70)


def _cli_main() -> None:
    """Parse tiny CLI for helpers like `upload-codex-auth` and `launch-loop`.

    Direct `python -m meta_agent.cloud.modal_runner ...` is for local helpers that
    talk to Modal's control plane. For one-shot runs, use
    ``modal run meta_agent/cloud/modal_runner.py::loop`` (ephemeral). For
    multi-hour production runs, use ``launch-loop`` here which dispatches
    onto the pre-deployed app with preemption-retry semantics.
    """
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(
            "Modal runner helpers.\n\n"
            "Commands:\n"
            "  upload-codex-auth   read ~/.codex/ and push to Modal as `codex-auth` secret\n"
            "  deploy-app          deploy the Modal app with CLI flags for app/volume names\n"
            "  launch-loop         launch `meta-agent loop` on the deployed app (preferred\n"
            "                      for multi-hour runs; preemption-resilient)\n\n"
            "Deployment (one-time, before first `launch-loop`):\n"
            "  modal deploy meta_agent/cloud/modal_runner.py\n\n"
            "Ephemeral alternative (one-shot, short runs):\n"
            "  modal run meta_agent/cloud/modal_runner.py::loop --benchmark ... --iterations 10\n\n"
            "One-shot eval on Modal:\n"
            "  modal run meta_agent/cloud/modal_runner.py::eval --benchmark ... --config ... --name ...\n"
        )
        return

    cmd = sys.argv[1].replace("_", "-")
    if cmd == "upload-codex-auth":
        upload_codex_auth()
        return
    if cmd == "deploy-app":
        deploy_app()
        return
    if cmd == "launch-loop":
        launch_loop()
        return
    raise SystemExit(f"unknown command: {sys.argv[1]!r}. See --help.")


if __name__ == "__main__":
    _cli_main()
