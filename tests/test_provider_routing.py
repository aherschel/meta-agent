"""Provider-routing tests: OpenRouter (Chat Completions) + Anthropic (Messages).

These run fully offline against `tests/mock_llm_server.MockLLMServer`. They prove:

* `invoke_model` routes to the selected provider with NO AWS/Bedrock involvement,
* OpenRouter uses `/chat/completions` and NEVER `/responses`,
* Anthropic uses `/v1/messages` with `x-api-key` + `anthropic-version`,
* a 1-task local benchmark runs end-to-end through the program-harness runtime,
* proxy env (`trust_env`) is honored on every outbound call.

Runnable with `pytest tests/test_provider_routing.py` or `python tests/test_provider_routing.py`.
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.mock_llm_server import MockLLMServer  # noqa: E402

_PROVIDER_ENV_KEYS = (
    "META_AGENT_LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_VERSION",
    "META_AGENT_MODEL",
    "META_AGENT_PROPOSER_MODEL",
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "NO_PROXY",
)


@contextlib.contextmanager
def provider_env(**overrides: str):
    """Set provider env vars for the duration of the block, then restore."""
    saved = {k: os.environ.get(k) for k in _PROVIDER_ENV_KEYS}
    for k in _PROVIDER_ENV_KEYS:
        os.environ.pop(k, None)
    os.environ.update({k: v for k, v in overrides.items() if v is not None})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _no_bedrock(monkeypatch=None):
    """Make any Bedrock client construction explode (assert it's never used)."""
    import meta_agent.services.llm as llm

    def _boom(*_a, **_k):
        raise AssertionError("Bedrock client must not be created for non-Bedrock providers")

    if monkeypatch is not None:
        monkeypatch.setattr(llm, "bedrock_runtime_client", _boom)
        monkeypatch.setattr(llm, "_load_boto3_module", _boom)
    else:
        llm.bedrock_runtime_client = _boom  # type: ignore[assignment]
        llm._load_boto3_module = _boom  # type: ignore[assignment]


_FORCED_TOOL = {
    "tools": [
        {
            "name": "record_score",
            "description": "record",
            "input_schema": {"type": "object", "properties": {"score": {"type": "integer"}}},
        }
    ],
    "tool_choice": {"type": "tool", "name": "record_score"},
}


def test_openrouter_routes_chat_completions_only():
    import meta_agent.services.llm as llm

    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test",
        OPENROUTER_BASE_URL=server.base_url,
    ):
        _no_bedrock()
        # Plain text completion.
        resp = asyncio.run(
            llm.invoke_model(
                model="minimax/minimax-m2.7",
                messages=[{"role": "user", "content": "two plus two times..."}],
                system="be terse",
                max_tokens=64,
            )
        )
        assert llm.extract_text(resp) == "55"
        assert resp["provider"] == "openrouter"

        # Forced-tool completion (the program-harness pointwise shape).
        server.state.tool_arguments = {"score": 1}
        tool_resp = asyncio.run(
            llm.invoke_model(
                model="openrouter/minimax/minimax-m2.7",  # routing prefix stripped
                messages=[{"role": "user", "content": "score it"}],
                max_tokens=32,
                extra_body=_FORCED_TOOL,
            )
        )
        tool_blocks = [b for b in tool_resp["content"] if b.get("type") == "tool_use"]
        assert tool_blocks and tool_blocks[0]["name"] == "record_score"
        assert tool_blocks[0]["input"] == {"score": 1}

    paths = server.state.paths()
    assert all(p.endswith("/chat/completions") for p in paths), paths
    assert server.state.responses_called is False
    # raw model slug forwarded (no openrouter/ prefix)
    assert server.state.requests[-1].body["model"] == "minimax/minimax-m2.7"
    # OpenAI function-tool translation happened
    assert server.state.requests[-1].body["tools"][0]["type"] == "function"
    assert "authorization" in server.state.requests[0].headers


def test_anthropic_routes_messages():
    import meta_agent.services.llm as llm

    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        ANTHROPIC_BASE_URL=server.base_url,
    ):
        _no_bedrock()
        resp = asyncio.run(
            llm.invoke_model(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "answer"}],
                system="terse",
                max_tokens=64,
            )
        )
        assert llm.extract_text(resp) == "55"
        assert resp["provider"] == "anthropic"

        server.state.tool_arguments = {"score": 2}
        tool_resp = asyncio.run(
            llm.invoke_model(
                model="claude-opus-4-8",
                messages=[{"role": "user", "content": "score"}],
                max_tokens=32,
                extra_body=_FORCED_TOOL,
            )
        )
        tool_blocks = [b for b in tool_resp["content"] if b.get("type") == "tool_use"]
        assert tool_blocks and tool_blocks[0]["input"] == {"score": 2}

    paths = server.state.paths()
    assert all(p.endswith("/v1/messages") for p in paths), paths
    assert server.state.responses_called is False
    headers = server.state.requests[0].headers
    assert headers.get("x-api-key") == "sk-ant-test"
    assert "anthropic-version" in headers
    # Anthropic tool spec passed through unchanged (still input_schema, not function)
    assert "input_schema" in server.state.requests[-1].body["tools"][0]


def test_proxy_env_is_honored():
    """The provider HTTP client must be created with trust_env=True so the ASP
    secrets-broker proxy (HTTPS_PROXY) is used for every outbound call."""
    import httpx

    import meta_agent.services.llm as llm

    captured: dict = {}
    real_client = httpx.Client

    def _spy_client(*args, **kwargs):
        captured["trust_env"] = kwargs.get("trust_env")
        return real_client(*args, **kwargs)

    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test",
        OPENROUTER_BASE_URL=server.base_url,
        HTTPS_PROXY="http://127.0.0.1:14322",
        NO_PROXY="127.0.0.1,localhost",  # keep the mock reachable
    ):
        _no_bedrock()
        httpx.Client = _spy_client  # type: ignore[assignment]
        try:
            asyncio.run(
                llm.invoke_model(
                    model="minimax/minimax-m2.7",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=16,
                )
            )
        finally:
            httpx.Client = real_client  # type: ignore[assignment]

    assert captured.get("trust_env") is True


def _write_program_harness(tmp: Path) -> Path:
    """A minimal program harness that asks the model and writes the answer."""
    cfg = tmp / "harness"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "harness.py").write_text(
        "from pathlib import Path\n"
        "async def run(ctx):\n"
        "    resp = await ctx.call_model(\n"
        "        system='solve',\n"
        "        messages=[{'role': 'user', 'content': str(ctx.task)}],\n"
        "        max_tokens=32,\n"
        "    )\n"
        "    Path(ctx.cwd, 'answer.txt').write_text(resp.text.strip())\n"
        "    return ctx.finish(resp.text.strip())\n"
    )
    return cfg


def _run_local_task(server: MockLLMServer, tmp: Path, model: str):
    from meta_agent.core.benchmark import Task
    from meta_agent.task_runner import run_task_with_runtime

    cfg = _write_program_harness(tmp)
    work = tmp / "work"
    work.mkdir(parents=True, exist_ok=True)
    task = Task(
        name="echo-answer",
        instruction="Return the answer.",
        workspace=str(work),
        verify=["python", "-c", "assert open('answer.txt').read().strip() == '55'"],
        timeout=30,
    )
    return asyncio.run(
        run_task_with_runtime(
            task=task,
            config_dir=str(cfg),
            model=model,
            work_dir=work,
            runtime="program_harness",
        )
    )


def test_local_benchmark_end_to_end_openrouter(tmp_path):
    server_tmp = tmp_path / "or"
    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test",
        OPENROUTER_BASE_URL=server.base_url,
    ):
        _no_bedrock()
        server.state.text_answer = "55"
        result = _run_local_task(server, server_tmp, "minimax/minimax-m2.7")

    assert result.passed is True
    assert server.state.responses_called is False
    assert any(p.endswith("/chat/completions") for p in server.state.paths())


def test_local_benchmark_end_to_end_anthropic(tmp_path):
    server_tmp = tmp_path / "an"
    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="sk-ant-test",
        ANTHROPIC_BASE_URL=server.base_url,
    ):
        _no_bedrock()
        server.state.text_answer = "55"
        result = _run_local_task(server, server_tmp, "claude-opus-4-8")

    assert result.passed is True
    assert server.state.responses_called is False
    assert any(p.endswith("/v1/messages") for p in server.state.paths())


def test_inprocess_proposer_emits_candidate(tmp_path):
    """The CLI-free proposer writes a candidate harness via a forced-tool call —
    no `claude`/`codex` binary required."""
    from meta_agent.core.targets import get_target
    from meta_agent.loop.inprocess_proposer import run_inprocess_proposer

    staging = tmp_path / "staging"
    staging.mkdir()
    target = get_target("program_harness")
    harness_src = "async def run(ctx):\n    return ctx.finish('ok')\n"

    with MockLLMServer() as server, provider_env(
        META_AGENT_LLM_PROVIDER="openrouter",
        OPENROUTER_API_KEY="sk-or-test",
        OPENROUTER_BASE_URL=server.base_url,
    ):
        _no_bedrock()
        server.state.tool_arguments = {
            "files": [{"path": "harness.py", "content": harness_src}],
            "notes": {"hypothesis": "test"},
        }
        run = run_inprocess_proposer(
            prompt="optimize the harness",
            system_append="follow the contract",
            staging_dir=staging,
            target=target,
            model="minimax/minimax-m2.7",
        )

    assert run.exit_code == 0
    assert (staging / "harness.py").read_text() == harness_src
    assert (staging / "proposal_notes.json").exists()
    assert server.state.responses_called is False


def _run_all():
    import tempfile

    simple = [
        test_openrouter_routes_chat_completions_only,
        test_anthropic_routes_messages,
        test_proxy_env_is_honored,
    ]
    tmp_based = [
        test_local_benchmark_end_to_end_openrouter,
        test_local_benchmark_end_to_end_anthropic,
        test_inprocess_proposer_emits_candidate,
    ]
    failures = 0
    for fn in simple:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    for fn in tmp_based:
        with tempfile.TemporaryDirectory() as d:
            try:
                fn(Path(d))
                print(f"PASS {fn.__name__}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
