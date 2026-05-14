"""Shared model client helpers for program-harness model calls.

All Anthropic/Claude API access in this project routes through AWS Bedrock
via boto3's `bedrock-runtime` client. We deliberately do NOT use the
Anthropic Python SDK (`AsyncAnthropicBedrock`) because as of anthropic==0.96
it doesn't support `AWS_BEARER_TOKEN_BEDROCK` env auth — it falls back to
IAM-identity SigV4 signing, which hits a different Anthropic use-case gate
than the bearer-token identity tied to our account's Bedrock approvals.

boto3 natively understands `AWS_BEARER_TOKEN_BEDROCK` (via its credential
chain) in addition to standard AWS keys / profiles / SSO / IAM roles, so it
works for every auth style the researcher might have.

Auth is whatever boto3's default credential chain finds:
- `AWS_BEARER_TOKEN_BEDROCK` (preferred for this project)
- Standard AWS keys (`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`)
- Profile (`AWS_PROFILE`) / SSO session / IAM role, etc.

Region: `AWS_REGION`, defaults to `us-east-1`.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from botocore.client import BaseClient


# ---------------------------------------------------------------------------
# Model ID resolution
# ---------------------------------------------------------------------------

# Canonical short name -> Bedrock Global cross-region inference profile.
# Uses the `global.` prefix because that's what our account is provisioned
# for (10M TPM / 10K RPM Global Cross-Region). If you need a different geo,
# change the prefix here.
#
# To add a model: one line + verify the id appears in `aws bedrock list-
# inference-profiles` for your account.
BEDROCK_MODEL_MAP: dict[str, str] = {
    "claude-haiku-4-5":   "global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6":  "global.anthropic.claude-sonnet-4-6",
    "claude-opus-4-6":    "global.anthropic.claude-opus-4-6-v1",
}


def resolve_bedrock_model(model: str) -> str:
    """Map a short model name to its Bedrock id; pass through unknown ids.

    Pass-through exists so a researcher can already provide a Bedrock id
    directly (e.g. `global.anthropic.claude-opus-4-6-v1`) and have it work.
    """
    if not model:
        return model
    return BEDROCK_MODEL_MAP.get(model, model)


def _looks_like_bedrock_model(model: str) -> bool:
    if model in BEDROCK_MODEL_MAP:
        return True
    normalized = model.lower()
    return (
        normalized.startswith("bedrock/")
        or normalized.startswith("global.anthropic.")
        or normalized.startswith("anthropic.")
        or normalized.startswith("us.anthropic.")
    )


def _looks_like_azure_or_openai_model(model: str) -> bool:
    normalized = model.strip().lower()
    if normalized.startswith(("azure:", "azure/", "openai/")):
        return True
    azure_deployment = os.environ.get("AZURE_GPT55_DEPLOYMENT", "").strip()
    return bool(azure_deployment and model == azure_deployment)


def _looks_like_tinker_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("tinker/", "tinker:"))


def _looks_like_tinker_chat_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("tinker-chat/", "tinker-chat:"))


def _looks_like_tinker_sampling_model(model: str) -> bool:
    normalized = model.strip().lower()
    return normalized.startswith(("tinker-sampling/", "tinker-sampling:"))


def _tinker_base_model(model: str) -> str:
    if model.startswith("tinker-chat/"):
        return model.split("/", 1)[1]
    if model.startswith("tinker-chat:"):
        return model.split(":", 1)[1]
    if model.startswith("tinker-sampling/"):
        return model.split("/", 1)[1]
    if model.startswith("tinker-sampling:"):
        return model.split(":", 1)[1]
    if model.startswith("tinker/"):
        return model.split("/", 1)[1]
    if model.startswith("tinker:"):
        return model.split(":", 1)[1]
    return model


# ---------------------------------------------------------------------------
# boto3 bedrock-runtime client
# ---------------------------------------------------------------------------

_BOTO3_IMPORT_LOCK = threading.Lock()


def _load_boto3_module() -> Any:
    """Import boto3 under a process-wide lock and validate the public API.

    Program-harness evals can call the model from hundreds of worker threads.
    Importing boto3 lazily inside those threads has shown intermittent partial
    module states in Modal (`module 'boto3' has no attribute 'client'`). Import
    and validate under a lock; if a broken partial module is present, evict and
    import once more before surfacing a readable error.
    """
    with _BOTO3_IMPORT_LOCK:
        boto3 = importlib.import_module("boto3")
        client = getattr(boto3, "client", None)
        if callable(client):
            return boto3

        sys.modules.pop("boto3", None)
        boto3 = importlib.import_module("boto3")
        client = getattr(boto3, "client", None)
        if not callable(client):
            module_file = getattr(boto3, "__file__", "<unknown>")
            raise RuntimeError(
                "boto3 imported without a callable client attribute "
                f"(module={module_file})"
            )
        return boto3


def bedrock_runtime_client() -> "BaseClient":
    """Return a fresh boto3 `bedrock-runtime` client. Callers own its lifetime.

    Region comes from `AWS_REGION`. Auth comes from boto3's credential chain.
    """
    boto3 = _load_boto3_module()

    region = os.environ.get("AWS_REGION") or os.environ.get("BEDROCK_REGION") or "us-east-1"
    return boto3.client("bedrock-runtime", region_name=region)


# ---------------------------------------------------------------------------
# Convenience: async Claude invocation
# ---------------------------------------------------------------------------

_ANTHROPIC_BEDROCK_VERSION = "bedrock-2023-05-31"


async def invoke_claude(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Async Claude call via Bedrock. Returns the raw response JSON.

    boto3's `bedrock-runtime` client is synchronous, so we offload the call
    to a worker thread. Concurrency at the adapter level (`asyncio.gather` +
    `Semaphore`) still bounds fan-out cleanly — boto3's connection pool
    tolerates a large thread count.

    The returned dict follows Anthropic's Messages API shape, e.g.
    ``{"content": [{"type": "text", "text": "..."}], "usage": {...}}``.
    Callers typically want ``_extract_text(response)``.
    """
    body: dict[str, Any] = {
        "anthropic_version": _ANTHROPIC_BEDROCK_VERSION,
        "max_tokens": max_tokens,
        "messages": list(messages),
    }
    if system is not None:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    if extra_body:
        body.update(extra_body)

    encoded = json.dumps(body)
    model_id = resolve_bedrock_model(model)

    def _invoke() -> dict[str, Any]:
        client = bedrock_runtime_client()
        resp = client.invoke_model(modelId=model_id, body=encoded)
        return json.loads(resp["body"].read())

    return await asyncio.to_thread(_invoke)


@dataclass(frozen=True)
class _OpenAIModelConfig:
    model: str
    base_url: str
    api_key: str


def _resolve_openai_model_config(model: str) -> _OpenAIModelConfig:
    """Resolve Azure/OpenAI-compatible model settings from env vars.

    The Codex proposer path already uses ``AZURE_OPENAI_V1_BASE`` /
    ``AZURE_FOUNDRY_OPENAI_BASE``. Program-harness calls share those secrets
    so pointwise reward harnesses can be evaluated with Azure GPT deployments.
    """
    requested = model
    deployment = model
    if model.startswith("azure:"):
        deployment = model.split(":", 1)[1]
    elif model.startswith("azure/"):
        deployment = model.split("/", 1)[1]
    elif model.startswith("openai/"):
        deployment = model.split("/", 1)[1]
    else:
        azure_deployment = os.environ.get("AZURE_GPT55_DEPLOYMENT", "").strip()
        if azure_deployment and model == azure_deployment:
            deployment = azure_deployment

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
            f"Azure/OpenAI model requested ({requested}) but no OpenAI-compatible "
            "base URL is configured. Set AZURE_OPENAI_V1_BASE, "
            "AZURE_FOUNDRY_OPENAI_BASE, or AZURE_API_BASE."
        )

    api_key = (
        os.environ.get("AZURE_FOUNDRY_API_KEY", "").strip()
        or os.environ.get("AZURE_API_KEY", "").strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not api_key:
        raise RuntimeError(
            f"Azure/OpenAI model requested ({requested}) but no API key is "
            "configured. Set AZURE_API_KEY, AZURE_FOUNDRY_API_KEY, or OPENAI_API_KEY."
        )

    return _OpenAIModelConfig(
        model=deployment,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
    )


def _openai_tools_from_extra_body(
    extra_body: Mapping[str, Any] | None,
) -> tuple[list[dict[str, Any]], str | dict[str, Any] | None]:
    if not extra_body:
        return [], None

    tools: list[dict[str, Any]] = []
    for tool in extra_body.get("tools", []) if isinstance(extra_body.get("tools"), list) else []:
        if not isinstance(tool, Mapping):
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = tool.get("input_schema")
        if not isinstance(parameters, Mapping):
            parameters = {"type": "object", "properties": {}}
        tools.append({
            "type": "function",
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": dict(parameters),
        })

    raw_choice = extra_body.get("tool_choice")
    tool_choice: str | dict[str, Any] | None = None
    if isinstance(raw_choice, Mapping):
        name = raw_choice.get("name")
        if isinstance(name, str) and name:
            tool_choice = {"type": "function", "name": name}
    elif isinstance(raw_choice, str):
        tool_choice = raw_choice

    return tools, tool_choice


def _openai_chat_tool_choice(tool_choice: str | dict[str, Any] | None) -> str | dict[str, Any] | None:
    if not isinstance(tool_choice, Mapping):
        return tool_choice
    name = tool_choice.get("name")
    if isinstance(name, str) and name:
        return {"type": "function", "function": {"name": name}}
    return tool_choice


def _openai_chat_tools(tools: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            out.append(dict(tool))
            continue
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description") or ""),
                "parameters": dict(tool.get("parameters") or {}),
            },
        })
    return out


def _messages_for_openai(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content", "")
        out.append({"role": role, "content": content if isinstance(content, str) else str(content)})
    return out


def _messages_for_openai_chat(
    *,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    out.extend(_messages_for_openai(messages))
    return out


def _json_from_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        return dict(arguments)
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _forced_tool_from_extra_body(
    extra_body: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    if not extra_body:
        return None

    raw_choice = extra_body.get("tool_choice")
    tool_name: str | None = None
    if isinstance(raw_choice, Mapping):
        name = raw_choice.get("name")
        if isinstance(name, str) and name:
            tool_name = name
    elif isinstance(raw_choice, str) and raw_choice:
        tool_name = raw_choice
    if tool_name is None:
        return None

    raw_tools = extra_body.get("tools")
    if not isinstance(raw_tools, list):
        return None
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        if raw_tool.get("name") != tool_name:
            continue
        schema = raw_tool.get("input_schema")
        return tool_name, dict(schema) if isinstance(schema, Mapping) else {}
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            obj, _ = decoder.raw_decode(cleaned[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _coerce_tool_input(obj: dict[str, Any], tool_name: str) -> dict[str, Any]:
    wrapped = obj.get(tool_name)
    if isinstance(wrapped, Mapping):
        return dict(wrapped)
    for key in ("arguments", "input", "parameters"):
        value = obj.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return obj


def _normalize_openai_responses_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    content_blocks: list[dict[str, Any]] = []
    output = raw.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            item_type = item.get("type")
            if item_type == "function_call":
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                content_blocks.append({
                    "type": "tool_use",
                    "id": str(item.get("call_id") or item.get("id") or ""),
                    "name": name,
                    "input": _json_from_tool_arguments(item.get("arguments")),
                })
                continue
            if item_type != "message":
                continue
            message_content = item.get("content")
            if not isinstance(message_content, list):
                continue
            for block in message_content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        content_blocks.append({"type": "text", "text": text})

    # Some SDK/proxy layers expose a convenience output_text field.
    text = raw.get("output_text")
    if isinstance(text, str) and text and not any(b.get("type") == "text" for b in content_blocks):
        content_blocks.append({"type": "text", "text": text})

    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    return {
        "content": content_blocks,
        "usage": {
            "input_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "output_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
        },
        "model": raw.get("model"),
        "provider": "openai-compatible",
        "provider_raw": dict(raw),
    }


def _normalize_openai_chat_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Compatibility helper for legacy chat-completions-shaped responses."""
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, Mapping) else {}
    if not isinstance(message, Mapping):
        message = {}

    content_blocks: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content_blocks.append({"type": "text", "text": text})

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            if not isinstance(function, Mapping):
                continue
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            content_blocks.append({
                "type": "tool_use",
                "id": str(call.get("id") or ""),
                "name": name,
                "input": _json_from_tool_arguments(function.get("arguments")),
            })

    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    return {
        "content": content_blocks,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        },
        "model": raw.get("model"),
        "provider": "openai-compatible",
        "provider_raw": dict(raw),
    }


def _normalize_tinker_sample_response(
    *,
    model: str,
    text: str,
    input_tokens: int,
    output_tokens: int,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    forced_tool = _forced_tool_from_extra_body(extra_body)
    content_blocks: list[dict[str, Any]] = []
    if forced_tool is not None:
        tool_name, _schema = forced_tool
        tool_input = _extract_json_object(text)
        if tool_input is not None:
            tool_input = _coerce_tool_input(tool_input, tool_name)
            content_blocks.append({
                "type": "tool_use",
                "id": "tinker_synth_tool_0",
                "name": tool_name,
                "input": tool_input,
            })

    if not content_blocks and text:
        content_blocks.append({"type": "text", "text": text})

    return {
        "content": content_blocks,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
        "model": model,
        "provider": "tinker",
        "provider_raw": {
            "text": text,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


_KIMI_TOOL_SECTION_RE = re.compile(
    r"<\|tool_calls_section_begin\|>(.*?)<\|tool_calls_section_end\|>",
    flags=re.DOTALL,
)
_KIMI_TOOL_CALL_RE = re.compile(
    r"<\|tool_call_begin\|>\s*"
    r"(?P<tool_call_id>[\w.-]+(?:\.[\w-]+)?(?::\d+)?)\s*"
    r"<\|tool_call_argument_begin\|>\s*"
    r"(?P<arguments>.*?)\s*"
    r"<\|tool_call_end\|>",
    flags=re.DOTALL,
)
_QWEN_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^>\n]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    flags=re.DOTALL,
)
_QWEN_XML_PARAM_RE = re.compile(
    r"<parameter=(?P<name>[^>\n]+)>\s*(?P<value>.*?)\s*</parameter>",
    flags=re.DOTALL,
)


def _name_from_kimi_tool_call_id(tool_call_id: str) -> str:
    name = tool_call_id
    if "." in name:
        name = name.split(".", 1)[1]
    if ":" in name:
        name = name.split(":", 1)[0]
    return name


def _parse_kimi_native_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for section_match in _KIMI_TOOL_SECTION_RE.finditer(text):
        section = section_match.group(1)
        for call_match in _KIMI_TOOL_CALL_RE.finditer(section):
            tool_call_id = call_match.group("tool_call_id").strip()
            arguments = call_match.group("arguments").strip()
            calls.append({
                "type": "tool_use",
                "id": tool_call_id,
                "name": _name_from_kimi_tool_call_id(tool_call_id),
                "input": _json_from_tool_arguments(arguments),
            })
    return calls


def _coerce_xml_param_value(value: str) -> Any:
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    try:
        as_float = float(stripped)
    except ValueError:
        return stripped
    if as_float.is_integer():
        return int(as_float)
    return as_float


def _parse_qwen_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for idx, call_match in enumerate(_QWEN_XML_TOOL_CALL_RE.finditer(text)):
        tool_name = call_match.group("name").strip()
        body = call_match.group("body")
        arguments: dict[str, Any] = {}
        for param_match in _QWEN_XML_PARAM_RE.finditer(body):
            param_name = param_match.group("name").strip()
            arguments[param_name] = _coerce_xml_param_value(param_match.group("value"))
        calls.append({
            "type": "tool_use",
            "id": f"qwen_xml_tool:{idx}",
            "name": tool_name,
            "input": arguments,
        })
    return calls


def _kimi_native_tool_prefill(tool_name: str) -> str:
    return (
        "<|tool_calls_section_begin|> "
        f"<|tool_call_begin|> functions.{tool_name}:0 "
        "<|tool_call_argument_begin|> {"
    )


def _qwen_xml_tool_prefill(tool_name: str) -> str:
    return f"<tool_call>\n<function={tool_name}>\n"


def _forced_tool_prefill(model: str, tool_name: str) -> str:
    base_model = _tinker_base_model(model).lower()
    if "qwen3.5" in base_model or "qwen3.6" in base_model:
        return _qwen_xml_tool_prefill(tool_name)
    return _kimi_native_tool_prefill(tool_name)


def _normalize_tinker_completions_response(
    raw: Mapping[str, Any],
    model: str = "",
    forced_tool: tuple[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    choices = raw.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices else {}
    text = choice.get("text") if isinstance(choice, Mapping) else None
    text = text if isinstance(text, str) else ""
    content_blocks = _parse_kimi_native_tool_calls(text) or _parse_qwen_xml_tool_calls(text)
    if not content_blocks and forced_tool is not None:
        tool_name, _schema = forced_tool
        content_blocks = _parse_kimi_native_tool_calls(
            _forced_tool_prefill(model, tool_name) + text
        ) or _parse_qwen_xml_tool_calls(_forced_tool_prefill(model, tool_name) + text)
    if not content_blocks and text:
        content_blocks.append({"type": "text", "text": text})
    usage = raw.get("usage") if isinstance(raw.get("usage"), Mapping) else {}
    return {
        "content": content_blocks,
        "usage": {
            "input_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "output_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
        },
        "model": raw.get("model"),
        "provider": "tinker-openai-compatible-completions",
        "provider_raw": dict(raw),
    }


def _openai_responses_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tools, tool_choice = _openai_tools_from_extra_body(extra_body)
    payload: dict[str, Any] = {
        "model": model,
        "input": _messages_for_openai(messages),
        # Reasoning models count internal reasoning against completion tokens.
        # Program harnesses often request 256 Claude tokens, but GPT-5.x can
        # spend far more before emitting the forced tool/function call.
        "max_output_tokens": max(max_tokens, 4096),
    }
    if system is not None:
        payload["instructions"] = system
    if temperature is not None and temperature != 0:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice
    return payload


_TINKER_OPENAI_BASE_URL = "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/v1"


def _tinker_api_key() -> str:
    api_key = os.environ.get("TINKER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Tinker model requested but TINKER_API_KEY is not configured.")
    return api_key


def _tinker_openai_chat_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tools, tool_choice = _openai_tools_from_extra_body(extra_body)
    payload: dict[str, Any] = {
        "model": _tinker_base_model(model),
        "messages": _messages_for_openai_chat(messages=messages, system=system),
        "max_tokens": max(max_tokens, 4096) if tools else max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = _openai_chat_tools(tools)
    chat_tool_choice = _openai_chat_tool_choice(tool_choice)
    if chat_tool_choice is not None:
        payload["tool_choice"] = chat_tool_choice
    return payload


def _normalize_tinker_openai_chat_response(raw: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_openai_chat_response(raw)
    normalized["provider"] = "tinker-openai-compatible"
    normalized["provider_raw"] = dict(raw)
    return normalized


_TINKER_TOKENIZER_LOCK = threading.Lock()
_TINKER_TOKENIZERS: dict[str, Any] = {}


def _get_tinker_tokenizer(base_model: str) -> Any:
    with _TINKER_TOKENIZER_LOCK:
        cached = _TINKER_TOKENIZERS.get(base_model)
        if cached is not None:
            return cached

        try:
            import tinker  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Tinker model requested but the `tinker` package is not installed. "
                "Install it with `pip install tinker` and set TINKER_API_KEY."
            ) from exc

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=base_model)
        tokenizer = sampling_client.get_tokenizer()
        _TINKER_TOKENIZERS[base_model] = tokenizer
        return tokenizer


def _apply_chat_template_compat(tokenizer: Any, kwargs: dict[str, Any]) -> str:
    attempts = [
        dict(kwargs),
        {k: v for k, v in kwargs.items() if k != "enable_thinking"},
        {k: v for k, v in kwargs.items() if k not in {"enable_thinking", "tools"}},
    ]
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            rendered = tokenizer.apply_chat_template(**attempt)
        except TypeError as exc:
            last_error = exc
            continue
        if isinstance(rendered, str):
            return rendered
        if hasattr(tokenizer, "decode"):
            return str(tokenizer.decode(rendered, skip_special_tokens=False))
        return str(rendered)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to render chat template")


def _tinker_completions_prompt(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    extra_body: Mapping[str, Any] | None,
) -> str:
    rendered = _tinker_completions_prompt_with_renderer(
        model=model,
        messages=messages,
        system=system,
        extra_body=extra_body,
    )
    if rendered is not None:
        return rendered

    tokenizer = _get_tinker_tokenizer(_tinker_base_model(model))
    tools, _tool_choice = _openai_tools_from_extra_body(extra_body)
    template_kwargs: dict[str, Any] = {
        "conversation": _messages_for_openai_chat(messages=messages, system=system),
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    if tools:
        template_kwargs["tools"] = _openai_chat_tools(tools)
    return _apply_chat_template_compat(tokenizer, template_kwargs)


def _renderer_tool_specs(extra_body: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    tools, _tool_choice = _openai_tools_from_extra_body(extra_body)
    out: list[dict[str, Any]] = []
    for tool in tools:
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            continue
        out.append({
            "name": name,
            "description": str(tool.get("description") or ""),
            "parameters": dict(tool.get("parameters") or {}),
        })
    return out


def _native_tool_instruction(extra_body: Mapping[str, Any] | None) -> str | None:
    forced_tool = _forced_tool_from_extra_body(extra_body)
    if forced_tool is None:
        return None
    tool_name, _schema = forced_tool
    return (
        f"You must call the `{tool_name}` tool exactly once. Do not write prose, "
        "analysis, or a final answer before the tool call."
    )


def _tinker_completions_prompt_with_renderer(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    extra_body: Mapping[str, Any] | None,
) -> str | None:
    base_model = _tinker_base_model(model)
    base_model_lower = base_model.lower()
    renderer_name: str | None = None
    if base_model_lower.startswith("moonshotai/kimi-k2.6"):
        renderer_name = "kimi_k26_disable_thinking"
    elif "qwen3.5" in base_model_lower or "qwen3.6" in base_model_lower:
        renderer_name = "qwen3_5_disable_thinking"
    if renderer_name is None:
        return None
    try:
        from tinker_cookbook import renderers  # type: ignore[import-not-found]
    except ImportError:
        return None

    tokenizer = _get_tinker_tokenizer(base_model)
    renderer = renderers.get_renderer(renderer_name, tokenizer, model_name=base_model)
    system_parts = [part for part in [system, _native_tool_instruction(extra_body)] if part]
    system_prompt = "\n\n".join(system_parts)
    renderer_messages: list[dict[str, Any]] = []
    create_prefix = getattr(renderer, "create_conversation_prefix_with_tools", None)
    if callable(create_prefix):
        renderer_messages.extend(create_prefix(_renderer_tool_specs(extra_body), system_prompt))
    elif system_prompt:
        renderer_messages.append({"role": "system", "content": system_prompt})
    for message in messages:
        renderer_messages.append({
            "role": str(message.get("role") or "user"),
            "content": _content_to_text(message.get("content", "")),
        })
    prompt_input = renderer.build_generation_prompt(renderer_messages)
    return str(tokenizer.decode(prompt_input.to_ints(), skip_special_tokens=False))


def _tinker_completions_payload(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    max_tokens: int,
    temperature: float | None,
    extra_body: Mapping[str, Any] | None,
) -> dict[str, Any]:
    tools, _tool_choice = _openai_tools_from_extra_body(extra_body)
    forced_tool = _forced_tool_from_extra_body(extra_body)
    prompt = _tinker_completions_prompt(
        model=model,
        messages=messages,
        system=system,
        extra_body=extra_body,
    )
    if forced_tool is not None:
        tool_name, _schema = forced_tool
        prompt += _forced_tool_prefill(model, tool_name)
    payload: dict[str, Any] = {
        "model": _tinker_base_model(model),
        "prompt": prompt,
        "max_tokens": max(max_tokens, 512) if tools else max_tokens,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


async def invoke_tinker_completions(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Tinker's completions endpoint using a rendered Kimi no-thinking prompt."""
    import httpx

    payload = _tinker_completions_payload(
        model=model,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    api_key = _tinker_api_key()
    base_url = os.environ.get("TINKER_OPENAI_BASE_URL", _TINKER_OPENAI_BASE_URL).rstrip("/")

    def _invoke() -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=300.0) as client:
            response: httpx.Response | None = None
            max_retries = 10
            backoff_cap_s = 60.0
            for attempt in range(max_retries):
                response = client.post(
                    f"{base_url}/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
                if attempt == max_retries - 1:
                    break
                retry_after = response.headers.get("retry-after")
                try:
                    retry_after_secs = float(retry_after) if retry_after else 0.0
                except ValueError:
                    retry_after_secs = 0.0
                if retry_after_secs > 0:
                    delay = retry_after_secs + random.uniform(0.0, 1.0)
                else:
                    base = min(2.0 * (2 ** attempt), backoff_cap_s)
                    delay = base / 2 + random.uniform(0.0, base / 2)
                time.sleep(delay)
            assert response is not None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            raise RuntimeError(
                f"Tinker OpenAI-compatible completions request failed: "
                f"{response.status_code} {body}"
            ) from exc
        return _normalize_tinker_completions_response(
            response.json(),
            model=model,
            forced_tool=_forced_tool_from_extra_body(extra_body),
        )

    return await asyncio.to_thread(_invoke)


async def invoke_tinker_openai_compatible(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Tinker's OpenAI-compatible chat endpoint with native tool support."""
    import httpx

    payload = _tinker_openai_chat_payload(
        model=model,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )
    api_key = _tinker_api_key()
    base_url = os.environ.get("TINKER_OPENAI_BASE_URL", _TINKER_OPENAI_BASE_URL).rstrip("/")

    def _invoke() -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=300.0) as client:
            response: httpx.Response | None = None
            max_retries = 10
            backoff_cap_s = 60.0
            for attempt in range(max_retries):
                response = client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
                if attempt == max_retries - 1:
                    break
                retry_after = response.headers.get("retry-after")
                try:
                    retry_after_secs = float(retry_after) if retry_after else 0.0
                except ValueError:
                    retry_after_secs = 0.0
                if retry_after_secs > 0:
                    delay = retry_after_secs + random.uniform(0.0, 1.0)
                else:
                    base = min(2.0 * (2 ** attempt), backoff_cap_s)
                    delay = base / 2 + random.uniform(0.0, base / 2)
                time.sleep(delay)
            assert response is not None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            raise RuntimeError(
                f"Tinker OpenAI-compatible chat request failed: "
                f"{response.status_code} {body}"
            ) from exc
        return _normalize_tinker_openai_chat_response(response.json())

    return await asyncio.to_thread(_invoke)


@dataclass
class _TinkerModelResources:
    sampling_client: Any
    tokenizer: Any


_TINKER_CLIENT_LOCK = threading.Lock()
_TINKER_CLIENTS: dict[str, _TinkerModelResources] = {}


def _get_tinker_resources(base_model: str) -> _TinkerModelResources:
    with _TINKER_CLIENT_LOCK:
        cached = _TINKER_CLIENTS.get(base_model)
        if cached is not None:
            return cached

        try:
            import tinker  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Tinker model requested but the `tinker` package is not installed. "
                "Install it with `pip install tinker` and set TINKER_API_KEY."
            ) from exc

        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=base_model)
        resources = _TinkerModelResources(
            sampling_client=sampling_client,
            tokenizer=sampling_client.get_tokenizer(),
        )
        _TINKER_CLIENTS[base_model] = resources
        return resources


def _tinker_tool_instruction(extra_body: Mapping[str, Any] | None) -> str | None:
    forced_tool = _forced_tool_from_extra_body(extra_body)
    if forced_tool is None:
        return None
    tool_name, schema = forced_tool
    schema_text = json.dumps(schema, sort_keys=True)
    return (
        "You are being called through a tool-use compatibility layer. "
        f"The requested tool is `{tool_name}`. Respond with only one valid JSON "
        "object containing the arguments for that tool. Do not include markdown, "
        "prose, code fences, or a wrapper object. The JSON object must satisfy "
        f"this schema: {schema_text}"
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _messages_for_tinker(
    *,
    messages: Sequence[Mapping[str, Any]],
    system: str | None,
    extra_body: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    tool_instruction = _tinker_tool_instruction(extra_body)
    system_parts = [part for part in [system, tool_instruction] if part]
    out: list[dict[str, str]] = []
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    for message in messages:
        role = str(message.get("role") or "user")
        out.append({"role": role, "content": _content_to_text(message.get("content", ""))})
    return out


def _encode_tinker_prompt(tokenizer: Any, chat_messages: Sequence[Mapping[str, str]]) -> list[int]:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        rendered = apply_chat_template(
            list(chat_messages),
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(rendered, Mapping):
            input_ids = rendered.get("input_ids")
            if hasattr(input_ids, "tolist"):
                return list(input_ids.tolist())
            if isinstance(input_ids, Sequence):
                return [int(token) for token in input_ids]
        if hasattr(rendered, "tolist"):
            return list(rendered.tolist())
        return list(rendered)

    prompt_text = "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in chat_messages
    )
    prompt_text = f"{prompt_text}\n\nASSISTANT:\n"
    return list(tokenizer.encode(prompt_text))


def _encode_tinker_text(tokenizer: Any, text: str) -> list[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def _decode_tinker_tokens(tokenizer: Any, tokens: Sequence[int]) -> str:
    return str(tokenizer.decode(list(tokens), skip_special_tokens=True)).strip()


async def invoke_tinker(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Tinker SamplingClient and normalize output to the program-harness shape."""
    base_model = _tinker_base_model(model)

    def _invoke() -> dict[str, Any]:
        try:
            import tinker  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Tinker model requested but the `tinker` package is not installed. "
                "Install it with `pip install tinker` and set TINKER_API_KEY."
            ) from exc

        resources = _get_tinker_resources(base_model)
        chat_messages = _messages_for_tinker(
            messages=messages,
            system=system,
            extra_body=extra_body,
        )
        prompt_tokens = _encode_tinker_prompt(resources.tokenizer, chat_messages)
        response_prefix = "{" if _forced_tool_from_extra_body(extra_body) else ""
        if response_prefix:
            prompt_tokens = prompt_tokens + _encode_tinker_text(resources.tokenizer, response_prefix)
        prompt = tinker.ModelInput.from_ints(prompt_tokens)
        effective_max_tokens = max(max_tokens, 512) if _forced_tool_from_extra_body(extra_body) else max_tokens
        sampling_params = tinker.SamplingParams(
            max_tokens=effective_max_tokens,
            temperature=0.0 if temperature is None else temperature,
        )
        result = resources.sampling_client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()
        sequence = result.sequences[0]
        output_tokens = list(sequence.tokens)
        text = response_prefix + _decode_tinker_tokens(resources.tokenizer, output_tokens)
        return _normalize_tinker_sample_response(
            model=base_model,
            text=text,
            input_tokens=len(prompt_tokens),
            output_tokens=len(output_tokens),
            extra_body=extra_body,
        )

    return await asyncio.to_thread(_invoke)


async def invoke_openai_compatible(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible Responses endpoint and normalize output."""
    import httpx

    cfg = _resolve_openai_model_config(model)
    payload = _openai_responses_payload(
        model=cfg.model,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )

    def _invoke() -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {cfg.api_key}",
            "api-key": cfg.api_key,
            "Content-Type": "application/json",
        }
        with httpx.Client(timeout=300.0) as client:
            response: httpx.Response | None = None
            max_retries = 10
            backoff_cap_s = 60.0
            for attempt in range(max_retries):
                response = client.post(
                    f"{cfg.base_url}/responses",
                    headers=headers,
                    json=payload,
                )
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
                if attempt == max_retries - 1:
                    break
                retry_after = response.headers.get("retry-after")
                try:
                    retry_after_secs = float(retry_after) if retry_after else 0.0
                except ValueError:
                    retry_after_secs = 0.0
                if retry_after_secs > 0:
                    # Honor server hint, but add small jitter so synchronized
                    # callers don't all retry at the exact same instant.
                    delay = retry_after_secs + random.uniform(0.0, 1.0)
                else:
                    # Equal-jitter exponential backoff: half deterministic,
                    # half random in [0, base/2). Caps at backoff_cap_s.
                    base = min(2.0 * (2 ** attempt), backoff_cap_s)
                    delay = base / 2 + random.uniform(0.0, base / 2)
                time.sleep(delay)
            assert response is not None
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            body = response.text[:1000]
            raise RuntimeError(
                f"OpenAI-compatible Responses request failed: "
                f"{response.status_code} {body}"
            ) from exc
        raw = response.json()
        return _normalize_openai_responses_response(raw)

    return await asyncio.to_thread(_invoke)


async def invoke_model(
    *,
    model: str,
    messages: Sequence[Mapping[str, Any]],
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float | None = None,
    extra_body: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Route a program-harness model call to Bedrock, OpenAI-compatible, or Tinker APIs."""
    if _looks_like_tinker_model(model):
        return await invoke_tinker_completions(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    if _looks_like_tinker_chat_model(model):
        return await invoke_tinker_openai_compatible(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    if _looks_like_tinker_sampling_model(model):
        return await invoke_tinker(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    if _looks_like_azure_or_openai_model(model):
        return await invoke_openai_compatible(
            model=model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    if _looks_like_bedrock_model(model):
        bedrock_model = model.split("/", 1)[1] if model.startswith("bedrock/") else model
        return await invoke_claude(
            model=bedrock_model,
            messages=messages,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
    return await invoke_claude(
        model=model,
        messages=messages,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra_body,
    )


def extract_text(response: Mapping[str, Any]) -> str:
    """Pull all `text`-block text from a Bedrock Claude response. Robust to
    empty/malformed content blocks.
    """
    parts: list[str] = []
    for block in response.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Subprocess helpers (for CLI paths like the `claude` proposer)
# ---------------------------------------------------------------------------


def bedrock_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an env dict with `CLAUDE_CODE_USE_BEDROCK=1` forced on.

    Use this when spawning `claude` CLI subprocesses that should route Claude
    model traffic through Bedrock. Preserves any existing env overrides the
    caller wants (`AWS_REGION`, permissions, etc.).
    """
    env = dict(base) if base is not None else dict(os.environ)
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    return env


def ensure_bedrock_env() -> None:
    """Force `CLAUDE_CODE_USE_BEDROCK=1` and a safe SDK init timeout.

    The Claude Agent SDK (`claude_agent_sdk.query`) reads `CLAUDE_CODE_USE_BEDROCK`
    to route Claude model traffic through Bedrock. It also reads
    `CLAUDE_CODE_STREAM_CLOSE_TIMEOUT` (ms, default 60000) as the initialize
    timeout for its control channel handshake — at high concurrency (e.g. 100
    parallel `query()` calls) the Bun subprocess the SDK spawns per call
    can't always complete handshake within 60s, producing
    `Exception: Control request timeout: initialize`.

    We raise that default to 5 min via `setdefault` so callers can still
    override by setting a different value in the environment. Call this
    exactly before any in-process SDK invocation. Safe to call repeatedly.
    """
    os.environ["CLAUDE_CODE_USE_BEDROCK"] = "1"
    os.environ.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "300000")
