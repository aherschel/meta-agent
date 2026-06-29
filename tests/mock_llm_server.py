"""A tiny in-process HTTP server that mocks the OpenRouter (OpenAI-compatible
Chat Completions) and Anthropic (Messages) endpoints.

Used by the provider-routing tests so the whole loop can run end-to-end with no
network, no AWS, and no real API keys. The server records every request so a
test can assert which endpoint was hit, that the auth headers were set, and that
``/responses`` was never called.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class MockState:
    requests: list[RecordedRequest] = field(default_factory=list)
    responses_called: bool = False
    # The plain-text answer the mock returns (used when no tool is forced).
    text_answer: str = "55"
    # When a tool is forced, the arguments the mock returns for the tool call.
    tool_arguments: dict[str, Any] = field(default_factory=lambda: {"score": 1})

    def paths(self) -> list[str]:
        return [r.path for r in self.requests]


def _forced_tool_name_openai(body: dict[str, Any]) -> Optional[str]:
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        fn = choice.get("function")
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
    tools = body.get("tools")
    if isinstance(tools, list) and tools:
        fn = tools[0].get("function") if isinstance(tools[0], dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            return fn["name"]
    return None


def _forced_tool_name_anthropic(body: dict[str, Any]) -> Optional[str]:
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and isinstance(choice.get("name"), str):
        return choice["name"]
    return None


def _make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # silence test noise
            pass

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                return json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return {}

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            body = self._read_body()
            state.requests.append(
                RecordedRequest(
                    path=self.path,
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=body,
                )
            )

            if self.path.endswith("/responses"):
                # OpenRouter does NOT implement /responses; a hit here is a bug.
                state.responses_called = True
                self._send(400, {"error": "responses endpoint not supported"})
                return

            if self.path.endswith("/chat/completions"):
                tool_name = _forced_tool_name_openai(body)
                if tool_name:
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": json.dumps(state.tool_arguments),
                                },
                            }
                        ],
                    }
                else:
                    message = {"role": "assistant", "content": state.text_answer}
                self._send(
                    200,
                    {
                        "id": "chatcmpl-mock",
                        "model": body.get("model"),
                        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
                    },
                )
                return

            if self.path.endswith("/v1/messages"):
                tool_name = _forced_tool_name_anthropic(body)
                if tool_name:
                    content = [
                        {
                            "type": "tool_use",
                            "id": "toolu_0",
                            "name": tool_name,
                            "input": state.tool_arguments,
                        }
                    ]
                else:
                    content = [{"type": "text", "text": state.text_answer}]
                self._send(
                    200,
                    {
                        "id": "msg_mock",
                        "type": "message",
                        "role": "assistant",
                        "model": body.get("model"),
                        "content": content,
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 11, "output_tokens": 7},
                    },
                )
                return

            self._send(404, {"error": f"unknown path {self.path}"})

    return Handler


class MockLLMServer:
    """Context manager that runs the mock server on a background thread."""

    def __init__(self) -> None:
        self.state = MockState()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self.state))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> "MockLLMServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
