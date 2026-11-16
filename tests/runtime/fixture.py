"""Deterministic OpenAI Chat Completions SSE fixture for native runtime tests.

This is a protocol instrument, not a model. It lets the official OpenCode
binary and the bundled companion exercise the real boundary without a hosted
request. No fixture is registered by production runtime construction.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "input_text", "output_text"}
        )
    return ""


class Fixture:
    """One loopback fixture server; `modes` is consumed per model request."""

    def __init__(self, modes: list[str] | None = None, *, marker: str | None = None):
        self.modes = modes or ["text_stop"]
        self.marker = marker
        self.delay = 0.0
        self.records: list[dict] = []
        self.lock = threading.Lock()
        self.chunk_sizes = [1, 2, 3, 5, 8, 13, 21, 34, 55]
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args):
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                with fixture.lock:
                    record = {
                        "seq": len(fixture.records) + 1,
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "content_type": self.headers.get("content-type"),
                        "headers": {str(key).lower(): value for key, value in self.headers.items()},
                        "body": body,
                    }
                    fixture.records.append(record)
                    index = len(fixture.records) - 1
                if not self.path.endswith("/chat/completions"):
                    self.send_error(404)
                    return
                if fixture.delay:
                    time.sleep(fixture.delay)
                mode = "text_stop" if fixture.has_tool_results(body) else fixture.modes[min(index, len(fixture.modes) - 1)]
                payload = b"".join(fixture.sse(body, mode))
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache")
                self.send_header("connection", "close")
                self.end_headers()
                offset = position = 0
                while offset < len(payload):
                    size = fixture.chunk_sizes[position % len(fixture.chunk_sizes)]
                    self.wfile.write(payload[offset : offset + size])
                    self.wfile.flush()
                    offset += size
                    position += 1
                    time.sleep(0.001)
                self.close_connection = True

            def do_GET(self):
                if self.path.split("?")[0] != "/__requests":
                    self.send_error(404)
                    return
                payload = json.dumps(fixture.records).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}/v1"

    @property
    def request_count(self) -> int:
        return len(self.records)

    def start(self) -> "Fixture":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def has_tool_results(body: dict) -> bool:
        return any(message.get("role") == "tool" for message in body.get("messages", []))

    @staticmethod
    def _chunk(model: str, choices: list, usage: dict | None = None) -> bytes:
        payload = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": choices,
        }
        if usage is not None:
            payload["usage"] = usage
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()

    @staticmethod
    def _usage() -> dict:
        return {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18}

    def _tool_call_events(self, model: str, call_id: str = "call_write_1", *, usage: bool = True):
        yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": "hello ✓ 你好"}, "finish_reason": None}])
        arguments = json.dumps({"path": self.marker, "content": "fixture-native"}, ensure_ascii=False)
        midpoint = max(1, len(arguments) // 2)
        yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function", "function": {"name": "write", "arguments": arguments[:midpoint]}}]}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": arguments[midpoint:]}}]}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        if usage:
            yield self._chunk(model, [], self._usage())

    def _text_events(self, model: str, text: str, finish: str = "stop", usage: dict | None = None,
                     reasoning: str | None = None):
        if reasoning is not None:
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "reasoning_content": reasoning}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": finish}])
        if usage is not None:
            yield self._chunk(model, [], usage)
        yield b"data: [DONE]\n\n"

    def _execute_events(self, model: str, code: str):
        yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_exec_1", "type": "function", "function": {"name": "execute", "arguments": json.dumps({"code": code})}}]}, "finish_reason": None}])
        yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
        yield self._chunk(model, [], self._usage())
        yield b"data: [DONE]\n\n"

    def sse(self, body: dict, mode: str):
        model = body.get("model", "unknown")
        results = [message for message in body.get("messages", []) if message.get("role") == "tool"]
        if results:
            text = "fixture final: " + "; ".join(_content_text(message.get("content")) for message in results)
            yield from self._text_events(model, text, usage=self._usage())
            return
        if mode == "text_stop":
            yield from self._text_events(model, "fixture text", usage=self._usage())
            return
        if mode == "text_reasoning":
            yield from self._text_events(model, "fixture text", usage=self._usage(), reasoning="think-more")
            return
        if mode == "text_zero_usage":
            yield from self._text_events(model, "zero usage", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
            return
        if mode == "text_bad_usage":
            yield from self._text_events(model, "bad usage", usage={"prompt_tokens": 3})
            return
        if mode == "text_conflicting_finish":
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": "x"}, "finish_reason": "stop"}])
            yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "length"}])
            yield self._chunk(model, [], self._usage())
            yield b"data: [DONE]\n\n"
            return
        if mode == "text_unsupported_finish":
            yield from self._text_events(model, "filtered", finish="content_filter", usage=self._usage())
            return
        if mode == "native_read":
            arguments = json.dumps({"path": self.marker})
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "content": "reading"}, "finish_reason": None}])
            yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_read_1", "type": "function", "function": {"name": "read", "arguments": arguments}}]}, "finish_reason": None}])
            yield self._chunk(model, [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}])
            yield self._chunk(model, [], self._usage())
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call":
            yield from self._tool_call_events(model)
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call_finish_with_arguments":
            # A chunk may carry finish_reason and the final argument fragment
            # together (observed from hosted big-pickle). The finish marker is
            # not a truncation signal; assembly continues to [DONE].
            arguments = json.dumps({"path": self.marker})
            midpoint = max(1, len(arguments) // 2)
            yield self._chunk(model, [{"index": 0, "delta": {"role": "assistant", "tool_calls": [
                {"index": 0, "id": "call_after", "type": "function",
                 "function": {"name": "read", "arguments": arguments[:midpoint]}}]}, "finish_reason": None}])
            yield self._chunk(model, [{"index": 0, "delta": {"tool_calls": [
                {"index": 0, "id": "call_after", "type": "function",
                 "function": {"name": "read", "arguments": arguments[midpoint:]}}]}, "finish_reason": "tool_calls"}])
            yield self._chunk(model, [], self._usage())
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call_no_usage":
            yield from self._tool_call_events(model, usage=False)
            yield b"data: [DONE]\n\n"
            return
        if mode == "text_length":
            yield from self._text_events(model, "truncated text", finish="length", usage=self._usage())
            return
        if mode == "text_no_usage":
            yield from self._text_events(model, "no usage")
            return
        if mode == "tool_call_no_done":
            yield from self._tool_call_events(model)
            return
        if mode == "tool_call_error":
            yield from self._tool_call_events(model)
            yield b'data: {"error": {"message": "upstream exploded", "code": 500}}\n\n'
            yield b"data: [DONE]\n\n"
            return
        if mode == "tool_call_invalid":
            yield from self._tool_call_events(model, call_id="")
            yield b"data: [DONE]\n\n"
            return
        if mode == "execute":
            code = (
                f"const r = await tools.write_file({{path: {json.dumps(self.marker)}, content: 'js-nested'}})\n"
                'return "JS_DONE:" + String(r)'
            )
            yield from self._execute_events(model, code)
            return
        if mode == "execute_infinite":
            yield from self._execute_events(model, "while (true) {}")
            return
        raise AssertionError(f"unknown fixture mode: {mode}")
