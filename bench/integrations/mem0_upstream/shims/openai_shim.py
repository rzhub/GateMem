"""Compatibility shim for the `openai` Python SDK used by vendored Mem0 upstream.

We avoid placing a top-level `openai/` package in the repo (which would shadow a
user's installed OpenAI SDK). Instead, we inject this module into `sys.modules`
*only when* the Mem0 upstream backend is used.

Implements the minimal subset Mem0 uses:
- OpenAI(...).chat.completions.create(...)
- OpenAI(...).embeddings.create(...)

The shim talks to an OpenAI-compatible REST API via HTTPS.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import requests


@dataclass
class _ToolFunction:
    name: str
    arguments: str


@dataclass
class _ToolCall:
    function: _ToolFunction


@dataclass
class _Message:
    content: Optional[str] = None
    tool_calls: Optional[List[_ToolCall]] = None


@dataclass
class _Choice:
    message: _Message


class _Embeddings:
    def __init__(self, client: "OpenAI") -> None:
        self._client = client

    def create(self, *, input: List[str], model: str, dimensions: Optional[int] = None, **kwargs):
        url = f"{self._client.base_url}/embeddings"
        payload: Dict[str, Any] = {"model": model, "input": input}
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        payload.update({k: v for k, v in kwargs.items() if v is not None})

        data = self._client._post(url, payload)
        items = []
        for d in data.get("data", []) or []:
            items.append(SimpleNamespace(embedding=d.get("embedding")))
        return SimpleNamespace(data=items)


class _ChatCompletions:
    def __init__(self, client: "OpenAI") -> None:
        self._client = client

    def create(self, **params):
        url = f"{self._client.base_url}/chat/completions"
        data = self._client._post(url, params)

        choices_out: List[_Choice] = []
        for c in data.get("choices", []) or []:
            msg = c.get("message", {}) or {}
            content = msg.get("content")
            tool_calls_raw = msg.get("tool_calls")
            tool_calls: Optional[List[_ToolCall]] = None
            if isinstance(tool_calls_raw, list):
                tool_calls = []
                for tc in tool_calls_raw:
                    fn = (tc.get("function") or {}) if isinstance(tc, dict) else {}
                    name = fn.get("name")
                    args = fn.get("arguments")
                    if isinstance(name, str) and isinstance(args, str):
                        tool_calls.append(_ToolCall(function=_ToolFunction(name=name, arguments=args)))
            choices_out.append(_Choice(message=_Message(content=content, tool_calls=tool_calls)))

        return SimpleNamespace(choices=choices_out)


class _Chat:
    def __init__(self, client: "OpenAI") -> None:
        self.completions = _ChatCompletions(client)


class OpenAI:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.openai.com/v1", **kwargs):
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        self.chat = _Chat(self)
        self.embeddings = _Embeddings(self)

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
        if resp.status_code < 200 or resp.status_code >= 300:
            raise RuntimeError(f"OpenAI API error {resp.status_code}: {resp.text[:500]}")
        return resp.json()


__all__ = ["OpenAI"]
