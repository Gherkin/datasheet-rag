"""Unit tests for the local (Ollama) chat/vision backend.

No network: every test monkeypatches ``httpx.post``. Settings are driven via
``RAG_*`` env vars with the ``get_settings`` cache cleared around each test.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import pytest

from datasheet_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _patch_httpx(monkeypatch, payload: dict[str, Any]) -> dict[str, Any]:
    """Patch ``httpx.post`` to record the request and return ``payload``."""
    import httpx

    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(payload)

    monkeypatch.setattr(httpx, "post", _fake_post)
    return captured


# ---------------------------------------------------------------------------
# Message translation + content-based model routing
# ---------------------------------------------------------------------------


def test_text_only_request_uses_chat_model(monkeypatch):
    from datasheet_rag.local_models import OllamaInvokeClient

    captured = _patch_httpx(monkeypatch, {"message": {"content": "hello"}})
    client = OllamaInvokeClient(chat_model="text-model", vision_model="vision-model")

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "temperature": 0.0,
            "system": "be terse",
            "messages": [{"role": "user", "content": "what is this?"}],
        }
    )
    resp = client.invoke_model(modelId="arn:ignored", body=body)

    req = captured["json"]
    assert req["model"] == "text-model"
    assert req["stream"] is False
    assert req["options"] == {"num_predict": 128, "temperature": 0.0}
    assert req["messages"][0] == {"role": "system", "content": "be terse"}
    assert req["messages"][1] == {"role": "user", "content": "what is this?"}

    payload = json.loads(resp["body"].read())
    assert payload["content"][0]["text"] == "hello"


def test_image_request_routes_to_vision_model_and_extracts_images(monkeypatch):
    from datasheet_rag.local_models import OllamaInvokeClient

    captured = _patch_httpx(
        monkeypatch,
        {"message": {"content": "a diagram"}, "prompt_eval_count": 7, "eval_count": 11},
    )
    client = OllamaInvokeClient(chat_model="text-model", vision_model="vision-model")

    b64 = base64.b64encode(b"\x89PNG fake").decode("ascii")
    body = json.dumps(
        {
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/png", "data": b64},
                        },
                    ],
                }
            ],
        }
    )
    resp = client.invoke_model(modelId="arn:ignored", body=body)

    req = captured["json"]
    assert req["model"] == "vision-model"
    user_msg = req["messages"][0]
    assert user_msg["content"] == "describe"
    assert user_msg["images"] == [b64]
    # temperature omitted from body -> not forced into options
    assert "temperature" not in req["options"]

    payload = json.loads(resp["body"].read())
    assert payload["content"][0]["text"] == "a diagram"
    assert payload["usage"] == {"input_tokens": 7, "output_tokens": 11}


# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------


def test_get_chat_client_local_ollama(monkeypatch):
    monkeypatch.setenv("RAG_VISION_BACKEND", "local")
    monkeypatch.setenv("RAG_LOCAL_VISION_RUNTIME", "ollama")
    get_settings.cache_clear()
    from datasheet_rag.local_models import OllamaInvokeClient, get_chat_client

    client = get_chat_client(kind="vision")
    assert isinstance(client, OllamaInvokeClient)
    assert client.model == get_settings().local_vision_model


def test_get_chat_client_local_huggingface(monkeypatch):
    monkeypatch.setenv("RAG_TEXT_BACKEND", "local")
    monkeypatch.setenv("RAG_LOCAL_TEXT_RUNTIME", "huggingface")
    get_settings.cache_clear()
    from datasheet_rag.local_models import TransformersChatClient, get_chat_client

    # Construction is lazy (no model load / no torch import) -> cheap to assert.
    client = get_chat_client(kind="text")
    assert isinstance(client, TransformersChatClient)
    assert client.is_vision is False


def test_get_chat_client_bedrock(monkeypatch):
    monkeypatch.setenv("RAG_VISION_BACKEND", "bedrock")
    get_settings.cache_clear()
    from datasheet_rag.local_models import (
        OllamaInvokeClient,
        TransformersChatClient,
        get_chat_client,
    )

    client = get_chat_client(kind="vision")
    # Real bedrock-runtime client, not a local shim.
    assert not isinstance(client, (OllamaInvokeClient, TransformersChatClient))
    assert hasattr(client, "invoke_model")
