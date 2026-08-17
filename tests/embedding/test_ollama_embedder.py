"""Unit tests for :class:`datasheet_rag.embedding.OllamaEmbedder` and the
``get_embedder`` backend factory. No network: ``httpx.post`` is monkeypatched.
"""

from __future__ import annotations

from typing import Any

import pytest

from datasheet_rag.config import get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _patch_embeddings(monkeypatch, vector_fn) -> dict[str, Any]:
    import httpx

    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], timeout: float) -> _FakeResponse:  # noqa: A002
        captured["url"] = url
        captured["json"] = json
        # New contract: /api/embed takes "input" and returns "embeddings" (list).
        return _FakeResponse({"embeddings": [vector_fn(json["input"])]})

    monkeypatch.setattr(httpx, "post", _fake_post)
    return captured


def test_embed_one_happy_path(monkeypatch):
    from datasheet_rag.embedding import OllamaEmbedder

    _patch_embeddings(monkeypatch, lambda _t: [0.1] * 8)
    emb = OllamaEmbedder(model="bge-m3", dimensions=8, host="http://x:1")
    vec = emb.embed_one("hello")
    assert vec == [0.1] * 8
    assert emb.stats()["total_invocations"] == 1


def test_embed_texts_preserves_order(monkeypatch):
    from datasheet_rag.embedding import OllamaEmbedder

    _patch_embeddings(monkeypatch, lambda t: [float(len(t))] * 4)
    emb = OllamaEmbedder(dimensions=4)
    out = emb.embed_texts(["a", "bbb", "cc"])
    assert out == [[1.0] * 4, [3.0] * 4, [2.0] * 4]


def test_dimension_mismatch_raises(monkeypatch):
    from datasheet_rag.embedding import OllamaEmbedder

    _patch_embeddings(monkeypatch, lambda _t: [0.0] * 768)
    emb = OllamaEmbedder(dimensions=1024)
    with pytest.raises(ValueError, match="768-dim"):
        emb.embed_one("hello")
    assert emb.stats()["total_errors"] == 1


def test_get_embedder_local_ollama(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "local")
    monkeypatch.setenv("RAG_LOCAL_EMBEDDING_RUNTIME", "ollama")
    get_settings.cache_clear()
    from datasheet_rag.embedding import OllamaEmbedder, get_embedder

    assert isinstance(get_embedder(), OllamaEmbedder)


def test_get_embedder_local_sentence_transformers_default(monkeypatch):
    # backend=local with no runtime override -> sentence-transformers (default).
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "local")
    monkeypatch.delenv("RAG_LOCAL_EMBEDDING_RUNTIME", raising=False)
    get_settings.cache_clear()
    from datasheet_rag.embedding import SentenceTransformerEmbedder, get_embedder

    # Construction is lazy (no model load / no torch import), so this is cheap.
    assert isinstance(get_embedder(), SentenceTransformerEmbedder)


def test_get_embedder_bedrock(monkeypatch):
    monkeypatch.setenv("RAG_EMBEDDING_BACKEND", "bedrock")
    get_settings.cache_clear()
    from datasheet_rag.embedding import BedrockEmbedder, get_embedder

    # verbose is accepted by both backends; ensure kwarg filtering keeps it.
    assert isinstance(get_embedder(verbose=True), BedrockEmbedder)
