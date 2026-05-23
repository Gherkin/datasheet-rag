"""Unit tests for :mod:`aws_rag.embedding.embedder`.

These tests never touch the network: every test injects a ``MagicMock``
in place of the real ``bedrock-runtime`` client. The mock's
``invoke_model`` is wired with a ``side_effect`` so we can both inspect
the request payload (``inputText``, ``dimensions``, ``normalize``) and
return a deterministic embedding per call.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from aws_rag.config import get_settings
from aws_rag.embedding import BedrockEmbedder, embed_chunk_graph, embed_texts
from aws_rag.models.chunk import (
    Chunk,
    ChunkGraph,
    ChunkLevel,
    ChunkMetadata,
    LayoutType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_body(payload: dict[str, Any]) -> MagicMock:
    """Build a fake ``response['body']`` whose ``.read()`` returns JSON bytes."""
    body = MagicMock()
    body.read.return_value = json.dumps(payload).encode("utf-8")
    return body


def _build_invoke_side_effect(
    vector_fn: Callable[[str], list[float]],
    token_fn: Callable[[str], int] | None = None,
) -> Callable[..., dict[str, Any]]:
    """Return a side_effect that emits per-text deterministic vectors.

    ``vector_fn`` is called with the ``inputText`` field of every request
    and its return is plugged into the response under ``"embedding"``.
    """

    def _side_effect(*, modelId: str, body: str, **_: Any) -> dict[str, Any]:
        payload_in = json.loads(body)
        text = payload_in["inputText"]
        vec = vector_fn(text)
        tokens = token_fn(text) if token_fn else max(len(text.split()), 1)
        return {"body": _make_body({"embedding": vec, "inputTextTokenCount": tokens})}

    return _side_effect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_bedrock_client() -> MagicMock:
    """A MagicMock standing in for a ``bedrock-runtime`` client.

    Default behavior: each invocation returns a 4-dim vector where
    ``len(inputText)`` is broadcast into every slot. That way the test
    can verify input → output ordering without hand-rolled mappings.
    """
    client = MagicMock()
    client.invoke_model.side_effect = _build_invoke_side_effect(
        lambda text: [float(len(text))] * 4
    )
    return client


@pytest.fixture()
def embedder(mock_bedrock_client: MagicMock) -> BedrockEmbedder:
    """A BedrockEmbedder wired to the mock client, Titan-defaults."""
    return BedrockEmbedder(
        client=mock_bedrock_client,
        dimensions=1024,
        normalize=True,
        max_concurrency=4,
    )


# ---------------------------------------------------------------------------
# 1. Defaults
# ---------------------------------------------------------------------------


def test_constructs_with_settings_defaults(mock_bedrock_client: MagicMock) -> None:
    settings = get_settings()
    e = BedrockEmbedder(client=mock_bedrock_client)

    assert e.model_id == settings.embedding_model_id
    assert e.dimensions == settings.embedding_dimensions
    assert e.normalize == settings.embedding_normalize
    assert e.max_concurrency == settings.embedding_batch_size
    assert e.stats() == {
        "total_tokens_in": 0,
        "total_invocations": 0,
        "total_errors": 0,
    }


# ---------------------------------------------------------------------------
# 2. Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_dim", [0, 1, 128, 768, 2048])
def test_invalid_titan_dimensions_raises(
    mock_bedrock_client: MagicMock, bad_dim: int
) -> None:
    with pytest.raises(ValueError, match="Titan v2 dimensions"):
        BedrockEmbedder(client=mock_bedrock_client, dimensions=bad_dim)


def test_non_titan_model_skips_dimension_validation(
    mock_bedrock_client: MagicMock,
) -> None:
    # 768 would normally be rejected for Titan, but a custom model_id
    # should bypass the check entirely (with a warning, not an error).
    e = BedrockEmbedder(
        client=mock_bedrock_client,
        model_id="cohere.embed-english-v3",
        dimensions=768,
    )
    assert e.dimensions == 768
    assert e._is_titan is False


def test_invalid_concurrency_raises(mock_bedrock_client: MagicMock) -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        BedrockEmbedder(client=mock_bedrock_client, max_concurrency=0)


# ---------------------------------------------------------------------------
# 3. embed_one — request body + response parsing
# ---------------------------------------------------------------------------


def test_embed_one_builds_correct_request_body(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    vec = embedder.embed_one("hello world")

    assert mock_bedrock_client.invoke_model.call_count == 1
    call = mock_bedrock_client.invoke_model.call_args
    assert call.kwargs["modelId"] == embedder.model_id

    body = json.loads(call.kwargs["body"])
    assert body == {
        "inputText": "hello world",
        "dimensions": 1024,
        "normalize": True,
    }

    # vector_fn returned [len("hello world")] * 4 == [11.0, 11.0, 11.0, 11.0]
    assert vec == [11.0, 11.0, 11.0, 11.0]


def test_embed_one_omits_titan_params_for_non_titan(
    mock_bedrock_client: MagicMock,
) -> None:
    e = BedrockEmbedder(
        client=mock_bedrock_client,
        model_id="cohere.embed-english-v3",
        dimensions=1024,  # ignored (not Titan)
    )
    e.embed_one("hi")
    body = json.loads(mock_bedrock_client.invoke_model.call_args.kwargs["body"])
    assert body == {"inputText": "hi"}
    assert "dimensions" not in body
    assert "normalize" not in body


# ---------------------------------------------------------------------------
# 4. embed_one empty input
# ---------------------------------------------------------------------------


def test_embed_one_empty_string_raises(embedder: BedrockEmbedder) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed_one("")


# ---------------------------------------------------------------------------
# 5. embed_texts preserves order
# ---------------------------------------------------------------------------


def test_embed_texts_preserves_input_order(
    mock_bedrock_client: MagicMock,
) -> None:
    # vector value == len(text), so order is verifiable
    mock_bedrock_client.invoke_model.side_effect = _build_invoke_side_effect(
        lambda text: [float(len(text))]
    )
    e = BedrockEmbedder(client=mock_bedrock_client, max_concurrency=8)

    texts = ["a", "bbbb", "ccccccc", "dd"]
    vectors = e.embed_texts(texts)

    assert [v[0] for v in vectors] == [1.0, 4.0, 7.0, 2.0]
    assert mock_bedrock_client.invoke_model.call_count == 4


# ---------------------------------------------------------------------------
# 6. embed_texts on empty input
# ---------------------------------------------------------------------------


def test_embed_texts_empty_returns_empty_without_invoking(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    out = embedder.embed_texts([])
    assert out == []
    mock_bedrock_client.invoke_model.assert_not_called()


def test_embed_texts_with_empty_string_inside_raises(
    embedder: BedrockEmbedder,
) -> None:
    with pytest.raises(ValueError, match="index 1"):
        embedder.embed_texts(["ok", "", "also ok"])


# ---------------------------------------------------------------------------
# 7. embed_chunks — context_text / fallback / skip
# ---------------------------------------------------------------------------


def _chunk(
    chunk_id: str,
    *,
    text: str = "",
    context_text: str = "",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="doc-1",
        level=ChunkLevel.MICRO,
        text=text,
        context_text=context_text,
        token_count=len(text.split()),
        metadata=ChunkMetadata(doc_id="doc-1", layout_type=LayoutType.TEXT),
    )


def test_embed_chunks_uses_context_text(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    chunks = [
        _chunk("c1", text="raw1", context_text="CONTEXT one"),
        _chunk("c2", text="raw2", context_text="CONTEXT two longer"),
    ]
    result = embedder.embed_chunks(chunks)

    assert list(result.keys()) == ["c1", "c2"]

    sent = [
        json.loads(call.kwargs["body"])["inputText"]
        for call in mock_bedrock_client.invoke_model.call_args_list
    ]
    assert "CONTEXT one" in sent
    assert "CONTEXT two longer" in sent
    # raw text must NOT have been sent (context_text was populated)
    assert "raw1" not in sent
    assert "raw2" not in sent


def test_embed_chunks_falls_back_to_text_when_context_empty(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    chunks = [
        _chunk("c1", text="raw-fallback", context_text=""),
        _chunk("c2", text="raw2", context_text="CONTEXT two"),
    ]
    result = embedder.embed_chunks(chunks)

    assert set(result.keys()) == {"c1", "c2"}
    sent = [
        json.loads(call.kwargs["body"])["inputText"]
        for call in mock_bedrock_client.invoke_model.call_args_list
    ]
    assert "raw-fallback" in sent
    assert "CONTEXT two" in sent


def test_embed_chunks_skips_both_empty(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    chunks = [
        _chunk("c1", text="raw1", context_text="CTX 1"),
        _chunk("c2", text="", context_text=""),     # skip
        _chunk("c3", text="raw3", context_text="CTX 3"),
    ]
    result = embedder.embed_chunks(chunks)

    assert set(result.keys()) == {"c1", "c3"}
    assert "c2" not in result
    assert mock_bedrock_client.invoke_model.call_count == 2


def test_embed_chunks_empty_iterable_returns_empty_dict(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    assert embedder.embed_chunks([]) == {}
    mock_bedrock_client.invoke_model.assert_not_called()


# ---------------------------------------------------------------------------
# 8. embed_chunk_graph end-to-end
# ---------------------------------------------------------------------------


def test_embed_chunk_graph_round_trip(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    graph = ChunkGraph(doc_id="doc-1")
    graph.add(_chunk("c1", text="t1", context_text="CTX 1"))
    graph.add(_chunk("c2", text="t2", context_text="CTX two"))
    graph.add(_chunk("c3", text="t3", context_text="CTX three!"))

    result = embed_chunk_graph(graph, embedder=embedder)

    assert set(result.keys()) == {"c1", "c2", "c3"}
    # Every value is a float list of the right length (4, from the mock).
    for vec in result.values():
        assert isinstance(vec, list)
        assert len(vec) == 4
        assert all(isinstance(x, float) for x in vec)


# ---------------------------------------------------------------------------
# 9. Throttling / retry behavior
# ---------------------------------------------------------------------------


def test_throttling_recovers_via_side_effect_sequence(
    mock_bedrock_client: MagicMock,
) -> None:
    """We can't easily test botocore's adaptive retry from the outside,
    but we can verify that if the first call surfaces a Throttling error
    and the second succeeds, the embedder treats it as a hard error on
    the surface and bumps ``total_errors`` — confirming the metric path.

    (Real throttling retries happen *inside* botocore before we ever see
    a response, so they're invisible at this layer.)
    """
    throttle_error = ClientError(
        error_response={
            "Error": {
                "Code": "ThrottlingException",
                "Message": "Rate exceeded",
            }
        },
        operation_name="InvokeModel",
    )

    # First call raises, second would succeed — but botocore's retry
    # layer is bypassed (we're talking to a MagicMock), so the first
    # raise propagates and we see it as a hard error.
    success_response = {
        "body": _make_body({"embedding": [0.1, 0.2, 0.3], "inputTextTokenCount": 3}),
    }
    mock_bedrock_client.invoke_model.side_effect = [throttle_error, success_response]

    e = BedrockEmbedder(client=mock_bedrock_client, max_concurrency=1)

    with pytest.raises(ClientError):
        e.embed_one("first")
    assert e.stats()["total_errors"] == 1
    assert e.stats()["total_invocations"] == 0

    # Second call succeeds, metrics advance.
    vec = e.embed_one("second")
    assert vec == [0.1, 0.2, 0.3]
    assert e.stats()["total_invocations"] == 1
    assert e.stats()["total_tokens_in"] == 3
    assert e.stats()["total_errors"] == 1  # unchanged after success


# ---------------------------------------------------------------------------
# 10. stats()
# ---------------------------------------------------------------------------


def test_stats_tracks_invocations_and_tokens(
    mock_bedrock_client: MagicMock,
) -> None:
    mock_bedrock_client.invoke_model.side_effect = _build_invoke_side_effect(
        lambda text: [float(len(text))] * 2,
        token_fn=lambda text: len(text),  # token == char count
    )
    e = BedrockEmbedder(client=mock_bedrock_client, max_concurrency=4)

    e.embed_texts(["aa", "bbb", "cccc"])  # tokens: 2 + 3 + 4 = 9
    stats = e.stats()
    assert stats["total_invocations"] == 3
    assert stats["total_tokens_in"] == 9
    assert stats["total_errors"] == 0


# ---------------------------------------------------------------------------
# Module-level helpers (embed_texts module-level)
# ---------------------------------------------------------------------------


def test_module_embed_texts_uses_provided_embedder(
    mock_bedrock_client: MagicMock, embedder: BedrockEmbedder
) -> None:
    out = embed_texts(["one", "two"], embedder=embedder)
    assert len(out) == 2
    assert mock_bedrock_client.invoke_model.call_count == 2
