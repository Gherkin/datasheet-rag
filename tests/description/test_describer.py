"""Tests for FigureDescriber (vision-LLM figure description).

Bedrock is fully mocked — no network. Tests verify:
* the API request body is correctly shaped (image base64, media_type,
  system prompt, surrounding-text inclusion)
* describe_chunk_in_context loads bytes + neighbours from the store
* describe_chunks runs concurrently and tolerates per-chunk failures
* describe_figures_in_store skips chunks that already have a description
  and writes back via update_figure_description
* stats() counts invocations + tokens + errors
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from datasheet_rag.description import FigureDescriber, describe_figures_in_store
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.store import connect, insert_chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mock_bedrock_response(
    text: str,
    *,
    input_tokens: int = 1200,
    output_tokens: int = 80,
) -> Any:
    body = json.dumps({
        "content": [{"type": "text", "text": text}],
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }).encode()
    resp = MagicMock()
    resp.__getitem__.side_effect = lambda k: {"body": MagicMock(read=lambda: body)}[k]
    return resp


def _figure_chunk(
    chunk_id: str,
    *,
    image_path: str,
    description: str | None = None,
    caption: str = "Figure 3-2: SPI4 timing",
    section: str = "SPI Timing",
    prev_id: str | None = None,
    next_id: str | None = None,
) -> Chunk:
    md = ChunkMetadata(
        doc_id="docA",
        chapter_title="Comm",
        section_title=section,
        page_numbers=[42],
        layout_type=LayoutType.FIGURE,
    )
    return Chunk(
        id=chunk_id,
        doc_id="docA",
        level=ChunkLevel.MICRO,
        text="[Figure]",
        context_text="SPI > Timing > [Figure] " + caption,
        token_count=5,
        metadata=md,
        figure_image_path=image_path,
        figure_caption=caption,
        figure_description=description,
        prev_id=prev_id,
        next_id=next_id,
    )


def _text_chunk(chunk_id: str, text: str) -> Chunk:
    md = ChunkMetadata(
        doc_id="docA",
        chapter_title="Comm",
        section_title="SPI Timing",
        page_numbers=[42],
        layout_type=LayoutType.TEXT,
    )
    return Chunk(
        id=chunk_id, doc_id="docA", level=ChunkLevel.MICRO,
        text=text, context_text=text, token_count=10, metadata=md,
    )


@pytest.fixture
def conn() -> Any:
    return connect(":memory:", embedding_dim=8)


@pytest.fixture
def fake_client() -> Any:
    """Bedrock client whose invoke_model returns a stub description response."""
    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response(
        "Block diagram of the SPI4 controller showing TX FIFO, RX FIFO, "
        "shift register and NSS gating from the APB2 bus."
    )
    return client


# ---------------------------------------------------------------------------
# Single-call shape
# ---------------------------------------------------------------------------


def test_describe_one_builds_correct_request(fake_client: Any) -> None:
    describer = FigureDescriber(client=fake_client, max_concurrency=1)
    out = describer.describe_one(
        image_bytes=b"PNGBYTES",
        image_format="png",
        caption="Figure 3-2: SPI4 timing",
        section_context="Comm > SPI Timing",
        surrounding_text="The SPI4 peripheral supports four modes…",
    )
    assert "SPI4" in out

    # Inspect the request body we sent to Bedrock.
    args, kwargs = fake_client.invoke_model.call_args
    body = json.loads(kwargs["body"])
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 400
    assert "describe technical figures" in body["system"].lower()

    user_blocks = body["messages"][0]["content"]
    # First block: image with correct media_type and base64
    image_block = next(b for b in user_blocks if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/png"
    assert base64.b64decode(image_block["source"]["data"]) == b"PNGBYTES"
    # Second block: text containing caption + surrounding
    text_block = next(b for b in user_blocks if b["type"] == "text")
    assert "Caption: Figure 3-2: SPI4 timing" in text_block["text"]
    assert "Surrounding text:" in text_block["text"]
    assert "Section context: Comm > SPI Timing" in text_block["text"]


def test_describe_one_media_type_inference(fake_client: Any) -> None:
    describer = FigureDescriber(client=fake_client)
    describer.describe_one(image_bytes=b"x", image_format="jpg")
    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    image_block = next(b for b in body["messages"][0]["content"] if b["type"] == "image")
    assert image_block["source"]["media_type"] == "image/jpeg"


def test_describe_one_strips_text_blocks_correctly() -> None:
    """If Bedrock returns multiple text blocks they're joined; non-text blocks ignored."""
    client = MagicMock()
    body = json.dumps({
        "content": [
            {"type": "text", "text": "First sentence."},
            {"type": "text", "text": "Second sentence."},
            {"type": "image", "source": {}},  # should be ignored
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode()
    client.invoke_model.return_value = {"body": MagicMock(read=lambda: body)}
    describer = FigureDescriber(client=client)
    out = describer.describe_one(image_bytes=b"x", image_format="png")
    assert out == "First sentence.\nSecond sentence."


# ---------------------------------------------------------------------------
# describe_chunk_in_context
# ---------------------------------------------------------------------------


def test_describe_chunk_in_context_pulls_neighbors_from_store(
    conn: Any, fake_client: Any, tmp_path: Any
) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    prev_chunk = _text_chunk("c-prev", "The preceding paragraph mentions SPI4 modes.")
    fig = _figure_chunk("c-fig", image_path=str(img),
                        prev_id="c-prev", next_id="c-next")
    next_chunk = _text_chunk("c-next", "The next paragraph discusses CR1 register layout.")
    insert_chunks(conn, [prev_chunk, fig, next_chunk])

    describer = FigureDescriber(client=fake_client)
    desc = describer.describe_chunk_in_context(fig, conn)
    assert "SPI4" in desc

    body = json.loads(fake_client.invoke_model.call_args.kwargs["body"])
    text = next(b["text"] for b in body["messages"][0]["content"] if b["type"] == "text")
    assert "preceding paragraph" in text
    assert "CR1 register layout" in text
    assert "Comm > SPI Timing" in text


def test_describe_chunk_in_context_rejects_non_figure(
    conn: Any, fake_client: Any
) -> None:
    plain = _text_chunk("c-plain", "Body text only.")
    insert_chunks(conn, [plain])
    describer = FigureDescriber(client=fake_client)
    with pytest.raises(ValueError, match="not a figure"):
        describer.describe_chunk_in_context(plain, conn)


def test_describe_chunk_in_context_missing_local_file_raises(
    conn: Any, fake_client: Any, tmp_path: Any
) -> None:
    fig = _figure_chunk("c-gone", image_path=str(tmp_path / "missing.png"))
    insert_chunks(conn, [fig])
    describer = FigureDescriber(client=fake_client)
    with pytest.raises(FileNotFoundError):
        describer.describe_chunk_in_context(fig, conn)


# ---------------------------------------------------------------------------
# describe_chunks (concurrent + failure-tolerant)
# ---------------------------------------------------------------------------


def test_describe_chunks_tolerates_per_chunk_failures(
    conn: Any, tmp_path: Any
) -> None:
    good_img = tmp_path / "good.png"
    good_img.write_bytes(b"\x89PNGGOOD")
    bad_img = tmp_path / "missing.png"  # never created
    good = _figure_chunk("c-good", image_path=str(good_img))
    bad = _figure_chunk("c-bad", image_path=str(bad_img))
    insert_chunks(conn, [good, bad])

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("good description")
    describer = FigureDescriber(client=client, max_concurrency=2)

    out = describer.describe_chunks([good, bad], conn)
    assert set(out.keys()) == {"c-good"}
    assert out["c-good"] == "good description"


def test_describe_chunks_retries_transient_failure(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr("datasheet_rag.description.describer._DESCRIBE_RETRY_WAIT", 0)
    img = tmp_path / "flaky.png"
    img.write_bytes(b"\x89PNGFLAKY")
    fig = _figure_chunk("c-flaky", image_path=str(img))
    insert_chunks(conn, [fig])

    client = MagicMock()
    # Two transient failures, then success → should retry and recover.
    client.invoke_model.side_effect = [
        RuntimeError("read timeout"),
        RuntimeError("read timeout"),
        _mock_bedrock_response("recovered description"),
    ]
    describer = FigureDescriber(client=client, max_concurrency=1)
    out = describer.describe_chunks([fig], conn)

    assert out == {"c-flaky": "recovered description"}
    assert client.invoke_model.call_count == 3


def test_describe_chunks_gives_up_after_max_attempts(
    conn: Any, tmp_path: Any, monkeypatch: Any
) -> None:
    monkeypatch.setattr("datasheet_rag.description.describer._DESCRIBE_RETRY_WAIT", 0)
    from datasheet_rag.description import describer as _d

    img = tmp_path / "dead.png"
    img.write_bytes(b"\x89PNGDEAD")
    fig = _figure_chunk("c-dead", image_path=str(img))
    insert_chunks(conn, [fig])

    client = MagicMock()
    client.invoke_model.side_effect = RuntimeError("always fails")
    describer = FigureDescriber(client=client, max_concurrency=1)
    out = describer.describe_chunks([fig], conn)

    assert out == {}  # skipped after exhausting retries
    assert client.invoke_model.call_count == _d._DESCRIBE_MAX_ATTEMPTS


def test_describe_chunks_ignores_non_figure_inputs(
    conn: Any, fake_client: Any, tmp_path: Any
) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    fig = _figure_chunk("c-fig", image_path=str(img))
    text = _text_chunk("c-text", "body")
    insert_chunks(conn, [fig, text])
    describer = FigureDescriber(client=fake_client)
    out = describer.describe_chunks([fig, text], conn)
    assert set(out.keys()) == {"c-fig"}


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_count_tokens_and_invocations(fake_client: Any) -> None:
    describer = FigureDescriber(client=fake_client)
    describer.describe_one(image_bytes=b"x", image_format="png")
    describer.describe_one(image_bytes=b"y", image_format="png")
    s = describer.stats()
    assert s["total_invocations"] == 2
    assert s["total_input_tokens"] == 2400  # 1200 per mocked call
    assert s["total_output_tokens"] == 160  # 80 per mocked call
    assert s["total_errors"] == 0


def test_stats_count_errors() -> None:
    client = MagicMock()
    client.invoke_model.side_effect = RuntimeError("boom")
    describer = FigureDescriber(client=client)
    with pytest.raises(RuntimeError):
        describer.describe_one(image_bytes=b"x", image_format="png")
    s = describer.stats()
    assert s["total_errors"] == 1
    assert s["total_invocations"] == 0  # error before counter bump


# ---------------------------------------------------------------------------
# describe_figures_in_store (orchestrator)
# ---------------------------------------------------------------------------


def test_describe_figures_in_store_writes_and_skips_missing_only(
    conn: Any, tmp_path: Any
) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    other = tmp_path / "fig2.png"
    other.write_bytes(b"\x89PNGFAKE2")

    already_described = _figure_chunk(
        "c-done", image_path=str(img),
        description="existing description — do not overwrite",
    )
    # A different image: chunks that share one are deliberately described
    # once and copied (see the dedupe test below).
    pending = _figure_chunk("c-todo", image_path=str(other))
    insert_chunks(conn, [already_described, pending])

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("new description")
    describer = FigureDescriber(client=client, max_concurrency=1)

    out = describe_figures_in_store(
        conn, missing_only=True, describer=describer,
    )
    assert set(out.keys()) == {"c-todo"}

    # The already-described chunk must be untouched.
    from datasheet_rag.store import get_chunk
    intact = get_chunk(conn, "c-done")
    assert intact is not None
    assert intact.figure_description == "existing description — do not overwrite"

    # The pending one got the new description and it's folded into context_text.
    updated = get_chunk(conn, "c-todo")
    assert updated is not None
    assert updated.figure_description == "new description"
    assert "new description" in updated.context_text


def test_describe_figures_in_store_dry_run_does_not_persist(
    conn: Any, tmp_path: Any
) -> None:
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    chunk = _figure_chunk("c-dry", image_path=str(img))
    insert_chunks(conn, [chunk])

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("hypothetical desc")
    describer = FigureDescriber(client=client)
    out = describe_figures_in_store(
        conn, describer=describer, dry_run=True,
    )
    assert out == {"c-dry": "hypothetical desc"}

    from datasheet_rag.store import get_chunk
    after = get_chunk(conn, "c-dry")
    assert after is not None
    assert after.figure_description is None  # not persisted


def test_describe_figures_in_store_limit_truncates(
    conn: Any, tmp_path: Any
) -> None:
    # One file per figure: chunks sharing an image are described once, so a
    # limit only bites when the images are genuinely distinct.
    chunks = []
    for i in range(5):
        img = tmp_path / f"fig{i}.png"
        img.write_bytes(b"\x89PNGFAKE" + str(i).encode())
        chunks.append(_figure_chunk(f"c-{i}", image_path=str(img)))
    insert_chunks(conn, chunks)

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("desc")
    describer = FigureDescriber(client=client, max_concurrency=1)
    out = describe_figures_in_store(conn, limit=2, describer=describer)
    assert len(out) == 2


def test_describe_figures_in_store_describes_shared_image_once(
    conn: Any, tmp_path: Any
) -> None:
    """A MESO wrapping a single figure carries its MICRO's image (GH #41).

    Both rows must end up described, but the vision model is only worth
    paying once for identical pixels.
    """
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    micro = _figure_chunk("c-micro", image_path=str(img))
    meso = _figure_chunk("c-meso", image_path=str(img))
    meso.level = ChunkLevel.MESO
    insert_chunks(conn, [micro, meso])

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("shared description")
    describer = FigureDescriber(client=client, max_concurrency=1)

    out = describe_figures_in_store(conn, describer=describer)

    assert out == {"c-micro": "shared description", "c-meso": "shared description"}
    assert client.invoke_model.call_count == 1

    from datasheet_rag.store import get_chunk

    for cid in ("c-micro", "c-meso"):
        stored = get_chunk(conn, cid)
        assert stored is not None
        assert stored.figure_description == "shared description"


def test_describe_figures_in_store_all_regenerates_a_shared_description(
    conn: Any, tmp_path: Any
) -> None:
    """`--all` asks for a fresh description, not a copy of the stale one."""
    img = tmp_path / "fig.png"
    img.write_bytes(b"\x89PNGFAKE")
    micro = _figure_chunk("c-micro", image_path=str(img), description="stale")
    meso = _figure_chunk("c-meso", image_path=str(img), description="stale")
    meso.level = ChunkLevel.MESO
    insert_chunks(conn, [micro, meso])

    client = MagicMock()
    client.invoke_model.return_value = _mock_bedrock_response("regenerated")
    describer = FigureDescriber(client=client, max_concurrency=1)

    out = describe_figures_in_store(conn, missing_only=False, describer=describer)

    assert out == {"c-micro": "regenerated", "c-meso": "regenerated"}
    assert client.invoke_model.call_count == 1
