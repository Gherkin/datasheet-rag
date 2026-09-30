"""The answer-eval agent loop and its tool sets (GH #93). No AWS: Converse is scripted."""

from __future__ import annotations

import json
from typing import Any

import pytest

from datasheet_rag.eval.agent import (
    MAX_RESULT_CHARS,
    NoTools,
    PdfText,
    PdfTools,
    RagTools,
    run_agent,
)


def _msg(*blocks: dict[str, Any], tin: int = 10, tout: int = 5) -> dict[str, Any]:
    return {
        "output": {"message": {"role": "assistant", "content": list(blocks)}},
        "usage": {"inputTokens": tin, "outputTokens": tout},
    }


def _use(name: str, args: dict[str, Any], uid: str = "t1") -> dict[str, Any]:
    return {"toolUse": {"toolUseId": uid, "name": name, "input": args}}


class Script:
    """A Converse client that replays canned replies and keeps the requests."""

    def __init__(self, *replies: dict[str, Any]):
        self.replies = list(replies)
        self.requests: list[dict[str, Any]] = []

    def converse(self, **kw: Any) -> dict[str, Any]:
        # Deep-copy through JSON: the loop mutates its message list.
        self.requests.append(json.loads(json.dumps(kw, default=str)))
        return self.replies.pop(0)


class EchoTools:
    """One tool, ``echo``, that returns its input."""

    def __init__(self, text: str = "") -> None:
        self.text = text

    def specs(self) -> list[dict[str, Any]]:
        return [{"toolSpec": {"name": "echo", "description": "d", "inputSchema": {"json": {}}}}]

    def call(self, name: str, args: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
        return [{"text": self.text or json.dumps(args)}], False

    def guidance(self) -> str:
        return "Use echo."


SUBMIT = {"answer": "5.25 V", "value": 5.25, "unit": "V", "citations": [{"doc_id": "d", "page": 3}]}


def test_tool_then_submit() -> None:
    client = Script(
        _msg({"text": "looking"}, _use("echo", {"x": 1})),
        _msg(_use("submit_answer", SUBMIT, uid="t2")),
    )
    run = run_agent(client, "m", "q?", EchoTools())
    assert run.stop == "submitted"
    assert run.final is not None and run.final.value == 5.25
    assert run.final.citations[0].page == 3
    assert run.turns == 2
    assert (run.input_tokens, run.output_tokens) == (20, 10)
    assert [c.name for c in run.tool_calls] == ["echo"]
    # The tool result went back under the same toolUseId.
    result = client.requests[1]["messages"][-1]["content"][0]["toolResult"]
    assert result["toolUseId"] == "t1" and result["status"] == "success"
    # The condition's guidance and both tools reached the model.
    assert "Use echo." in client.requests[0]["system"][0]["text"]
    names = [t["toolSpec"]["name"] for t in client.requests[0]["toolConfig"]["tools"]]
    assert names == ["echo", "submit_answer"]


def test_end_turn_without_submit_is_forced() -> None:
    client = Script(_msg({"text": "It is 5.25 V."}), _msg(_use("submit_answer", SUBMIT)))
    run = run_agent(client, "m", "q?", NoTools())
    assert run.final is not None
    assert run.stop == "forced after end_turn"
    assert client.requests[1]["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_answer"}}


def test_max_turns_forces_submit() -> None:
    client = Script(
        _msg(_use("echo", {}, "a")),
        _msg(_use("echo", {}, "b")),
        _msg(_use("submit_answer", SUBMIT, "c")),
    )
    run = run_agent(client, "m", "q?", EchoTools(), max_turns=2)
    assert run.stop == "forced after max turns"
    assert "toolChoice" not in client.requests[1]["toolConfig"]
    assert client.requests[2]["toolConfig"]["toolChoice"] == {"tool": {"name": "submit_answer"}}


def test_no_submit_even_when_forced() -> None:
    client = Script(_msg({"text": "hmm"}), _msg({"text": "still no"}))
    run = run_agent(client, "m", "q?", NoTools())
    assert run.final is None
    assert run.stop == "no submit when forced"


def test_malformed_submit_keeps_the_text() -> None:
    client = Script(_msg(_use("submit_answer", {"answer": "x", "citations": "page 3"})))
    run = run_agent(client, "m", "q?", NoTools())
    assert run.stop == "malformed submit"
    assert run.final is not None and "page 3" in run.final.answer


def test_long_tool_result_is_cut() -> None:
    client = Script(_msg(_use("echo", {})), _msg(_use("submit_answer", SUBMIT, "t2")))
    run_agent(client, "m", "q?", EchoTools(text="x" * (MAX_RESULT_CHARS + 500)))
    text = client.requests[1]["messages"][-1]["content"][0]["toolResult"]["content"][0]["text"]
    assert len(text) < MAX_RESULT_CHARS + 100
    assert text.endswith(f"[... cut at {MAX_RESULT_CHARS} characters]")


# ---- B · raw PDF --------------------------------------------------------------


class FixedText(PdfText):
    def __init__(self, pages: dict[str, list[str]]):
        super().__init__(lambda _doc: b"")
        self._pages = pages


DOCS = [{"doc_id": "docA", "title": "Part A", "mpn": "A1", "manufacturer": "X", "pages": 3}]


def _pdf_tools() -> PdfTools:
    pages = ["intro\nnothing here", "VBUS max 5.25 V\nVDD 3.3 V", "vbus again"]
    return PdfTools(DOCS, FixedText({"docA": pages}))


def test_find_text_reports_pages() -> None:
    blocks, failed = _pdf_tools().call("find_text", {"doc_id": "docA", "pattern": "vbus"})
    assert not failed
    text = blocks[0]["text"]
    assert text.splitlines()[0] == "2 matching line(s)"
    assert "p2: VBUS max 5.25 V" in text and "p3: vbus again" in text


def test_read_pages_marks_pages_and_checks_range() -> None:
    tools = _pdf_tools()
    blocks, failed = tools.call("read_pages", {"doc_id": "docA", "first": 2, "last": 3})
    assert not failed
    assert "=== page 2 ===\nVBUS max" in blocks[0]["text"]
    assert "=== page 3 ===" in blocks[0]["text"]
    _, failed = tools.call("read_pages", {"doc_id": "docA", "first": 3, "last": 9})
    assert failed


def test_pdf_tools_reject_unknown_doc_and_bad_regex() -> None:
    tools = _pdf_tools()
    assert tools.call("find_text", {"doc_id": "nope", "pattern": "x"})[1]
    assert tools.call("find_text", {"doc_id": "docA", "pattern": "("})[1]


def test_list_pdfs() -> None:
    blocks, failed = _pdf_tools().call("list_pdfs", {})
    assert not failed and json.loads(blocks[0]["text"]) == DOCS


# ---- C · datasheet-rag --------------------------------------------------------


def test_rag_tools_expose_the_server_and_scope_to_the_project() -> None:
    pytest.importorskip("mcp")
    from datasheet_rag.backend import LocalBackend
    from datasheet_rag.mcp.server import build_server
    from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata
    from datasheet_rag.store import connect, insert_chunks

    conn = connect(":memory:", embedding_dim=8)
    for doc, project in (("docA", "p1"), ("docB", "p2")):
        chunk = Chunk(
            id=f"{doc}:0",
            doc_id=doc,
            level=ChunkLevel.MICRO,
            text="t",
            context_text="t",
            metadata=ChunkMetadata(doc_id=doc, page_numbers=[1]),
        )
        insert_chunks(conn, [chunk], project_id=project)

    tools = RagTools(build_server(LocalBackend(conn=conn), local_client=False), "p1")
    names = {s["toolSpec"]["name"] for s in tools.specs()}
    assert {"search", "list_documents", "show_page"} <= names
    assert "Server instructions" in tools.guidance()

    blocks, failed = tools.call("list_documents", {})
    assert not failed
    listed = " ".join(b["text"] for b in blocks)
    assert "docA" in listed and "docB" not in listed
