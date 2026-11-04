"""CLI-level tests for `rag metadata` and `rag list` — specifically the
arbitrary key=value tagging (`--attr`/`--unset-attr`) and tag-list handling
(`--tag`/`--clear-tags`) added to close GH #3 ("arbitrary tagging is not
exposed from cli").

Exercises the commands end-to-end against a real on-disk SQLite store,
driving the CLI through Click's ``CliRunner`` — mirrors the pattern used by
``test_cli_stats.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from datasheet_rag.cli import cli
from datasheet_rag.config import get_settings
from datasheet_rag.models.chunk import Chunk, ChunkLevel, ChunkMetadata, LayoutType
from datasheet_rag.project_config import get_project_config
from datasheet_rag.store.schema import connect
from datasheet_rag.store.sqlite import insert_chunks

DOC_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
DOC_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _chunk(doc_id: str) -> Chunk:
    md = ChunkMetadata(
        doc_id=doc_id,
        doc_title=f"Doc {doc_id[:8]}",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=f"{doc_id}:L1:0",
        doc_id=doc_id,
        level=ChunkLevel.MICRO,
        text="chunk 0",
        context_text="chunk 0",
        token_count=2,
        metadata=md,
    )


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Two bare documents (no metadata sidecar row yet), one chunk each."""
    path = tmp_path / "store" / "rag.sqlite"
    conn = connect(path, embedding_dim=get_settings().embedding_dimensions)
    insert_chunks(conn, [_chunk(DOC_A)], project_id="proj-a")
    insert_chunks(conn, [_chunk(DOC_B)], project_id="proj-a")
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _meta(db_path: Path, *args: str):
    """`rag metadata <doc_id> [options]` — shows with no options, sets with them."""
    runner = CliRunner()
    return runner.invoke(cli, ["metadata", *args, "--db", str(db_path)])


def _list(db_path: Path, *args: str):
    """`rag list [filters]` — absorbed the old `metadata list` filters.

    Widened well past the default 80 columns: the sidecar view is eight
    columns, and at 80 Rich squeezes doc_id down until it wraps mid-hash,
    which would have these tests asserting on the wrapping rather than on
    the filtering they exist to check.
    """
    runner = CliRunner()
    return runner.invoke(cli, ["list", *args, "--db", str(db_path)], env={"COLUMNS": "200"})


def _get_json(db_path: Path, doc_id: str) -> dict:
    result = _meta(db_path, doc_id)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


_KEY_FIELDS = ("mpn", "manufacturer", "doc_type")


def _field(meta: dict, field: str) -> str:
    """A metadata field as text; a built-in key reads from ``keys``, a list
    joined, and an absent key reads as "" — that is what clearing it leaves."""
    if field not in _KEY_FIELDS:
        return meta[field]
    value = meta["keys"].get(field, "")
    return ", ".join(value) if isinstance(value, list) else value


def test_tag_sets_full_list(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["tags"] == ["mcu", "reviewed"]


def test_tag_replaces_wholesale_not_additive(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _meta(db_path, DOC_A, "--tag", "power")
    assert result.exit_code == 0, result.output
    # Second call REPLACES the list — "mcu"/"reviewed" do not survive.
    assert _get_json(db_path, DOC_A)["tags"] == ["power"]


def test_tag_omitted_leaves_existing_tags_untouched(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu")
    result = _meta(db_path, DOC_A, "--mpn", "STM32H743VIT6")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["tags"] == ["mcu"]
    assert meta["keys"]["mpn"] == ["STM32H743VIT6"]


def test_clear_tags_wipes_list(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _meta(db_path, DOC_A, "--clear-tags")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["tags"] == []


def test_attr_sets_arbitrary_key_value(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--attr", "revision=B", "--attr", "reviewed_by=hector")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {
        "revision": "B",
        "reviewed_by": "hector",
    }


def test_attr_merges_key_by_key_preserving_others(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "revision=B", "--attr", "notes=draft")
    result = _meta(db_path, DOC_A, "--attr", "revision=C")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {
        "revision": "C",
        "notes": "draft",
    }


def test_unset_attr_removes_single_key(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "revision=B", "--attr", "notes=draft")
    result = _meta(db_path, DOC_A, "--unset-attr", "notes")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {"revision": "B"}


def test_attr_bad_format_rejected(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--attr", "no-equals-sign")
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_attr_empty_key_rejected(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--attr", "=novalue")
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_list_shows_tags_column(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _list(db_path, "--global", "--wide")
    assert result.exit_code == 0, result.output
    assert "mcu" in result.output
    assert "reviewed" in result.output


def test_list_filters_by_tag(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu")
    _meta(db_path, DOC_B, "--tag", "rf")
    result = _list(db_path, "--global", "--tag", "mcu")
    assert result.exit_code == 0, result.output
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


def test_list_filter_by_tag_requires_all_given_tags(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    _meta(db_path, DOC_B, "--tag", "mcu")
    result = _list(db_path, "--global", "--tag", "mcu", "--tag", "reviewed")
    assert result.exit_code == 0, result.output
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


def test_list_filters_by_attr(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "revision=B")
    _meta(db_path, DOC_B, "--attr", "revision=C")
    result = _list(db_path, "--global", "--attr", "revision=B")
    assert result.exit_code == 0, result.output
    assert DOC_A[:10] in result.output
    assert DOC_B[:12] not in result.output


# --- explicit blank values (GH #37) -----------------------------------------
#
# An explicitly passed empty string is a value, not an absent option. Each
# case is checked alone and alongside another flag: the alongside form already
# worked before #37, so only the alone form guards the read-vs-write gate.

_SCALAR_FIELDS = [
    ("--project-id", "project_id"),
    ("--group", "group_name"),
    ("--mpn", "mpn"),
    ("--manufacturer", "manufacturer"),
    ("--subsystem", "subsystem"),
    ("--doc-type", "doc_type"),
]

# A second write flag that touches none of the fields above.
_OTHER_FLAG = ("--attr", "note=x")


def _doc_title(db_path: Path, doc_id: str) -> str:
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    try:
        row = conn.execute("SELECT doc_title FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()
    finally:
        conn.close()
    return row[0]


@pytest.mark.parametrize("extra", [(), _OTHER_FLAG], ids=["alone", "alongside"])
def test_blank_title_clears_title(db_path: Path, extra: tuple[str, ...]) -> None:
    assert _doc_title(db_path, DOC_A) == f"Doc {DOC_A[:8]}"
    result = _meta(db_path, DOC_A, "--title", "", *extra)
    assert result.exit_code == 0, result.output
    assert "Title set" in result.output
    assert _doc_title(db_path, DOC_A) == ""
    # A blank title is still a manual choice, so re-ingest must not refill it.
    assert _get_json(db_path, DOC_A)["attributes"]["title_source"] == "manual"


@pytest.mark.parametrize("extra", [(), _OTHER_FLAG], ids=["alone", "alongside"])
@pytest.mark.parametrize(("flag", "field"), _SCALAR_FIELDS)
def test_blank_scalar_clears_field(
    db_path: Path, flag: str, field: str, extra: tuple[str, ...]
) -> None:
    _meta(db_path, DOC_A, flag, "seeded")
    assert _field(_get_json(db_path, DOC_A), field) == "seeded"
    result = _meta(db_path, DOC_A, flag, "", *extra)
    assert result.exit_code == 0, result.output
    assert "Saved metadata" in result.output
    assert _field(_get_json(db_path, DOC_A), field) == ""


@pytest.mark.parametrize(("flag", "field"), _SCALAR_FIELDS)
def test_blank_scalar_beats_project_config_default(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, field: str
) -> None:
    toml_key = "group" if field == "group_name" else field
    (tmp_path / ".rag.toml").write_text(f'{toml_key} = "from-config"\n')
    monkeypatch.chdir(tmp_path)
    # Alongside another flag, so this reaches the merge whatever the gate does.
    result = _meta(db_path, DOC_A, flag, "", *_OTHER_FLAG)
    assert result.exit_code == 0, result.output
    assert _field(_get_json(db_path, DOC_A), field) == ""


# --- .rag.toml doc_type and [attributes] (GH #26) ----------------------------


def test_project_config_supplies_doc_type_and_attributes(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text(
        'doc_type = "technical-manual"\n[attributes]\nrevision = "A"\nowner = "hw"\n'
    )
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--mpn", "V700")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["keys"]["doc_type"] == "technical-manual"
    assert meta["attributes"] == {"revision": "A", "owner": "hw"}


def test_attr_flags_beat_project_config_attributes(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _meta(db_path, DOC_A, "--attr", "owner=seeded")
    (tmp_path / ".rag.toml").write_text('[attributes]\nrevision = "A"\nowner = "hw"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--attr", "revision=C", "--unset-attr", "owner")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["attributes"] == {"revision": "C"}


def test_bad_project_config_is_a_clean_cli_error(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('[attributes]\nneeds_reembed = "yes"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--mpn", "V700")
    assert result.exit_code == 1
    assert "reserved" in result.output
    assert "Traceback" not in result.output


def test_bad_project_config_fails_scoped_read_commands(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('colour = "red"\n')
    monkeypatch.chdir(tmp_path)
    result = _list(db_path)
    assert result.exit_code == 1
    assert "unknown key(s) colour" in result.output


def test_config_attribute_that_is_now_a_key_is_applied_as_the_key(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('[attributes]\nrevision = "B"\nowner = "hw"\n')
    monkeypatch.chdir(tmp_path)
    assert _keys("add", "revision", "--db", str(db_path)).exit_code == 0

    result = _meta(db_path, DOC_A, "--subsystem", "power")

    assert result.exit_code == 0, result.output
    assert "[attributes] revision: registered key(s) now" in result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["keys"]["revision"] == "B"
    assert meta["attributes"] == {"owner": "hw"}


def test_no_flags_is_still_a_read(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--mpn", "INA226")
    result = _meta(db_path, DOC_A)
    assert result.exit_code == 0, result.output
    assert "Saved metadata" not in result.output
    assert json.loads(result.output)["keys"]["mpn"] == ["INA226"]


# --- registered keys (GH #120) ----------------------------------------------


def _keys(*args: str):
    return CliRunner().invoke(cli, ["keys", *args], env={"COLUMNS": "200"})


def test_mpn_comma_list_and_alias_append(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--mpn", "INA226, INA226A")
    result = _meta(db_path, DOC_A, "--mpn-alias", "INA226B", "--mpn-alias", "INA226A")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["keys"]["mpn"] == ["INA226", "INA226A", "INA226B"]


def test_key_needs_registration_then_sets_and_unsets(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--key", "acme-id=D-1")
    assert result.exit_code == 1
    assert "unregistered key" in result.output and "rag keys add" in result.output
    assert "Traceback" not in result.output

    assert _keys("add", "acme-id", "--db", str(db_path)).exit_code == 0
    assert _meta(db_path, DOC_A, "--key", "acme-id=D-1").exit_code == 0
    assert _get_json(db_path, DOC_A)["keys"]["acme-id"] == "D-1"

    assert _meta(db_path, DOC_A, "--unset-key", "acme-id").exit_code == 0
    assert "acme-id" not in _get_json(db_path, DOC_A)["keys"]


def test_repeated_key_builds_a_list_and_single_key_rejects_it(db_path: Path) -> None:
    _keys("add", "acme-pn", "--shape", "list", "--role", "part", "--db", str(db_path))
    _keys("add", "acme-id", "--db", str(db_path))
    result = _meta(db_path, DOC_A, "--key", "acme-pn=P1", "--key", "acme-pn=P2")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["keys"]["acme-pn"] == ["P1", "P2"]

    result = _meta(db_path, DOC_A, "--key", "acme-id=A", "--key", "acme-id=B")
    assert result.exit_code == 1
    assert "expected a string" in result.output


def test_keys_add_promotes_attribute_and_remove_demotes_it(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--attr", "intel-id=123456")
    result = _keys("add", "intel-id", "--db", str(db_path))
    assert result.exit_code == 0, result.output
    assert "1 document" in result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["keys"]["intel-id"] == "123456"
    assert "intel-id" not in meta["attributes"]

    refused = _keys("remove", "intel-id", "--db", str(db_path))
    assert refused.exit_code == 1 and "demote" in refused.output

    assert _keys("remove", "intel-id", "--demote", "--db", str(db_path)).exit_code == 0
    meta = _get_json(db_path, DOC_A)
    assert meta["attributes"]["intel-id"] == "123456"
    assert "intel-id" not in meta["keys"]


def test_keys_list_and_builtin_cannot_be_removed(db_path: Path) -> None:
    listed = _keys("list", "--db", str(db_path))
    assert listed.exit_code == 0, listed.output
    for name in ("mpn", "manufacturer", "doc_type"):
        assert name in listed.output
    result = _keys("remove", "mpn", "--db", str(db_path))
    assert result.exit_code == 1 and "built in" in result.output


def test_keys_add_rejects_fixed_field_name(db_path: Path) -> None:
    result = _keys("add", "project_id", "--db", str(db_path))
    assert result.exit_code == 1 and "fixed metadata field" in result.output


def test_list_filters_by_key_and_shows_user_keys_in_wide_view(db_path: Path) -> None:
    _keys("add", "acme-id", "--db", str(db_path))
    _meta(db_path, DOC_A, "--key", "acme-id=D-1")
    _meta(db_path, DOC_B, "--key", "acme-id=D-2")
    result = _list(db_path, "--global", "--key", "acme-id=d-1")
    assert result.exit_code == 0, result.output
    # The wide view (implied by the filter) has a column per user key; the
    # extra column squeezes doc_id, so check the key values instead.
    assert "acme-id" in result.output
    assert "D-1" in result.output
    assert "D-2" not in result.output


def test_list_rejects_unregistered_key_filter(db_path: Path) -> None:
    result = _list(db_path, "--global", "--key", "nope=x")
    assert result.exit_code != 0
    assert "unregistered key" in result.output


def test_ingest_tag_help_mentions_replace_semantics() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0, result.output
    assert "--attr key=value" in result.output
    assert "--attr KEY=VALUE" in result.output
