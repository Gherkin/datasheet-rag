"""CLI-level tests for `rag metadata`, `rag list` and `rag fields` — registered
fields (`--field`/`--unset-field`, GH #120) and tag-list handling
(`--tag`/`--clear-tags`, GH #3).

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


def _chunk(rev_id: str) -> Chunk:
    md = ChunkMetadata(
        rev_id=rev_id,
        doc_title=f"Doc {rev_id[:8]}",
        chapter_title="",
        section_title="",
        page_numbers=[1],
        layout_type=LayoutType.TEXT,
        context_string="",
    )
    return Chunk(
        id=f"{rev_id}:L1:0",
        rev_id=rev_id,
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
    insert_chunks(conn, [_chunk(DOC_A)])
    insert_chunks(conn, [_chunk(DOC_B)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture(autouse=True)
def _clear_project_config_cache() -> None:
    get_project_config.cache_clear()
    yield
    get_project_config.cache_clear()


def _meta(db_path: Path, *args: str):
    """`rag metadata <rev_id> [options]` — shows with no options, sets with them."""
    runner = CliRunner()
    return runner.invoke(cli, ["metadata", *args, "--db", str(db_path)])


def _list(db_path: Path, *args: str):
    """`rag list [filters]` — absorbed the old `metadata list` filters.

    Widened well past the default 80 columns: the sidecar view is eight
    columns, and at 80 Rich squeezes rev_id down until it wraps mid-hash,
    which would have these tests asserting on the wrapping rather than on
    the filtering they exist to check.
    """
    runner = CliRunner()
    return runner.invoke(cli, ["list", *args, "--db", str(db_path)], env={"COLUMNS": "200"})


def _get_json(db_path: Path, rev_id: str) -> dict:
    result = _meta(db_path, rev_id)
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


_BUILTIN_FIELDS = ("mpn", "manufacturer", "doc_type")


def _field(meta: dict, field: str) -> str:
    """A metadata value as text; a built-in field reads from ``fields``, a list
    joined, and an absent field reads as "" — that is what clearing it leaves."""
    if field not in _BUILTIN_FIELDS:
        return meta[field]
    value = meta["fields"].get(field, "")
    return ", ".join(value) if isinstance(value, list) else value


def _fields(*args: str):
    return CliRunner().invoke(cli, ["fields", *args], env={"COLUMNS": "200"})


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
    assert meta["fields"]["mpn"] == ["STM32H743VIT6"]


def test_clear_tags_wipes_list(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--tag", "mcu", "--tag", "reviewed")
    result = _meta(db_path, DOC_A, "--clear-tags")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["tags"] == []


def test_field_bad_format_rejected(db_path: Path) -> None:
    for bad in ("no-equals-sign", "=novalue"):
        result = _meta(db_path, DOC_A, "--field", bad)
        assert result.exit_code != 0
        assert "NAME=VALUE" in result.output


def test_attr_options_are_gone(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--attr", "drawing=B")
    assert result.exit_code == 2 and "No such option" in result.output


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


# --- explicit blank values (GH #37) -----------------------------------------
#
# An explicitly passed empty string is a value, not an absent option. Each
# case is checked alone and alongside another flag: the alongside form already
# worked before #37, so only the alone form guards the read-vs-write gate.

#: Document-level shortcuts. --subsystem is per project; see its own tests.
_SCALAR_FIELDS = [
    ("--mpn", "mpn"),
    ("--manufacturer", "manufacturer"),
    ("--doc-type", "doc_type"),
]

# A second write flag that touches none of the fields above.
_OTHER_FLAG = ("--tag", "x")


def _doc_title(db_path: Path, rev_id: str) -> str:
    conn = connect(db_path, embedding_dim=get_settings().embedding_dimensions)
    try:
        row = conn.execute("SELECT doc_title FROM chunks WHERE rev_id = ?", (rev_id,)).fetchone()
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
    assert _get_json(db_path, DOC_A)["title_source"] == "manual"


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
    (tmp_path / ".rag.toml").write_text(f'[fields]\n{field} = "from-config"\n')
    monkeypatch.chdir(tmp_path)
    # Alongside another flag, so this reaches the merge whatever the gate does.
    result = _meta(db_path, DOC_A, flag, "", *_OTHER_FLAG)
    assert result.exit_code == 0, result.output
    assert _field(_get_json(db_path, DOC_A), field) == ""


# --- .rag.toml [fields] (GH #26, GH #120) ------------------------------------


def test_project_config_supplies_fields(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _fields("add", "drawing", "--db", str(db_path)).exit_code == 0
    (tmp_path / ".rag.toml").write_text('[fields]\ndoc_type = "technical-manual"\ndrawing = "A"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--mpn", "V700")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["fields"] == {"mpn": ["V700"], "doc_type": "technical-manual", "drawing": "A"}


def test_field_flags_beat_project_config_fields(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fields("add", "drawing", "--db", str(db_path))
    _fields("add", "owner", "--db", str(db_path))
    _meta(db_path, DOC_A, "--field", "owner=seeded")
    (tmp_path / ".rag.toml").write_text('[fields]\ndrawing = "A"\nowner = "hw"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--field", "drawing=C", "--unset-field", "owner")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["fields"] == {"drawing": "C"}


def test_old_attributes_table_is_an_unknown_key(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('[attributes]\ndrawing = "B"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--mpn", "V700")
    assert result.exit_code == 1
    assert "unknown key(s) attributes" in result.output
    assert "arbitrary values go under [fields]" in result.output
    assert "Traceback" not in result.output


def test_bad_project_config_fails_scoped_read_commands(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('colour = "red"\n')
    monkeypatch.chdir(tmp_path)
    result = _list(db_path)
    assert result.exit_code == 1
    assert "unknown key(s) colour" in result.output


# --- projects (GH #121) ------------------------------------------------------


def test_add_and_remove_project_keep_the_other_projects(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--add-project", "board-a", "--add-project", "board-b")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["projects"] == ["board-a", "board-b"]

    result = _meta(db_path, DOC_A, "--add-project", "board-c", "--remove-project", "board-a")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["projects"] == ["board-b", "board-c"]


def test_remove_project_alone_is_a_write(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a")
    result = _meta(db_path, DOC_A, "--remove-project", "board-a")
    assert result.exit_code == 0, result.output
    assert "Saved metadata" in result.output
    assert _get_json(db_path, DOC_A)["projects"] == []


def test_project_both_added_and_removed_is_rejected(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--add-project", "x", "--remove-project", "x")
    assert result.exit_code == 1
    assert "both to add and to remove" in result.output
    assert "Traceback" not in result.output


def test_blank_project_name_is_rejected(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--add-project", " ")
    assert result.exit_code == 1
    assert "cannot be empty" in result.output


def test_project_id_option_is_gone(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--project-id", "x")
    assert result.exit_code == 2 and "No such option" in result.output


def test_project_config_adds_its_project_on_a_write(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a")
    (tmp_path / ".rag.toml").write_text('project_id = "board-b"\n')
    monkeypatch.chdir(tmp_path)
    get_project_config.cache_clear()  # the call above cached "no .rag.toml"
    result = _meta(db_path, DOC_A, "--tag", "x")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["projects"] == ["board-a", "board-b"]


def test_project_config_does_not_undo_a_remove(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a", "--add-project", "board-b")
    (tmp_path / ".rag.toml").write_text('project_id = "board-b"\n')
    monkeypatch.chdir(tmp_path)
    get_project_config.cache_clear()
    result = _meta(db_path, DOC_A, "--remove-project", "board-b")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["projects"] == ["board-a"]


def test_blank_project_config_project_is_ignored_on_a_write(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = " "\n')
    monkeypatch.chdir(tmp_path)
    get_project_config.cache_clear()
    result = _meta(db_path, DOC_A, "--tag", "x")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["projects"] == []


def test_shared_project_skips_documents_missing_from_the_store(db_path: Path) -> None:
    from types import SimpleNamespace

    from datasheet_rag.backend import LocalBackend
    from datasheet_rag.cli import _shared_project

    _meta(db_path, DOC_A, "--add-project", "board-a", "--add-project", "board-b")
    _meta(db_path, DOC_B, "--add-project", "board-b")
    be = LocalBackend(conn=connect(db_path))
    needs = [SimpleNamespace(rev_id=d) for d in (DOC_A, DOC_B, "not-in-store")]
    assert _shared_project(be, needs) == "board-b"


def test_list_scopes_to_every_project_a_document_is_in(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a", "--add-project", "board-b")
    _meta(db_path, DOC_B, "--add-project", "board-b")
    in_a = _list(db_path, "--project-id", "board-a")
    assert in_a.exit_code == 0, in_a.output
    assert DOC_A[:10] in in_a.output and DOC_B[:10] not in in_a.output
    in_b = _list(db_path, "--project-id", "board-b")
    assert DOC_A[:10] in in_b.output and DOC_B[:10] in in_b.output


# --- subsystem, one value per project (GH #123) -------------------------------


def test_group_options_are_gone(db_path: Path) -> None:
    for args in (["metadata", DOC_A, "--group", "x"], ["list", "--group", "x"]):
        result = CliRunner().invoke(cli, [*args, "--db", str(db_path)])
        assert result.exit_code == 2 and "No such option" in result.output, args


def test_subsystem_needs_a_named_project(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a")
    # The document's only project is not taken as the answer.
    result = _meta(db_path, DOC_A, "--subsystem", "power")
    assert result.exit_code == 1
    assert "one value per project" in result.output and "--project" in result.output
    result = _meta(db_path, DOC_A, "--project", "board-z", "--subsystem", "power")
    assert result.exit_code == 1 and "not in project 'board-z'" in result.output
    assert _get_json(db_path, DOC_A)["project_fields"] == {}


def test_subsystem_differs_per_project_and_list_shows_it(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a", "--add-project", "board-b")
    _meta(db_path, DOC_B, "--add-project", "board-a")
    assert _meta(db_path, DOC_A, "--project", "board-a", "--subsystem", "power").exit_code == 0
    assert _meta(db_path, DOC_A, "--project", "board-b", "--subsystem", "aux").exit_code == 0
    assert _meta(db_path, DOC_B, "--project", "board-a", "--subsystem", "rf").exit_code == 0
    assert _get_json(db_path, DOC_A)["project_fields"] == {
        "board-a": {"subsystem": "power"},
        "board-b": {"subsystem": "aux"},
    }

    in_a = _list(db_path, "--project-id", "board-a", "--subsystem", "power")
    assert in_a.exit_code == 0, in_a.output
    assert DOC_A[:10] in in_a.output and DOC_B[:10] not in in_a.output
    assert "aux" not in in_a.output  # board-b's value is not this listing's
    in_b = _list(db_path, "--project-id", "board-b", "--subsystem", "power")
    assert "No ingested documents match" in in_b.output

    everywhere = _list(db_path, "--global", "--columns", "subsystem")
    assert everywhere.exit_code == 0, everywhere.output
    assert "board-a: power; board-b: aux" in everywhere.output

    # A blank value clears it in that project only.
    assert _meta(db_path, DOC_A, "--project", "board-b", "--subsystem", "").exit_code == 0
    assert _get_json(db_path, DOC_A)["project_fields"] == {"board-a": {"subsystem": "power"}}


def test_project_config_sets_subsystem_in_its_project(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text('project_id = "netdaq"\n[fields]\nsubsystem = "power"\n')
    monkeypatch.chdir(tmp_path)
    result = _meta(db_path, DOC_A, "--tag", "x")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["projects"] == ["netdaq"]
    assert meta["project_fields"] == {"netdaq": {"subsystem": "power"}}


_BOARD_A_TOML = 'project_id = "board-a"\n[fields]\nsubsystem = "power"\n'


def test_project_config_subsystem_does_not_block_leaving_its_project(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text(_BOARD_A_TOML)
    monkeypatch.chdir(tmp_path)
    assert _meta(db_path, DOC_A, "--tag", "x").exit_code == 0
    result = _meta(db_path, DOC_A, "--remove-project", "board-a")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["projects"] == [] and meta["project_fields"] == {}


def test_project_config_checks_compare_stripped_project_names(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text(_BOARD_A_TOML)
    monkeypatch.chdir(tmp_path)
    # The store strips project names; the .rag.toml checks must too.
    result = _meta(db_path, DOC_A, "--project", " board-a ", "--tag", "x")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["project_fields"] == {"board-a": {"subsystem": "power"}}
    result = _meta(db_path, DOC_A, "--remove-project", "board-a ")
    assert result.exit_code == 0, result.output
    meta = _get_json(db_path, DOC_A)
    assert meta["projects"] == [] and meta["project_fields"] == {}
    # A blank --project is no project: the config's board-a applies.
    result = _meta(db_path, DOC_A, "--project", " ", "--tag", "y")
    assert result.exit_code == 0, result.output
    assert "left out" not in result.output
    assert _get_json(db_path, DOC_A)["project_fields"] == {"board-a": {"subsystem": "power"}}


def test_project_config_subsystem_is_left_out_with_a_note(
    db_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".rag.toml").write_text(_BOARD_A_TOML)
    monkeypatch.chdir(tmp_path)
    # Joining another board only: the doc is not in board-a, where the
    # config's subsystem belongs. The rest of the write goes ahead.
    result = _meta(db_path, DOC_A, "--add-project", "board-b", "--mpn", "X")
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())  # Rich wraps the note
    assert "sets subsystem for project 'board-a'; left out" in output
    assert "is not in that project" in output
    meta = _get_json(db_path, DOC_A)
    assert meta["projects"] == ["board-b"] and meta["project_fields"] == {}
    # A pure membership change on another board is not blocked either.
    result = _meta(db_path, DOC_A, "--remove-project", "board-b")
    assert result.exit_code == 0, result.output
    # Writing in board-b: board-a's subsystem is not carried there.
    result = _meta(db_path, DOC_A, "--add-project", "board-b")
    assert result.exit_code == 0, result.output
    result = _meta(db_path, DOC_A, "--project", "board-b", "--mpn", "Y")
    assert result.exit_code == 0, result.output
    assert "this write goes to project 'board-b'" in " ".join(result.output.split())
    assert _get_json(db_path, DOC_A)["project_fields"] == {}
    # A value given on the command line is what the user asked for.
    result = _meta(
        db_path, DOC_A, "--add-project", "board-b", "--project", "board-b", "--subsystem", "aux"
    )
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["project_fields"] == {"board-b": {"subsystem": "aux"}}


def test_clearing_a_subsystem_outside_the_project_prints_a_note(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--add-project", "board-a")
    result = _meta(db_path, DOC_A, "--project", "gone", "--unset-field", "subsystem")
    assert result.exit_code == 0, result.output
    output = " ".join(result.output.split())  # Rich wraps the note
    assert "not in project 'gone'" in output and "nothing was cleared" in output
    # A blank value is a clear too.
    result = _meta(db_path, DOC_A, "--project", "gone", "--field", "subsystem= ")
    assert result.exit_code == 0, result.output
    assert "nothing was cleared" in " ".join(result.output.split())
    # In a project the document is in, no note.
    result = _meta(db_path, DOC_A, "--project", "board-a", "--unset-field", "subsystem")
    assert result.exit_code == 0, result.output
    assert "Note:" not in result.output


def test_project_alone_is_not_a_write(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--project", "board-a")
    assert result.exit_code == 2 and "--project" in result.output


def test_fields_add_project_scope(db_path: Path) -> None:
    result = _fields("add", "board-ref", "--scope", "project", "--db", str(db_path))
    assert result.exit_code == 0, result.output
    assert "per project" in result.output
    listed = _fields("list", "--db", str(db_path))
    assert "subsystem" in listed.output and "project" in listed.output


def test_no_flags_is_still_a_read(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--mpn", "INA226")
    result = _meta(db_path, DOC_A)
    assert result.exit_code == 0, result.output
    assert "Saved metadata" not in result.output
    assert json.loads(result.output)["fields"]["mpn"] == ["INA226"]


# --- registered fields (GH #120) ---------------------------------------------


def test_mpn_is_repeatable_and_alias_appends(db_path: Path) -> None:
    _meta(db_path, DOC_A, "--mpn", "INA226", "--mpn", "INA226A")
    result = _meta(db_path, DOC_A, "--mpn-alias", "INA226B", "--mpn-alias", "INA226A")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["fields"]["mpn"] == ["INA226", "INA226A", "INA226B"]


def test_mpn_does_not_split_on_commas(db_path: Path) -> None:
    assert _meta(db_path, DOC_A, "--mpn", "A,B").exit_code == 0
    assert _get_json(db_path, DOC_A)["fields"]["mpn"] == ["A,B"]


def test_field_needs_registration_then_sets_and_unsets(db_path: Path) -> None:
    result = _meta(db_path, DOC_A, "--field", "acme-id=D-1")
    assert result.exit_code == 1
    assert "unregistered field" in result.output and "rag fields add" in result.output
    assert "Traceback" not in result.output

    assert _fields("add", "acme-id", "--db", str(db_path)).exit_code == 0
    assert _meta(db_path, DOC_A, "--field", "acme-id=D-1").exit_code == 0
    assert _get_json(db_path, DOC_A)["fields"]["acme-id"] == "D-1"

    assert _meta(db_path, DOC_A, "--unset-field", "acme-id").exit_code == 0
    assert "acme-id" not in _get_json(db_path, DOC_A)["fields"]


def test_repeated_field_builds_a_list_and_single_field_rejects_it(db_path: Path) -> None:
    _fields("add", "acme-pn", "--shape", "list", "--role", "part", "--db", str(db_path))
    _fields("add", "acme-id", "--db", str(db_path))
    result = _meta(db_path, DOC_A, "--field", "acme-pn=P1", "--field", "acme-pn=P2")
    assert result.exit_code == 0, result.output
    assert _get_json(db_path, DOC_A)["fields"]["acme-pn"] == ["P1", "P2"]

    result = _meta(db_path, DOC_A, "--field", "acme-id=A", "--field", "acme-id=B")
    assert result.exit_code == 1
    assert "expected a string" in result.output


def test_fields_remove_refuses_while_held_then_clear_deletes(db_path: Path) -> None:
    _fields("add", "intel-id", "--db", str(db_path))
    _meta(db_path, DOC_A, "--field", "intel-id=123456")

    refused = _fields("remove", "intel-id", "--db", str(db_path))
    assert refused.exit_code == 1 and "clear" in refused.output
    assert _get_json(db_path, DOC_A)["fields"]["intel-id"] == "123456"

    result = _fields("remove", "intel-id", "--clear", "--db", str(db_path))
    assert result.exit_code == 0, result.output
    assert "1 document" in result.output
    assert "intel-id" not in _get_json(db_path, DOC_A)["fields"]


def test_fields_list_and_builtin_cannot_be_removed(db_path: Path) -> None:
    listed = _fields("list", "--db", str(db_path))
    assert listed.exit_code == 0, listed.output
    for name in ("mpn", "manufacturer", "doc_type"):
        assert name in listed.output
    result = _fields("remove", "mpn", "--db", str(db_path))
    assert result.exit_code == 1 and "built in" in result.output


def test_fields_add_rejects_reserved_name(db_path: Path) -> None:
    for name in ("project_id", "needs_reembed"):
        result = _fields("add", name, "--db", str(db_path))
        assert result.exit_code == 1 and "reserved" in result.output


def test_list_filters_by_field_and_shows_user_fields_in_wide_view(db_path: Path) -> None:
    _fields("add", "acme-id", "--db", str(db_path))
    _meta(db_path, DOC_A, "--field", "acme-id=D-1")
    _meta(db_path, DOC_B, "--field", "acme-id=D-2")
    result = _list(db_path, "--global", "--field", "acme-id=d-1")
    assert result.exit_code == 0, result.output
    # The wide view (implied by the filter) has a column per user field; the
    # extra column squeezes rev_id, so check the field values instead.
    assert "acme-id" in result.output
    assert "D-1" in result.output
    assert "D-2" not in result.output


def test_list_rejects_unregistered_field_filter(db_path: Path) -> None:
    result = _list(db_path, "--global", "--field", "nope=x")
    assert result.exit_code != 0
    assert "unregistered field" in result.output


def test_ingest_help_names_field_not_attr() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0, result.output
    assert "--field NAME=VALUE" in result.output
    assert "--attr" not in result.output
