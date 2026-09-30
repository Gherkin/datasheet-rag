"""Bootstrap intervals and paired differences (GH #94). Deterministic, no AWS."""

from __future__ import annotations

import pytest

from datasheet_rag.eval.dataset import GoldenItem
from datasheet_rag.eval.metrics import CategoryMetrics, QueryOutcome, compare_by_category
from datasheet_rag.eval.stats import bootstrap_ci, paired_diff


def test_ci_brackets_the_mean() -> None:
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    ci = bootstrap_ci(values, [str(i) for i in range(len(values))])
    assert ci is not None
    assert ci.lo < sum(values) / len(values) < ci.hi


def test_ci_is_reproducible() -> None:
    values = [0.2, 0.9, 0.4, 0.7, 0.1]
    clusters = list("abcde")
    assert bootstrap_ci(values, clusters) == bootstrap_ci(values, clusters)


def test_one_cluster_has_no_ci() -> None:
    # One cluster resamples to itself: a zero-width interval would claim certainty.
    assert bootstrap_ci([1.0, 0.0, 1.0], ["need", "need", "need"]) is None


def test_clusters_widen_the_interval() -> None:
    # Four rephrasings of each need move together. Treating them as sixteen
    # independent queries overstates the evidence four times over.
    per_need = [1.0, 0.0, 1.0, 0.0]
    values = [v for v in per_need for _ in range(4)]
    as_queries = bootstrap_ci(values, [str(i) for i in range(len(values))])
    as_needs = bootstrap_ci(values, [str(i // 4) for i in range(len(values))])
    assert as_queries is not None and as_needs is not None
    assert as_needs.hi - as_needs.lo > as_queries.hi - as_queries.lo


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([1.0, 0.0], ["a"])


def test_paired_diff_small_move_is_n_too_small() -> None:
    # The case from the issue: one query of five moves from rank 1 to 2.
    a = [1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.5, 1.0, 1.0, 1.0, 1.0]
    d = paired_diff(a, b, list("abcde"))
    assert d.diff == pytest.approx(-0.1)
    assert d.verdict == "n too small"
    assert d.ci is not None and d.ci.spans_zero()


def test_paired_diff_consistent_gain_is_better() -> None:
    a = [0.0] * 20
    b = [1.0] * 18 + [0.0] * 2
    d = paired_diff(a, b, [str(i) for i in range(20)])
    assert d.verdict == "better"
    assert d.ci is not None and d.ci.lo > 0


def test_paired_diff_consistent_loss_is_worse() -> None:
    d = paired_diff([1.0] * 20, [0.0] * 20, [str(i) for i in range(20)])
    assert d.verdict == "worse"


def test_paired_diff_identical_runs_is_no_change() -> None:
    d = paired_diff([1.0, 0.0, 1.0], [1.0, 0.0, 1.0], list("abc"))
    assert d.verdict == "no change"
    assert d.diff == 0.0


def test_paired_diff_one_need_is_n_too_small() -> None:
    d = paired_diff([0.0, 0.0], [1.0, 1.0], ["need", "need"])
    assert d.ci is None
    assert d.verdict == "n too small"


# ---- wiring into the metrics ------------------------------------------------


def _outcome(question: str, need: str, hit: float, category: str = "identifier") -> QueryOutcome:
    return QueryOutcome(
        question=question,
        category=category,  # type: ignore[arg-type]
        doc_id="d",
        need_id=need,
        num_retrieved=1,
        reciprocal_rank=hit,
        ndcg=hit,
        hit_at_ks={1: hit},
        hit_at_ks_loose={1: hit},
    )


def test_outcome_carries_need_id_from_item() -> None:
    item = GoldenItem(question="q", category="identifier", doc_id="d", need_id="N1")
    o = QueryOutcome.score(item, [], ks=(1,), ndcg_k=1)
    assert o.need_id == "N1"
    assert o.cluster == "N1"


def test_outcome_without_need_is_its_own_cluster() -> None:
    o = _outcome("q7", "", 1.0)
    o.need_id = None
    assert o.cluster == "q7"


def test_aggregate_reports_intervals_on_every_mean() -> None:
    outcomes = [_outcome(f"q{i}", f"n{i}", float(i % 2)) for i in range(10)]
    m = CategoryMetrics.aggregate(outcomes, ks=(1,))
    for ci, mean in (
        (m.hit_rate_at_k_ci[1], m.hit_rate_at_k[1]),
        (m.hit_rate_at_k_loose_ci[1], m.hit_rate_at_k_loose[1]),
        (m.mrr_ci, m.mrr),
        (m.ndcg_ci, m.ndcg),
    ):
        assert ci is not None and ci.lo <= mean <= ci.hi


def test_compare_by_category_pairs_per_category_and_overall() -> None:
    base = [_outcome(f"q{i}", f"n{i}", 0.0, "table_spec") for i in range(20)]
    base += [_outcome("f1", "nf", 1.0, "figure")]
    variant = [_outcome(f"q{i}", f"n{i}", 1.0, "table_spec") for i in range(20)]
    variant += [_outcome("f1", "nf", 1.0, "figure")]
    cmp = compare_by_category(base, variant, k=1)
    assert set(cmp) == {"table_spec", "figure", "overall"}
    assert cmp["table_spec"].hit_rate.verdict == "better"
    assert cmp["figure"].hit_rate.verdict == "no change"
    assert cmp["overall"].n == 21


def test_compare_by_category_rejects_unpaired_runs() -> None:
    with pytest.raises(ValueError):
        compare_by_category([_outcome("a", "n", 1.0)], [_outcome("b", "n", 1.0)], k=1)


def test_matrix_table_prints_n_too_small_not_a_direction(capsys, monkeypatch) -> None:
    from datasheet_rag import cli
    from datasheet_rag.cli import _render_matrix_table
    from datasheet_rag.eval.harness import RunConfig, RunReport
    from datasheet_rag.eval.metrics import aggregate_by_category

    def report(label: str, hits: list[float]) -> RunReport:
        outs = [_outcome(f"q{i}", f"n{i}", h) for i, h in enumerate(hits)]
        return RunReport(
            config=RunConfig(k=1, ks=(1,), label=label),
            outcomes=outs,
            by_category=aggregate_by_category(outs, ks=(1,)),
        )

    base = report("base", [1.0] * 5)
    worse = report("worse", [0.0] + [1.0] * 4)
    # Wide enough that no cell wraps, so the text can be matched whole.
    monkeypatch.setattr(cli.console, "width", 200)
    _render_matrix_table([base, worse], headline_k=1)
    out = capsys.readouterr().out
    assert "Paired difference" in out
    assert "n too small" in out
    assert "-0.20" not in out
    # The interval is shown, and rich did not eat its brackets as markup.
    assert "[0.40, 1.00]" in out
