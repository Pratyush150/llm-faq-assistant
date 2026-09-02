"""Evaluation harness: hand-computed metrics and a real run on the bundled goldset."""

from __future__ import annotations

import json
import os

import pytest

from faqbot.eval import (
    GoldQuestion,
    GoldSet,
    compare,
    format_comparison_table,
    load_goldset,
    mean_reciprocal_rank,
    percentile,
    recall_at_k,
    run_eval,
)
from faqbot.pipeline import FAQPipeline, PipelineConfig


# -- metric primitives -------------------------------------------------------


def test_recall_at_k_matches_a_hand_computed_value():
    """Three of four queries found a relevant chunk in the top k: 0.75."""
    assert recall_at_k([True, True, False, True]) == pytest.approx(0.75)
    assert recall_at_k([False, False]) == pytest.approx(0.0)
    assert recall_at_k([True]) == pytest.approx(1.0)
    assert recall_at_k([]) == pytest.approx(0.0)


def test_mrr_matches_a_hand_computed_value():
    """(1/1 + 1/2 + 0 + 1/4) / 4 = 0.4375."""
    assert mean_reciprocal_rank([1, 2, None, 4]) == pytest.approx(0.4375)
    assert mean_reciprocal_rank([1, 1]) == pytest.approx(1.0)
    assert mean_reciprocal_rank([None, None]) == pytest.approx(0.0)


def test_percentile_uses_nearest_rank():
    values = list(range(1, 11))
    assert percentile(values, 50) == 5
    assert percentile(values, 90) == 9
    assert percentile(values, 95) == 10
    assert percentile(values, 0) == 1
    assert percentile([], 50) == 0.0


# -- goldset loading ---------------------------------------------------------


def test_goldset_json_round_trip(tmp_path):
    path = os.path.join(str(tmp_path), "gold.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "name": "tiny",
                "corpus": "corpus",
                "questions": [
                    {"id": "a", "question": "How long?", "relevant_sources": ["battery.md"],
                     "expected_contains": ["90 minutes"]},
                    {"id": "b", "question": "Flights?", "answerable": False},
                ],
            },
            fh,
        )
    goldset = load_goldset(path)
    assert goldset.name == "tiny"
    assert len(goldset) == 2
    assert len(goldset.answerable) == 1
    assert len(goldset.unanswerable) == 1
    assert goldset.questions[0].relevant_sources == ("battery.md",)
    assert goldset.corpus.endswith(os.path.join(os.path.basename(str(tmp_path)), "corpus"))


def test_goldset_accepts_a_bare_list(tmp_path):
    path = os.path.join(str(tmp_path), "gold.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([{"id": "a", "question": "How long?"}], fh)
    assert len(load_goldset(path)) == 1


def test_bundled_goldset_has_refusal_cases():
    from conftest import GOLDSET_PATH

    if not os.path.exists(GOLDSET_PATH):
        pytest.skip("bundled goldset is not present")
    goldset = load_goldset(GOLDSET_PATH)
    assert len(goldset) >= 25
    assert len(goldset.unanswerable) >= 5
    assert all(q.qid for q in goldset.questions)
    for q in goldset.answerable:
        assert q.relevant_sources, "answerable question %s has no relevant source" % q.qid


# -- a controlled end-to-end evaluation --------------------------------------


TINY_DOCS = {
    "battery.md": "## How long does the AR-1 battery last?\n\nUp to 90 minutes in Eco mode.\n",
    "filters.md": "## Which filter does the AR-1 use?\n\nPart number NW-FILT-02, washable.\n",
    "warranty.md": "## What is the warranty period?\n\n24 months on the robot and dock.\n",
}


@pytest.fixture()
def tiny_corpus(tmp_path):
    root = os.path.join(str(tmp_path), "corpus")
    os.makedirs(root)
    for name, body in TINY_DOCS.items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as fh:
            fh.write(body)
    return root


def _tiny_goldset() -> GoldSet:
    return GoldSet(
        name="tiny",
        corpus="",
        questions=[
            GoldQuestion("g1", "How long does the AR-1 battery last?", True, ("battery.md",), ("90 minutes",)),
            GoldQuestion("g2", "Which filter does the AR-1 use?", True, ("filters.md",), ("NW-FILT-02",)),
            GoldQuestion("g3", "What is the warranty period?", True, ("warranty.md",), ("24 months",)),
            GoldQuestion("g4", "What is the best sourdough bread recipe?", False),
        ],
    )


def test_run_eval_scores_a_controlled_corpus(tiny_corpus):
    pipe = FAQPipeline(PipelineConfig(label="tiny"))
    pipe.ingest([tiny_corpus])
    report = run_eval(pipe, _tiny_goldset(), k_values=(1, 3))

    assert report.n_questions == 4
    assert report.n_answerable == 3
    assert report.n_unanswerable == 1
    # Three answerable questions, each with its answer in its own document.
    assert report.recall_at[1] == pytest.approx(1.0)
    assert report.recall_at[3] == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.content_accuracy == pytest.approx(1.0)
    assert report.correct_refusals == pytest.approx(1.0)
    assert report.false_refusals == pytest.approx(0.0)
    assert report.groundedness == pytest.approx(1.0)
    assert report.latency_p50 > 0.0
    assert report.latency_p95 >= report.latency_p50


def test_recall_at_1_drops_when_the_goldset_points_elsewhere(tiny_corpus):
    """A sanity check that the metric is measuring retrieval, not just passing."""
    pipe = FAQPipeline(PipelineConfig(label="tiny"))
    pipe.ingest([tiny_corpus])
    wrong = GoldSet(
        name="wrong",
        corpus="",
        questions=[GoldQuestion("g1", "How long does the AR-1 battery last?", True, ("warranty.md",))],
    )
    report = run_eval(pipe, wrong, k_values=(1,))
    assert report.recall_at[1] == pytest.approx(0.0)
    assert report.mrr == pytest.approx(0.0)


def test_report_serialises_and_renders(tiny_corpus):
    pipe = FAQPipeline(PipelineConfig(label="tiny"))
    pipe.ingest([tiny_corpus])
    report = run_eval(pipe, _tiny_goldset())
    payload = report.to_dict()
    assert payload["questions"] == 4
    assert "1" in payload["recall_at"]
    assert len(payload["results"]) == 4
    assert "recall@1" in report.render()
    assert "refusal behaviour" in report.render()
    assert report.to_dict(include_results=False).get("results") is None


def test_failures_lists_only_real_problems(tiny_corpus):
    pipe = FAQPipeline(PipelineConfig(label="tiny"))
    pipe.ingest([tiny_corpus])
    assert run_eval(pipe, _tiny_goldset()).failures() == []


def test_compare_produces_one_row_per_configuration(tiny_corpus):
    configs = [
        PipelineConfig(chunker="structure", label="structure"),
        PipelineConfig(chunker="fixed", chunker_kwargs={"max_tokens": 40, "overlap": 8}, label="fixed40"),
    ]
    reports = compare(configs, _tiny_goldset(), corpus_paths=[tiny_corpus])
    assert [r.label for r in reports] == ["structure", "fixed40"]
    table = format_comparison_table(reports)
    assert "structure" in table
    assert "fixed40" in table
    assert "r@1" in table
    assert len(table.strip().split("\n")) >= 5


def test_compare_without_a_corpus_is_an_error():
    with pytest.raises(ValueError):
        compare([PipelineConfig()], GoldSet(name="x", corpus="", questions=[]))


def test_format_comparison_table_handles_no_reports():
    assert "no reports" in format_comparison_table([])
