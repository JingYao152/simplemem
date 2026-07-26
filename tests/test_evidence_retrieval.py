"""Tests for the evidence-based retrieval hit rate (locomo_evidence.py)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locomo_evidence import (
    EVIDENCE_HIT_THRESHOLD,
    RETRIEVAL_METRIC_KEYS,
    EvidenceScorer,
    content_tokens,
)


class Entry:
    """Minimal stand-in for a retrieved MemoryEntry."""

    _counter = 0

    def __init__(self, restatement, keywords=(), entry_id=None):
        Entry._counter += 1
        self.entry_id = entry_id or f"e{Entry._counter}"
        self.lossless_restatement = restatement
        self.keywords = list(keywords)


TURNS = [
    (1, "D1:1", "Hey Mel! Good to see you! How have you been?"),
    (1, "D1:2", "I've been good, just busy with work and the usual."),
    (1, "D1:3", "I went to a LGBTQ support group yesterday and it was so powerful."),
    (2, "D2:1", "How was your week? Anything new going on with you?"),
    (2, "D2:2", "I signed up for a pottery class at the community centre."),
    (3, "D3:1", "I adopted a tabby kitten called Shadow from the shelter."),
]


@pytest.fixture
def scorer():
    return EvidenceScorer(TURNS)


def test_content_tokens_drops_stopwords_and_punctuation():
    assert content_tokens("I went to a LGBTQ support group!") == {
        "went",
        "lgbtq",
        "support",
        "group",
    }
    assert content_tokens("") == set()
    assert content_tokens(None) == set()


def test_a_faithful_restatement_hits_its_evidence_turn(scorer):
    metrics = scorer.score(
        ["D1:3"],
        [
            Entry(
                "Caroline attended an LGBTQ support group on 7 May 2023 and found "
                "it powerful.",
                keywords=["LGBTQ", "support group"],
            )
        ],
    )

    assert metrics["retrieval_hit_any"] == 1.0
    assert metrics["retrieval_hit_all"] == 1.0
    assert metrics["retrieval_coverage"] >= EVIDENCE_HIT_THRESHOLD
    assert metrics["retrieval_session_hit_all"] == 1.0
    assert metrics["retrieval_evidence_count"] == 1.0
    assert set(metrics) == set(RETRIEVAL_METRIC_KEYS)


def test_unrelated_contexts_miss(scorer):
    metrics = scorer.score(
        ["D1:3"], [Entry("Melanie repaired a tractor engine in the barn.")]
    )

    assert metrics["retrieval_hit_any"] == 0.0
    assert metrics["retrieval_coverage"] < EVIDENCE_HIT_THRESHOLD
    assert metrics["retrieval_session_hit_all"] == 0.0


def test_idf_weighting_ignores_filler_only_overlap(scorer):
    """Sharing chatty words with a turn is not a retrieval hit."""
    filler = scorer.score(
        ["D1:3"], [Entry("Melanie said it was so powerful to see the work.")]
    )
    distinctive = scorer.score(
        ["D1:3"], [Entry("Caroline joined the LGBTQ support group.")]
    )

    assert filler["retrieval_hit_any"] == 0.0
    assert distinctive["retrieval_hit_any"] == 1.0
    assert distinctive["retrieval_coverage"] > filler["retrieval_coverage"]


def test_hit_any_and_hit_all_differ_on_multi_evidence_questions(scorer):
    metrics = scorer.score(
        ["D2:2", "D3:1"],
        [Entry("Melanie signed up for a pottery class at the community centre.")],
    )

    assert metrics["retrieval_evidence_count"] == 2.0
    assert metrics["retrieval_hit_any"] == 1.0
    assert metrics["retrieval_hit_all"] == 0.0
    assert 0.0 < metrics["retrieval_coverage"] < 1.0


def test_all_evidence_covered_sets_hit_all(scorer):
    metrics = scorer.score(
        ["D2:2", "D3:1"],
        [
            Entry("Melanie signed up for a pottery class at the community centre."),
            Entry("Melanie adopted a tabby kitten called Shadow from the shelter."),
        ],
    )

    assert metrics["retrieval_hit_all"] == 1.0
    assert metrics["retrieval_session_hit_all"] == 1.0


def test_session_hit_needs_every_evidence_session(scorer):
    metrics = scorer.score(
        ["D2:2", "D3:1"],
        [Entry("Melanie signed up for a pottery class at the community centre.")],
    )

    # Session 2 was reached, session 3 was not.
    assert metrics["retrieval_session_hit_all"] == 0.0


def test_empty_or_unknown_evidence_is_not_scored(scorer):
    assert scorer.score([], [Entry("anything")]) == {}
    assert scorer.score(["D9:99"], [Entry("anything")]) == {}
    assert scorer.score(None, [Entry("anything")]) == {}


def test_no_retrieved_contexts_scores_zero(scorer):
    metrics = scorer.score(["D1:3"], [])

    assert metrics["retrieval_hit_any"] == 0.0
    assert metrics["retrieval_coverage"] == 0.0
    assert metrics["retrieval_session_hit_all"] == 0.0


def test_keywords_count_towards_coverage(scorer):
    metrics = scorer.score(
        ["D3:1"],
        [Entry("A new pet joined the family.", keywords=["Shadow", "tabby", "kitten", "shelter"])],
    )

    assert metrics["retrieval_hit_any"] == 1.0


def test_session_attribution_is_memoized_per_entry(scorer):
    entry = Entry("Melanie signed up for a pottery class.", entry_id="stable")

    scorer.score(["D2:2"], [entry])
    assert scorer._session_cache == {"stable": 2}

    # Second question reuses the cached attribution rather than rescanning.
    scorer.score(["D2:2"], [entry])
    assert scorer._session_cache == {"stable": 2}


def test_from_sample_reads_parsed_locomo_structures():
    class Turn:
        def __init__(self, dia_id, text):
            self.dia_id = dia_id
            self.text = text

    class Session:
        def __init__(self, turns):
            self.turns = turns

    class Conversation:
        def __init__(self, sessions):
            self.sessions = sessions

    class Sample:
        conversation = Conversation(
            {
                1: Session([Turn("D1:1", "I adopted a tabby kitten called Shadow.")]),
                2: Session([Turn("D2:1", "I signed up for a pottery class.")]),
            }
        )

    scorer = EvidenceScorer.from_sample(Sample())

    assert set(scorer.turns) == {"D1:1", "D2:1"}
    assert scorer.turns["D2:1"][0] == 2
    metrics = scorer.score(["D1:1"], [Entry("Shadow the tabby kitten was adopted.")])
    assert metrics["retrieval_hit_any"] == 1.0
