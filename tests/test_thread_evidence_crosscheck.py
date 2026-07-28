from simplemem.core.answer_generator import AnswerGenerator
from simplemem.core.models.memory_entry import (
    KIND_FACT,
    KIND_THREAD_SUMMARY,
    MemoryEntry,
)


def _summary(thread_id: str = "t1") -> MemoryEntry:
    return MemoryEntry(
        entry_id=f"thread::{thread_id}",
        lossless_restatement="Alice now lives in Paris.",
        topic="Alice residence",
        kind=KIND_THREAD_SUMMARY,
        thread_id=thread_id,
    )


def _fact(
    entry_id: str,
    thread_id: str,
    text: str,
    valid_until: str = "",
) -> MemoryEntry:
    return MemoryEntry(
        entry_id=entry_id,
        lossless_restatement=text,
        kind=KIND_FACT,
        thread_id=thread_id,
        valid_from="2023-07-01",
        valid_until=valid_until,
    )


def test_crosscheck_groups_thread_summary_with_same_thread_facts():
    generator = AnswerGenerator(
        llm_client=object(),
        enable_thread_evidence_crosscheck=True,
    )
    formatted = generator._format_contexts(
        [
            _summary(),
            _fact("fact-paris", "t1", "Alice moved to Paris on 2023-07-01."),
            _fact("fact-job", "t2", "Alice works as a designer."),
        ]
    )

    assert "[Thread Summaries]" in formatted
    assert "[Thread t1]" in formatted
    assert "[Supporting Facts]" in formatted
    assert "[Fact 1 | thread=t1 | current | supports Thread t1]" in formatted
    assert "Alice moved to Paris on 2023-07-01." in formatted
    assert "[Other Retrieved Facts]" in formatted
    assert "Alice works as a designer." in formatted
    assert formatted.index("[Thread t1]") < formatted.index("[Supporting Facts]")
    assert formatted.index("[Supporting Facts]") < formatted.index(
        "Alice moved to Paris on 2023-07-01."
    )
    assert formatted.index("Alice moved to Paris on 2023-07-01.") < formatted.index(
        "[Other Retrieved Facts]"
    )


def test_crosscheck_marks_closed_fact_as_historical():
    generator = AnswerGenerator(
        llm_client=object(),
        enable_thread_evidence_crosscheck=True,
    )
    formatted = generator._format_contexts(
        [
            _summary(),
            _fact(
                "fact-london",
                "t1",
                "Alice lived in London before moving to Paris.",
                valid_until="2023-07-01",
            ),
        ]
    )

    assert "[Historical or Linked Facts]" in formatted
    assert (
        "[Fact 1 | thread=t1 | historical, superseded on 2023-07-01 | "
        "supports Thread t1]"
    ) in formatted


def test_crosscheck_prompt_requires_fact_priority_over_summary():
    generator = AnswerGenerator(
        llm_client=object(),
        enable_thread_evidence_crosscheck=True,
    )
    prompt = generator._build_answer_prompt(
        "Where does Alice live?",
        generator._format_contexts(
            [
                _summary(),
                _fact("fact-paris", "t1", "Alice moved to Paris on 2023-07-01."),
            ]
        ),
    )

    assert "Thread summaries are compressed orientation only." in prompt
    assert "Verify every state claimed by a thread summary against its supporting facts." in prompt
    assert "If a summary conflicts with any supporting fact, use the fact." in prompt
    assert "For historical questions, respect the fact validity interval and supersede markers." in prompt


def test_crosscheck_disabled_preserves_flat_context_format():
    generator = AnswerGenerator(
        llm_client=object(),
        enable_thread_evidence_crosscheck=False,
    )
    formatted = generator._format_contexts(
        [
            _summary(),
            _fact("fact-paris", "t1", "Alice moved to Paris on 2023-07-01."),
        ]
    )

    assert "[Context 1]" in formatted
    assert "[Thread Summaries]" not in formatted
