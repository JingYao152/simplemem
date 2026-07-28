import json

from simplemem.core.answer_canonicalizer import canonicalize_answer
from simplemem.core.answer_generator import AnswerGenerator
from simplemem.core.models.memory_entry import MemoryEntry


class AnswerLLM:
    """Minimal answer-model double for exercising AnswerGenerator output."""

    @staticmethod
    def chat_completion(messages, temperature=0.1, response_format=None):
        return json.dumps(
            {"reasoning": "The context supplies the answer.",
             "answer": "No, Dave's shop does not employ a lot of people."}
        )

    @staticmethod
    def extract_json(response):
        return json.loads(response)


def test_canonicalize_binary_answer_keeps_only_polarity():
    answer = canonicalize_answer(
        "Yes, Calvin wants to expand his brand worldwide.",
        "Does Calvin want to expand his brand worldwide?",
    )

    assert answer == "Yes"


def test_canonicalize_answer_removes_label_and_preserves_date_and_name():
    answer = canonicalize_answer(
        "Answer: Alice met Bob on 16 June 2023.",
        "When did Alice meet Bob?",
    )

    assert answer == "Alice met Bob on 16 June 2023."


def test_answer_generator_canonicalizes_only_when_the_switch_is_enabled():
    contexts = [MemoryEntry(lossless_restatement="Dave owns a shop.")]
    question = "Does Dave's shop employ a lot of people?"

    enabled = AnswerGenerator(
        llm_client=AnswerLLM(),
        enable_answer_canonicalization=True,
    ).generate_answer(question, contexts)
    disabled = AnswerGenerator(
        llm_client=AnswerLLM(),
        enable_answer_canonicalization=False,
    ).generate_answer(question, contexts)

    assert enabled == "No"
    assert disabled == "No, Dave's shop does not employ a lot of people."
