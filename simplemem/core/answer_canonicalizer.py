"""Deterministic post-processing for concise factual answers."""

import re


_ANSWER_LABEL = re.compile(r"^(?:final\s+)?answer\s*[:\-]\s*", re.IGNORECASE)
_BINARY_QUESTION = re.compile(
    r"^\s*(?:is|are|was|were|do|does|did|can|could|will|would|has|have|had|"
    r"should|may|might)\b",
    re.IGNORECASE,
)
_POLARITY = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def canonicalize_answer(answer: str, query: str) -> str:
    """Remove answer labels and compact expanded binary responses.

    The transformation is intentionally conservative: dates, names, numbers,
    and non-binary explanations are retained verbatim after whitespace cleanup.
    """
    text = " ".join(str(answer or "").split())
    if not text:
        return text

    text = _ANSWER_LABEL.sub("", text).strip()
    polarity = _POLARITY.match(text)
    if polarity and _BINARY_QUESTION.match(query or ""):
        return polarity.group(1).capitalize()
    return text
