"""Context-inheriting embeddings (design doc section 4, contribution C2).

Two layers, deliberately decoupled:

* **Text layer** stays pure SimpleMem. ``lossless_restatement`` is immutable and
  keeps speaker / absolute time / resolved references inside the sentence, so the
  answering side always sees clean facts.
* **Vector layer** prepends the fact's thread context at embedding time:
  ``"[<one-line thread summary>] <lossless_restatement>"``. The prefix goes
  **only** into the vector - never into any stored text.

Motivation: a lossless restatement is self-contained at *sentence* level but
carries no *topic*-level context. "Melanie bought watercolour brushes" embedded
alone does not know it belongs to a painting thread spanning eight sessions,
which is where open-domain and single-hop misses come from.

Difference from static contextual retrieval: the context is a living thread
summary that is rewritten as the fabric evolves, and it has update semantics -
when a summary is rewritten, the facts Call B names in ``outdated_facts`` are
re-embedded (local compute, zero API cost). ``context_digest`` records which
context a stored vector was built with, which makes re-embedding idempotent.
"""

import hashlib
from typing import Optional


#: Formatting cap for the one-line context prefix, in characters. Plumbing, not
#: a decision knob: it only bounds how much of the living summary is rendered
#: (compare ThreadState.one_line() for the Call A catalogue).
CONTEXT_PREFIX_MAX_CHARS = 200

#: A token this short before a period (or one that already contains a period) is
#: read as an abbreviation rather than a sentence end, so "Melanie moved to St.
#: Louis" is not clipped to "Melanie moved to St.". Formatting heuristic.
_ABBREVIATION_MAX_LEN = 3

_SENTENCE_PUNCTUATION = ".!?"


def context_prefix(summary: Optional[str], title: Optional[str] = "") -> str:
    """One-line rendering of a thread's living summary, for the vector prefix.

    Falls back to the thread title when the summary is empty, and to ``""`` when
    neither exists - in which case the fact is embedded as its bare sentence,
    exactly as pure SimpleMem would.
    """
    text = " ".join((summary or "").split())
    if not text:
        text = " ".join((title or "").split())
    if not text:
        return ""

    # First sentence, so the prefix stays a single topical statement.
    text = _first_sentence(text)

    if len(text) > CONTEXT_PREFIX_MAX_CHARS:
        clipped = text[:CONTEXT_PREFIX_MAX_CHARS]
        boundary = clipped.rfind(" ")
        text = (clipped[:boundary] if boundary > 0 else clipped).rstrip() + "..."
    return text


def _looks_like_abbreviation(text: str) -> bool:
    """True when the token ending at ``text``'s last character reads as "St."."""
    token = text.rsplit(" ", 1)[-1]
    return len(token) <= _ABBREVIATION_MAX_LEN or "." in token


def _first_sentence(text: str) -> str:
    """First sentence of a single-line text, tolerating common abbreviations."""
    for index, char in enumerate(text):
        if char not in _SENTENCE_PUNCTUATION:
            continue
        if index + 1 >= len(text) or text[index + 1] != " ":
            continue
        if char == "." and _looks_like_abbreviation(text[:index]):
            continue
        return text[: index + 1].strip()
    return text.strip()


def contextual_embed_text(prefix: str, restatement: str) -> str:
    """Text handed to the embedder; identical to the fact when there is no prefix."""
    if not prefix:
        return restatement
    return f"[{prefix}] {restatement}"


def context_digest(prefix: str) -> str:
    """Fingerprint of the context a vector was built with (``""`` when none)."""
    if not prefix:
        return ""
    return hashlib.sha1(prefix.encode("utf-8")).hexdigest()[:16]
