#!/usr/bin/env python3
"""Check that the configured LLM endpoint and local embedding model are usable.

    python scripts/preflight_llm.py

Exits non-zero with the actual failure, so a long evaluation run fails in
seconds instead of after the first sample. Checks, in order:

1. chat completion against OPENAI_BASE_URL / LLM_MODEL
2. the local embedding model loads (it is downloaded on first use)

Sandboxed environments often allow neither: a network policy that blocks the LLM
host or the model host makes an end-to-end run impossible regardless of the key.
"""

import argparse
import os
import sys

# Runnable as scripts/preflight_llm.py from the repository root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_llm(verbose: bool = True) -> bool:
    from simplemem.core.settings import settings as config

    base_url = getattr(config, "OPENAI_BASE_URL", None)
    model = config.LLM_MODEL
    key = config.OPENAI_API_KEY
    if verbose:
        print(f"LLM      : model={model} base_url={base_url or 'https://api.openai.com/v1'}")
    if not key:
        print("  FAIL: OPENAI_API_KEY is empty")
        return False

    try:
        from simplemem.core.utils.llm_client import LLMClient

        client = LLMClient(use_streaming=False)
        reply = client.chat_completion(
            [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}],
            temperature=0.0,
            max_retries=1,
        )
    except Exception as error:
        print(f"  FAIL: {type(error).__name__}: {error}")
        print(
            "  If this is a proxy/policy denial, the endpoint host is not "
            "reachable from this environment - allow it in the environment's "
            "network policy or run somewhere with egress to it."
        )
        return False

    if verbose:
        print(f"  OK: {str(reply).strip()[:80]}")
    return True


def check_embeddings(verbose: bool = True) -> bool:
    from simplemem.core.settings import settings as config

    if verbose:
        print(f"Embedding: {config.EMBEDDING_MODEL} (dim {config.EMBEDDING_DIMENSION})")
    try:
        from simplemem.core.utils.embedding import EmbeddingModel

        vector = EmbeddingModel().encode_single("preflight", is_query=True)
    except Exception as error:
        print(f"  FAIL: {type(error).__name__}: {error}")
        print(
            "  The embedding model is downloaded from huggingface.co on first "
            "use; a blocked model host fails here even when the LLM works."
        )
        return False

    if verbose:
        print(f"  OK: vector length {len(vector)}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm-only", action="store_true")
    parser.add_argument("--embeddings-only", action="store_true")
    args = parser.parse_args()

    ok = True
    if not args.embeddings_only:
        ok = check_llm() and ok
    if not args.llm_only:
        ok = check_embeddings() and ok

    print("\npreflight:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
