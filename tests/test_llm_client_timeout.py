"""Regression coverage for local long-running OpenAI-compatible calls."""

from simplemem.core.utils.llm_client import LLMClient


def test_llm_client_uses_openai_timeout_environment(monkeypatch):
    """An explicit timeout must reach the real OpenAI client instance."""
    monkeypatch.setenv("OPENAI_TIMEOUT", "1800")

    client = LLMClient(
        api_key="test-key",
        model="test-model",
        base_url="http://127.0.0.1:8001/v1",
    )

    assert client.client.timeout == 1800.0
