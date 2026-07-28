"""Behavioral coverage for OpenAI-compatible API key rotation."""

from types import SimpleNamespace

from simplemem.core.settings import settings as config
from simplemem.core.utils.llm_client import LLMClient


class RateLimitedError(Exception):
    """OpenAI-compatible rate-limit response used at the HTTP boundary."""

    status_code = 429


class AuthenticationFailedError(Exception):
    """OpenAI-compatible authentication response used at the HTTP boundary."""

    status_code = 401


class FakeCompletion:
    """Minimal completion endpoint with a hand-authored success or failure."""

    def __init__(self, outcome):
        self.outcome = outcome

    def create(self, **_kwargs):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.outcome))]
        )


def fake_client(outcome):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletion(outcome))
    )


def test_client_uses_each_configured_key_in_round_robin_order(monkeypatch):
    """A configured key list must start with its first key and then advance."""
    monkeypatch.setattr(config, "OPENAI_API_KEYS", "key-a, key-b\nkey-c", raising=False)

    client = LLMClient(model="test-model", use_streaming=False)

    assert client.api_keys == ("key-a", "key-b", "key-c")
    assert client.next_api_key() == "key-a"
    assert client.next_api_key() == "key-b"
    assert client.next_api_key() == "key-c"
    assert client.next_api_key() == "key-a"


def test_rate_limited_key_switches_to_next_configured_key(monkeypatch):
    """A 429 response must make the same answer request continue on key-b."""
    monkeypatch.setattr(config, "OPENAI_API_KEYS", "key-a,key-b", raising=False)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = LLMClient(model="test-model", use_streaming=False)
    selected_keys = []

    def client_for_key(key):
        selected_keys.append(key)
        return fake_client(RateLimitedError() if key == "key-a" else "answer from key-b")

    monkeypatch.setattr(client, "_client_for_key", client_for_key)
    client.client = fake_client(RateLimitedError())

    answer = client.chat_completion(
        [{"role": "user", "content": "test"}], max_retries=2
    )

    assert answer == "answer from key-b"
    assert selected_keys == ["key-a", "key-b"]


def test_rejected_key_is_disabled_before_the_next_request(monkeypatch):
    """A rejected key must not receive another request after a 401 response."""
    monkeypatch.setattr(config, "OPENAI_API_KEYS", "key-a,key-b", raising=False)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    client = LLMClient(model="test-model", use_streaming=False)
    selected_keys = []

    def client_for_key(key):
        selected_keys.append(key)
        if key == "key-a":
            return fake_client(AuthenticationFailedError())
        return fake_client("answer from key-b")

    monkeypatch.setattr(client, "_client_for_key", client_for_key)

    first_answer = client.chat_completion(
        [{"role": "user", "content": "first"}], max_retries=2
    )
    second_answer = client.chat_completion(
        [{"role": "user", "content": "second"}], max_retries=1
    )

    assert first_answer == "answer from key-b"
    assert second_answer == "answer from key-b"
    assert selected_keys == ["key-a", "key-b", "key-b"]


def test_explicit_api_key_keeps_a_fixed_single_key(monkeypatch):
    """A caller-supplied key must retain the pre-existing fixed-key behavior."""
    monkeypatch.setattr(config, "OPENAI_API_KEYS", "key-a,key-b", raising=False)

    client = LLMClient(
        api_key="fixed-key",
        model="test-model",
        use_streaming=False,
    )

    assert client.api_keys == ("fixed-key",)
    assert client.next_api_key() == "fixed-key"
