"""
LLM Client - Handles all LLM interactions
"""
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional
from openai import OpenAI
from simplemem.core.settings import settings as config


def _parse_api_keys(value: Any) -> tuple[str, ...]:
    """Normalize a comma or newline separated API key setting."""
    if isinstance(value, str):
        candidates: Iterable[str] = value.replace("\n", ",").split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = ()

    keys: List[str] = []
    for candidate in candidates:
        key = str(candidate).strip()
        if key and key not in keys:
            keys.append(key)
    return tuple(keys)


class APIKeyPool:
    """Thread-safe round-robin selection for one OpenAI-compatible endpoint."""

    def __init__(self, keys: Iterable[str], cooldown_seconds: float = 30.0):
        self._keys = tuple(keys)
        if not self._keys:
            raise ValueError("At least one OpenAI API key is required")
        if cooldown_seconds < 0:
            raise ValueError("API key cooldown must not be negative")
        self._next_index = 0
        self._cooldown_seconds = cooldown_seconds
        self._cooldown_until: Dict[str, float] = {}
        self._disabled: set[str] = set()
        self._lock = threading.Lock()

    @property
    def keys(self) -> tuple[str, ...]:
        return self._keys

    def next_key(self) -> str:
        with self._lock:
            now = time.monotonic()
            for _ in self._keys:
                key = self._keys[self._next_index]
                self._next_index = (self._next_index + 1) % len(self._keys)
                if key not in self._disabled and self._cooldown_until.get(key, 0) <= now:
                    return key
        raise APIKeyPoolExhaustedError("No API key is currently available")

    def disable(self, key: str) -> None:
        """Exclude a key after an authentication or permission failure."""
        with self._lock:
            self._disabled.add(key)

    def cool_down(self, key: str) -> None:
        """Temporarily exclude a rate-limited key."""
        with self._lock:
            self._cooldown_until[key] = time.monotonic() + self._cooldown_seconds


class APIKeyPoolExhaustedError(RuntimeError):
    """Raised when every configured key is disabled or cooling down."""


class LLMClient:
    """
    Unified LLM client interface
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: Optional[bool] = None,
        use_streaming: Optional[bool] = None
    ):
        configured_keys = _parse_api_keys(getattr(config, "OPENAI_API_KEYS", ""))
        keys = (api_key,) if api_key else configured_keys
        if not keys:
            keys = _parse_api_keys(config.OPENAI_API_KEY)
        if not keys:
            raise ValueError("Set OPENAI_API_KEY or OPENAI_API_KEYS")
        cooldown_seconds = float(getattr(config, "OPENAI_KEY_COOLDOWN_SECONDS", 30))
        self._key_pool = APIKeyPool(keys, cooldown_seconds=cooldown_seconds)
        self.api_keys = self._key_pool.keys
        self.api_key = self.api_keys[0]
        self.model = model or config.LLM_MODEL
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.enable_thinking = enable_thinking if enable_thinking is not None else config.ENABLE_THINKING
        self.use_streaming = use_streaming if use_streaming is not None else config.USE_STREAMING

        self._clients: Dict[str, OpenAI] = {}
        self.client = self._create_openai_client(self.api_key)
        self._clients[self.api_key] = self.client

        if self.base_url:
            print(f"Using custom OpenAI base URL: {self.base_url}")
        if self.enable_thinking:
            print("Deep thinking mode enabled")

    def next_api_key(self) -> str:
        """Return the next configured key for observability and tests."""
        return self._key_pool.next_key()

    def _create_openai_client(self, api_key: str) -> OpenAI:
        """Construct one SDK client while preserving endpoint-wide settings."""
        client_kwargs = {"api_key": api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        timeout_value = os.getenv("OPENAI_TIMEOUT")
        if timeout_value:
            timeout = float(timeout_value)
            if timeout <= 0:
                raise ValueError("OPENAI_TIMEOUT must be positive")
            client_kwargs["timeout"] = timeout

        return OpenAI(**client_kwargs)

    def _client_for_key(self, api_key: str) -> OpenAI:
        client = self._clients.get(api_key)
        if client is None:
            client = self._create_openai_client(api_key)
            self._clients[api_key] = client
        self.client = client
        return client

    @staticmethod
    def _status_code(error: Exception) -> Optional[int]:
        status_code = getattr(error, "status_code", None)
        try:
            return int(status_code) if status_code is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _failure_action(cls, error: Exception) -> str:
        """Classify a failure without relying on a provider-specific exception type."""
        status_code = cls._status_code(error)
        if status_code in {401, 403}:
            return "disable"
        if status_code == 429:
            return "cool_down"
        if status_code is not None and 500 <= status_code < 600:
            return "rotate"
        if error.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }:
            return "rotate"
        return "retry"

    @staticmethod
    def _key_label(api_key: str) -> str:
        return f"...{api_key[-4:]}" if len(api_key) > 4 else "configured key"

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        response_format: Optional[Dict[str, str]] = None,
        max_retries: int = 3
    ) -> str:
        """
        Standard chat completion with optional thinking mode and retry mechanism
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format:
            kwargs["response_format"] = response_format

        # Enable thinking mode if configured (for Qwen and compatible models only)
        # Only add enable_thinking parameter for Qwen API (identified by base_url)
        is_qwen_api = self.base_url and "dashscope.aliyuncs.com" in self.base_url
        
        if is_qwen_api:
            # Qwen API requires explicit enable_thinking parameter
            # - Streaming + thinking: enable_thinking=True
            # - Non-streaming: enable_thinking=False (required, not optional)
            # - JSON format: enable_thinking=False (incompatible with thinking mode)
            if self.use_streaming and self.enable_thinking and not response_format:
                kwargs["extra_body"] = {"enable_thinking": True}
            else:
                # Explicitly set to False for non-streaming calls or JSON format
                kwargs["extra_body"] = {"enable_thinking": False}
        # For OpenAI and other APIs, don't add extra_body parameters

        # Retry mechanism
        last_exception = None
        retry_key: Optional[str] = None
        for attempt in range(max_retries):
            try:
                api_key = retry_key or self._key_pool.next_key()
                client = self._client_for_key(api_key)
                # Use streaming if configured
                if self.use_streaming:
                    kwargs["stream"] = True
                    return self._handle_streaming_response(client, **kwargs)
                else:
                    response = client.chat.completions.create(**kwargs)
                    return response.choices[0].message.content
                
                # kwargs["stream"] = True
                # return self._handle_streaming_response(**kwargs)
                    
            except Exception as e:
                last_exception = e
                action = self._failure_action(e)
                if action == "disable":
                    self._key_pool.disable(api_key)
                    retry_key = None
                elif action == "cool_down":
                    self._key_pool.cool_down(api_key)
                    retry_key = None
                elif action == "rotate":
                    retry_key = None
                else:
                    retry_key = api_key
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    print(
                        "LLM API call failed "
                        f"on {self._key_label(api_key)} "
                        f"(attempt {attempt + 1}/{max_retries}, {e.__class__.__name__})"
                    )
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(
                        "LLM API call failed after "
                        f"{max_retries} attempts ({e.__class__.__name__})"
                    )
        
        # If all retries failed, raise the last exception
        raise last_exception

    def _handle_streaming_response(self, client: OpenAI, **kwargs) -> str:
        """
        Handle streaming response and collect full content
        """
        full_content = []
        stream = client.chat.completions.create(**kwargs)

        # for chunk in stream:
        #     if chunk.choices is not None:
        #         print(chunk.choices[0].delta.content)
        
        # print('---------')

        for chunk in stream:
            # print(chunk)
            # fix list index out of range
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_content.append(content)
                # print(full_content)
                # Optional: print streaming content in real-time
                # print(content, end='', flush=True)
        # print(full_content)
        print()
        return ''.join(full_content)

    def extract_json(self, text: str) -> Any:
        """
        Extract JSON from LLM response with robust parsing
        Supports multiple formats:
        1. Pure JSON
        2. ```json ... ```
        3. ``` ... ``` (generic code block)
        4. JSON embedded in text with common prefixes
        5. Multiple JSON objects (returns first valid one)
        """
        if not text or not text.strip():
            raise ValueError("Empty response received")

        text = text.strip()

        # Remove common LLM prefixes/suffixes
        common_prefixes = [
            "Here's the JSON:",
            "Here is the JSON:",
            "The JSON is:",
            "JSON:",
            "Result:",
            "Output:",
            "Answer:",
        ]
        for prefix in common_prefixes:
            if text.lower().startswith(prefix.lower()):
                text = text[len(prefix):].strip()

        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from ```json ... ``` block
        if "```json" in text.lower():
            # Case insensitive search for ```json
            start_marker = "```json"
            start_idx = text.lower().find(start_marker)
            if start_idx != -1:
                start = start_idx + len(start_marker)
                # Find the closing ```
                end = text.find("```", start)
                if end != -1:
                    json_str = text[start:end].strip()
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Try to clean up common issues
                        json_str = self._clean_json_string(json_str)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

        # Try extracting from generic ``` ... ``` code block
        if "```" in text:
            start = text.find("```") + 3
            # Skip language identifier if present
            newline = text.find("\n", start)
            if newline != -1 and newline - start < 20:
                start = newline + 1
            end = text.find("```", start)
            if end != -1:
                json_str = text[start:end].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to clean up
                    json_str = self._clean_json_string(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass

        # Try finding balanced JSON object/array by scanning for { or [
        for start_char in ['{', '[']:
            result = self._extract_balanced_json(text, start_char)
            if result is not None:
                return result

        # Last resort: try to find any JSON-like structure and clean it
        for start_char in ['{', '[']:
            start_idx = text.find(start_char)
            if start_idx != -1:
                # Extract a large chunk and try to parse
                chunk = text[start_idx:]
                cleaned = self._clean_json_string(chunk)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass

        raise ValueError(f"Failed to extract valid JSON from response. First 300 chars: {text[:300]}...")

    def _clean_json_string(self, json_str: str) -> str:
        """
        Clean common issues in JSON strings from LLM output
        """
        # Remove trailing commas before } or ]
        import re
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

        # Remove comments (// and /* */)
        json_str = re.sub(r'//.*?$', '', json_str, flags=re.MULTILINE)
        json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)

        return json_str.strip()

    def _extract_balanced_json(self, text: str, start_char: str) -> Any:
        """
        Extract a balanced JSON object or array starting with start_char
        """
        end_char = '}' if start_char == '{' else ']'
        start_idx = text.find(start_char)

        if start_idx == -1:
            return None

        # Track depth to find matching closing bracket
        depth = 0
        in_string = False
        escape_next = False

        for i in range(start_idx, len(text)):
            char = text[i]

            # Handle string escaping
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            # Handle strings (don't count brackets inside strings)
            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            # Count depth
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        # Try cleaning and parsing again
                        cleaned = self._clean_json_string(json_str)
                        try:
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            # Continue searching for next occurrence
                            break

        return None
