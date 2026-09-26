"""
Integration-style tests for query_ai(): fallback between models, retry behavior,
and its interaction with CircuitBreaker - all with the network mocked out via
unittest.mock, so these run in CI with no API keys and no real HTTP calls.

test_circuit.py already covers CircuitBreaker's own file-lock/open-close logic
in isolation, so here the CircuitBreaker is a MagicMock: we're testing query_ai's
control flow (which model gets tried, when it moves on, when it retries), not
the breaker's persistence.

IMPORTANT: query_ai() constructs a fresh OpenAI(...) client on every retry
attempt, not once per model. Mocking the OpenAI constructor with a plain
side_effect=[client_a, client_b] list is a trap - a retried attempt silently
consumes the *next* list entry (meant for the fallback model) and the test
passes for the wrong reason. openai_factory() below keys the returned mock
client by base_url instead, so it's stable across any number of retries.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import agent_smart_router as router


def make_openai_chunk(content=None, reasoning=None):
    """Fake streaming chunk shaped like the OpenAI SDK's ChatCompletionChunk - just
    enough for query_ai's `chunk.choices[0].delta.content` / .reasoning_content access."""
    delta = SimpleNamespace(content=content)
    if reasoning is not None:
        delta.reasoning_content = reasoning
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


def make_anthropic_stream_cm(text_chunks):
    """Fake context manager shaped like client.messages.stream(...)."""
    cm = MagicMock()
    cm.__enter__.return_value = SimpleNamespace(text_stream=iter(text_chunks))
    cm.__exit__.return_value = False
    return cm


def openai_factory(clients_by_base_url):
    """Fake OpenAI(...) constructor returning a fixed mock client per base_url,
    however many times it's called - see module docstring for why this matters."""
    def _factory(*args, base_url=None, **kwargs):
        return clients_by_base_url[base_url]
    return _factory


GROQ_URL = "https://api.groq.com/openai/v1"
NVIDIA_URL = "https://integrate.api.nvidia.com/v1"


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    # Retry backoff sleeps for real seconds otherwise; skip it in tests.
    monkeypatch.setattr(router.time, "sleep", lambda *_: None)


@pytest.fixture
def mock_cb():
    cb = MagicMock(spec=router.CircuitBreaker)
    cb.check_health.return_value = True
    return cb


def test_query_ai_returns_content_on_first_model_success(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "test-key"})

    client = MagicMock()
    client.chat.completions.create.return_value = [
        make_openai_chunk(content="Hello"), make_openai_chunk(content=" world"),
    ]
    with patch.object(router, "OpenAI", side_effect=openai_factory({GROQ_URL: client})):
        result = router.query_ai("groq:llama-3.3-70b-versatile", "hi", mock_cb)

    assert result == "Hello world"
    mock_cb.record_success.assert_called_once_with("groq:llama-3.3-70b-versatile")
    mock_cb.record_failure.assert_not_called()


def test_query_ai_falls_back_to_next_model_on_failure(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "key-a"})
    monkeypatch.setitem(router.PROVIDERS, "nvidia", {"base_url": NVIDIA_URL, "api_key": "key-b"})

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = Exception("500 Internal Server Error")
    working_client = MagicMock()
    working_client.chat.completions.create.return_value = [make_openai_chunk(content="fallback worked")]

    factory = openai_factory({GROQ_URL: failing_client, NVIDIA_URL: working_client})
    with patch.object(router, "OpenAI", side_effect=factory):
        result = router.query_ai("groq:llama-3.3-70b-versatile,nvidia:nemotron", "hi", mock_cb, max_retries=2)

    assert result == "fallback worked"
    # groq must be retried max_retries=2 times (both against the SAME failing client) before moving on
    assert failing_client.chat.completions.create.call_count == 2
    assert mock_cb.record_failure.call_count == 2
    mock_cb.record_failure.assert_called_with("groq:llama-3.3-70b-versatile")
    mock_cb.record_success.assert_called_once_with("nvidia:nemotron")


def test_query_ai_stops_retrying_immediately_on_404(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "key-a"})
    monkeypatch.setitem(router.PROVIDERS, "nvidia", {"base_url": NVIDIA_URL, "api_key": "key-b"})

    not_found_client = MagicMock()
    not_found_client.chat.completions.create.side_effect = Exception("Error code: 404 - model not found")
    working_client = MagicMock()
    working_client.chat.completions.create.return_value = [make_openai_chunk(content="ok")]

    factory = openai_factory({GROQ_URL: not_found_client, NVIDIA_URL: working_client})
    with patch.object(router, "OpenAI", side_effect=factory):
        result = router.query_ai("groq:does-not-exist,nvidia:nemotron", "hi", mock_cb, max_retries=3)

    assert result == "ok"
    # A 404 must break the retry loop after a single attempt, not consume all max_retries=3.
    assert not_found_client.chat.completions.create.call_count == 1
    assert mock_cb.record_failure.call_count == 1


def test_query_ai_skips_models_in_cooldown(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "key-a"})
    monkeypatch.setitem(router.PROVIDERS, "nvidia", {"base_url": NVIDIA_URL, "api_key": "key-b"})
    # groq is in cooldown, nvidia is healthy
    mock_cb.check_health.side_effect = lambda model_id: model_id != "groq:llama-3.3-70b-versatile"

    working_client = MagicMock()
    working_client.chat.completions.create.return_value = [make_openai_chunk(content="from nvidia")]
    # No entry for GROQ_URL at all - if the code wrongly tried to call groq, this KeyErrors.
    factory = openai_factory({NVIDIA_URL: working_client})
    with patch.object(router, "OpenAI", side_effect=factory):
        result = router.query_ai("groq:llama-3.3-70b-versatile,nvidia:nemotron", "hi", mock_cb)

    assert result == "from nvidia"
    working_client.chat.completions.create.assert_called_once()


def test_query_ai_skips_provider_without_api_key(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": None})
    monkeypatch.setitem(router.PROVIDERS, "nvidia", {"base_url": NVIDIA_URL, "api_key": "key-b"})

    working_client = MagicMock()
    working_client.chat.completions.create.return_value = [make_openai_chunk(content="ok")]
    # No entry for GROQ_URL - a missing-key provider must never reach the OpenAI constructor.
    factory = openai_factory({NVIDIA_URL: working_client})
    with patch.object(router, "OpenAI", side_effect=factory):
        result = router.query_ai("groq:llama-3.3-70b-versatile,nvidia:nemotron", "hi", mock_cb)

    assert result == "ok"
    mock_cb.record_failure.assert_not_called()


def test_query_ai_exits_when_every_model_fails(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "key-a"})

    failing_client = MagicMock()
    failing_client.chat.completions.create.side_effect = Exception("timeout")
    factory = openai_factory({GROQ_URL: failing_client})
    with patch.object(router, "OpenAI", side_effect=factory):
        with pytest.raises(SystemExit) as exc_info:
            router.query_ai("groq:llama-3.3-70b-versatile", "hi", mock_cb, max_retries=2)

    assert exc_info.value.code == 1
    assert failing_client.chat.completions.create.call_count == 2


def test_query_ai_uses_anthropic_client_for_anthropic_provider(monkeypatch, mock_cb):
    if not router.HAS_ANTHROPIC:
        pytest.skip("anthropic package not installed")
    monkeypatch.setitem(router.PROVIDERS, "anthropic", {"api_key": "key-c"})

    mock_client = MagicMock()
    mock_client.messages.stream.return_value = make_anthropic_stream_cm(["Hi", " there"])
    with patch.object(router.anthropic, "Anthropic", return_value=mock_client):
        result = router.query_ai("anthropic:claude-sonnet-4-6", "hi", mock_cb)

    assert result == "Hi there"
    mock_cb.record_success.assert_called_once_with("anthropic:claude-sonnet-4-6")


def test_query_ai_resolves_auto_alias_before_calling_provider(monkeypatch, mock_cb):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": GROQ_URL, "api_key": "key-a"})
    monkeypatch.setattr(router, "discover_best_model", lambda provider, mode="smart", force_refresh=False: "resolved-model-xyz")

    client = MagicMock()
    client.chat.completions.create.return_value = [make_openai_chunk(content="ok")]
    with patch.object(router, "OpenAI", side_effect=openai_factory({GROQ_URL: client})):
        result = router.query_ai("groq:auto-smart", "hi", mock_cb)

    assert result == "ok"
    client.chat.completions.create.assert_called_once_with(
        model="resolved-model-xyz", messages=[{"role": "user", "content": "hi"}],
        temperature=0.7, max_tokens=4096, extra_body=None, stream=True,
    )
    # circuit breaker must key on the resolved model, never on the literal "auto-smart" alias
    mock_cb.check_health.assert_called_with("groq:resolved-model-xyz")
    mock_cb.record_success.assert_called_once_with("groq:resolved-model-xyz")
