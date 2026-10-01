"""
Unit tests for the auto-discovery scoring/filtering logic in llm_proxy_cli.py.
These are all pure functions - no network calls, no API keys needed - so they
run in CI the same way test_circuit.py does. The live verification call itself
(_verify_chat_model) is intentionally NOT unit tested here since it requires a
real API round trip; that's what the --refresh-models manual smoke test is for.
"""
import llm_proxy_cli as router


def test_is_chat_candidate_filters_known_non_chat_models():
    assert not router._is_chat_candidate("meta-llama/llama-prompt-guard-2-86m")
    assert not router._is_chat_candidate("text-embedding-3-large")
    assert not router._is_chat_candidate("whisper-large-v3")
    assert not router._is_chat_candidate("llama-guard-4-12b")


def test_is_chat_candidate_keeps_ordinary_chat_models():
    assert router._is_chat_candidate("llama-3.3-70b-versatile")
    assert router._is_chat_candidate("gpt-oss-120b")
    assert router._is_chat_candidate("gemma2-9b-it")


def test_is_chat_candidate_does_not_catch_everything():
    # Documents the known gap this filter has: a non-chat model whose name gives
    # no hint (e.g. Orpheus, a TTS model) will pass the name filter and must be
    # caught by live verification instead. This is why verification exists.
    assert router._is_chat_candidate("canopylabs/orpheus-v1-english")


def test_score_smart_prefers_bigger_models():
    small = router._score_model_smart("llama-3.1-8b-instant")
    big = router._score_model_smart("llama-3.1-405b-instruct")
    assert big > small


def test_score_smart_prefers_newer_generation_at_similar_size():
    older = router._score_model_smart("llama-3.1-70b-versatile")
    newer = router._score_model_smart("llama-4-70b-versatile")
    assert newer > older


def test_score_fast_prefers_smaller_models():
    small = router._score_model_smart("llama-3.1-8b-instant")
    big = router._score_model_smart("llama-3.1-405b-instruct")
    assert router._score_model_fast("llama-3.1-8b-instant") > router._score_model_fast("llama-3.1-405b-instruct")
    # sanity: smart and fast should disagree on which of these two is "better"
    assert (big > small) != (
        router._score_model_fast("llama-3.1-405b-instruct") > router._score_model_fast("llama-3.1-8b-instant")
    )


def test_score_fast_rewards_speed_keywords():
    plain = router._score_model_fast("llama-3.1-8b")
    instant = router._score_model_fast("llama-3.1-8b-instant")
    assert instant > plain


def test_discover_best_model_returns_none_without_api_key(monkeypatch):
    monkeypatch.setitem(router.PROVIDERS, "groq", {"base_url": "https://api.groq.com/openai/v1", "api_key": None})
    assert router.discover_best_model("groq", mode="smart") is None
