import sys
import os
import json
import time
import random
import re
import argparse
import logging
from openai import OpenAI
from filelock import FileLock, Timeout

__version__ = "0.5.1"

# Optional import for anthropic
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Setup Logging
logger = logging.getLogger("smart_router")
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def get_config_dir():
    d = os.path.expanduser("~/.config/llm-proxy-cli")
    os.makedirs(d, exist_ok=True)
    return d

def get_cache_dir():
    d = os.path.expanduser("~/.cache/llm-proxy-cli")
    os.makedirs(d, exist_ok=True)
    return d

# --- Auto model discovery config ---
AUTO_CACHE_TTL = 24 * 60 * 60     # refresh discovery at most once per day
AUTO_DISCOVERY_TIMEOUT = 5        # seconds - keep the "auto" resolve snappy


def get_api_key(provider):
    keys_file = os.path.join(get_config_dir(), "keys.json")
    if os.path.exists(keys_file):
        try:
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys = json.load(f)
            env_name = f"{provider.upper()}_API_KEY"
        except Exception:
            pass
    return os.environ.get(f"{provider.upper()}_API_KEY") or (keys.get(env_name) if 'keys' in locals() else None)


PROVIDERS = {
    "nvidia": {"base_url": "https://integrate.api.nvidia.com/v1", "api_key": get_api_key("NVIDIA")},
    "openai": {"base_url": "https://api.openai.com/v1", "api_key": get_api_key("OPENAI")},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "api_key": get_api_key("GROQ")},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key": get_api_key("GEMINI")},
    "anthropic": {"api_key": get_api_key("ANTHROPIC")}
}


# ---------------------------------------------------------------------------
# Dynamic Model Auto-Discovery
#
# Usage: pass "provider:auto-smart" or "provider:auto-fast" instead of a
# hardcoded model name, e.g.
#   llm-proxy-cli -m "groq:auto-smart,nvidia:auto-fast" -p "..."
# ("auto" and "auto-max" are kept as aliases of "auto-smart" for backwards
# compatibility with existing scripts/cron jobs.)
#
# The router hits the provider's /models endpoint, drops anything that isn't
# a general-purpose chat/completions model (guardrail filters, moderation,
# embeddings, TTS/STT, rerankers - these show up in /models but will 400 on
# a normal chat request), scores what's left, and picks the best one.
# Results are cached locally per (provider, mode) for AUTO_CACHE_TTL seconds
# so normal calls never pay the discovery latency.
# ---------------------------------------------------------------------------

# Model ids containing any of these are never valid chat-completion candidates,
# regardless of mode - excluding them up front is what "auto-fast" needs to
# avoid latching onto a tiny 86M-parameter safety/guard classifier just
# because it has the smallest size in its name.
NON_CHAT_KEYWORDS = (
    "guard", "moderation", "safety", "embed", "embedding", "rerank",
    "whisper", "tts", "speech", "audio", "clip", "classifier",
)


def _is_chat_candidate(model_id):
    name = model_id.lower()
    return not any(kw in name for kw in NON_CHAT_KEYWORDS)


def _score_model_smart(model_id):
    """Higher score = bigger / newer / more capable. Used by auto-smart."""
    name = model_id.lower()
    # Handle MoE like 8x7b -> 56b
    def _moe(m):
        return str(float(m.group(1)) * float(m.group(2))) + 'b'
    name = re.sub(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b', _moe, name)
    # Strip dates and huge context
    name = re.sub(r'20\d{2}[-]?\d{2}[-]?\d{2}', '', name)
    name = re.sub(r'\d{4,}', '', name)
    score = 0.0
    
    if "opus" in name: score += 50
    elif "sonnet" in name: score += 40
    elif "haiku" in name: score += 30

    # Version extraction (handle 3-5 as 3.5 for claude/llama)
    name_wo_size = re.sub(r'\d+(?:\.\d+)?\s*b(?!\w)', '', name)
    ver_str = name_wo_size.replace('_', '-').replace('-r', '-')
    ver_matches = re.findall(r'(\d+(?:[-.]\d+)*)', ver_str)
    best_ver = 0.0
    for v in ver_matches:
        parts = v.replace('-', '.').split('.')
        val = float(f"{parts[0]}.{parts[1]}") if len(parts) >= 2 else (float(parts[0]) if parts[0] else 0.0)
        best_ver = max(best_ver, val)
    score += best_ver * 5

    # Parameter size
    size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*b(?!\w)', name)
    if size_matches:
        score += max(float(s) for s in size_matches) * 10

    for kw, bonus in (("instruct", 3), ("versatile", 3), ("reasoning", 4), ("chat", 1)):
        if kw in name: score += bonus
    for kw, penalty in (("mini", -2), ("lite", -2), ("tiny", -3), ("preview", -1), ("deprecated", -100)):
        if kw in name: score += penalty

    return score


def _score_model_fast(model_id):
    """Higher score = smaller / snappier chat model. Used by auto-fast."""
    name = model_id.lower()
    def _moe(m):
        return str(float(m.group(1)) * float(m.group(2))) + 'b'
    name = re.sub(r'(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)\s*b', _moe, name)
    name = re.sub(r'20\d{2}[-]?\d{2}[-]?\d{2}', '', name)
    name = re.sub(r'\d{4,}', '', name)
    score = 0.0

    if "haiku" in name: score += 20
    elif "sonnet" in name: score += 10

    name_wo_size = re.sub(r'\d+(?:\.\d+)?\s*b(?!\w)', '', name)
    ver_str = name_wo_size.replace('_', '-').replace('-r', '-')
    ver_matches = re.findall(r'(\d+(?:[-.]\d+)*)', ver_str)
    best_ver = 0.0
    for v in ver_matches:
        parts = v.replace('-', '.').split('.')
        val = float(f"{parts[0]}.{parts[1]}") if len(parts) >= 2 else (float(parts[0]) if parts[0] else 0.0)
        best_ver = max(best_ver, val)
    score += best_ver * 5

    size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*b(?!\w)', name)
    if size_matches:
        score -= max(float(s) for s in size_matches) * 10

    for kw, bonus in (("instant", 4), ("flash", 4), ("turbo", 4), ("mini", 3), ("lite", 3), ("small", 2), ("instruct", 1)):
        if kw in name: score += bonus
    for kw, penalty in (("preview", -1), ("deprecated", -100)):
        if kw in name: score += penalty

    return score


_SCORERS = {"smart": _score_model_smart, "fast": _score_model_fast}


def _auto_cache_path(provider, mode):
    return os.path.join(get_cache_dir(), f"auto_models_cache_{provider}_{mode}.json")


def _load_auto_cache(provider, mode):
    path = _auto_cache_path(provider, mode)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if time.time() - data.get('timestamp', 0) < AUTO_CACHE_TTL:
                return data.get('best_model')
        except Exception:
            pass
    return None


def _save_auto_cache(provider, mode, best_model):
    path = _auto_cache_path(provider, mode)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': time.time(), 'best_model': best_model}, f)
    except Exception:
        pass


VERIFY_TIMEOUT = 8          # seconds per live verification call
MAX_VERIFY_CANDIDATES = 3   # how many top-scored candidates to actually try before giving up


def _verify_chat_model(provider, provider_config, model_id):
    """Send one minimal real chat request to confirm this model id actually serves
    chat/completions (not TTS, embeddings, a gated model needing terms acceptance, etc.).
    This is the real safety net - NON_CHAT_KEYWORDS is only a cheap pre-filter to cut
    down how many of these we need to make; the keyword list will always be incomplete
    on its own (see: 'orpheus-v1-english', a TTS model with no matching keyword)."""
    try:
        if provider == "anthropic":
            client = anthropic.Anthropic(api_key=provider_config["api_key"], timeout=VERIFY_TIMEOUT)
            client.messages.create(model=model_id, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
        else:
            client = OpenAI(base_url=provider_config["base_url"], api_key=provider_config["api_key"], timeout=VERIFY_TIMEOUT)
            client.chat.completions.create(model=model_id, messages=[{"role": "user", "content": "hi"}], max_tokens=1)
        return True
    except Exception as e:
        error_str = str(e).lower()
        if provider == "nvidia" and "404" in error_str and "not found for account" in error_str:
            logger.warning(f"Auto-discovery: Nvidia model '{model_id}' requires EULA approval. Please visit https://build.nvidia.com to search and accept the terms for this model.")
        else:
            logger.debug(f"Auto-discovery: '{model_id}' failed live verification: {e}")
        return False


def discover_best_model(provider, mode="smart", force_refresh=False):
    """Return the best available chat model id for `provider` under `mode`
    ("smart" = biggest/most capable, "fast" = smallest/snappiest), using a
    24h local cache per (provider, mode). Returns None on any failure so
    callers can fall back gracefully.

    Only runs on a cache miss (once per day per provider/mode), so paying
    a few extra live-verification calls here is worth it for correctness -
    every call after this one comes straight from cache."""
    if mode not in _SCORERS:
        logger.warning(f"Auto-discovery: unknown mode '{mode}', defaulting to 'smart'.")
        mode = "smart"

    if not force_refresh:
        cached = _load_auto_cache(provider, mode)
        if cached:
            return cached

    provider_config = PROVIDERS.get(provider)
    if not provider_config or not provider_config.get("api_key"):
        logger.warning(f"Auto-discovery: provider '{provider}' not configured or missing API key.")
        return None

    try:
        if provider == "anthropic":
            if not HAS_ANTHROPIC:
                return None
            client = anthropic.Anthropic(api_key=provider_config["api_key"], timeout=AUTO_DISCOVERY_TIMEOUT)
            model_ids = [m.id for m in client.models.list().data]
        else:
            client = OpenAI(
                base_url=provider_config["base_url"], api_key=provider_config["api_key"], timeout=AUTO_DISCOVERY_TIMEOUT
            )
            model_ids = [m.id for m in client.models.list().data]
    except Exception as e:
        logger.warning(f"Auto-discovery failed for '{provider}': {e}")
        return None

    # Cheap pre-filter by name, purely to reduce how many live calls we make below.
    chat_candidates = [m for m in model_ids if _is_chat_candidate(m)]
    if not chat_candidates:
        logger.warning(f"Auto-discovery: no chat-capable models found for '{provider}' (got {len(model_ids)} total, all filtered out by name).")
        return None

    ranked = sorted(chat_candidates, key=_SCORERS[mode], reverse=True)

    for candidate in ranked[:MAX_VERIFY_CANDIDATES]:
        if _verify_chat_model(provider, provider_config, candidate):
            logger.info(f"Auto-discovery ({mode}): picked '{candidate}' for '{provider}' (live-verified) out of {len(chat_candidates)}/{len(model_ids)} name-filtered candidates.")
            _save_auto_cache(provider, mode, candidate)
            return candidate
        logger.warning(f"Auto-discovery ({mode}): '{candidate}' failed live verification (gated, non-chat, or otherwise unusable), trying next candidate.")

    logger.warning(f"Auto-discovery ({mode}): none of the top {min(MAX_VERIFY_CANDIDATES, len(ranked))} name-filtered candidates for '{provider}' passed live verification.")
    return None


class CircuitBreaker:
    def __init__(self, project_id, max_failures=2, cooldown_seconds=120):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        safe_proj = "".join([c if c.isalnum() else "_" for c in project_id])
        self.circuit_file = os.path.join(get_cache_dir(), f"circuit_breaker_{safe_proj}.json")
        self.lock_file = os.path.join(get_cache_dir(), f"circuit_breaker_{safe_proj}.json.lock")

    def load(self):
        for _ in range(3):
            if os.path.exists(self.circuit_file):
                try:
                    with open(self.circuit_file, 'r', encoding='utf-8') as f:
                        return json.load(f)
                except Exception:
                    time.sleep(random.uniform(0.01, 0.05))
        return {}

    def save(self, data):
        for _ in range(3):
            try:
                current_time = time.time()
                active_data = {
                    k: v for k, v in data.items()
                    if v.get('failures', 0) > 0 or v.get('cooldown_until', 0) > current_time
                }
                if not active_data:
                    if os.path.exists(self.circuit_file):
                        os.remove(self.circuit_file)
                    return
                with open(self.circuit_file, 'w', encoding='utf-8') as f:
                    json.dump(active_data, f)
                break
            except Exception:
                time.sleep(random.uniform(0.01, 0.05))

    def check_health(self, model_id):
        try:
            with FileLock(self.lock_file, timeout=5):
                circuit = self.load()
                if model_id in circuit:
                    stats = circuit[model_id]
                    if stats.get('cooldown_until', 0) > time.time():
                        return False
                return True
        except Timeout:
            logger.debug(f"Timeout acquiring lock for {model_id} health check. Assuming healthy.")
            return True

    def record_failure(self, model_id):
        try:
            with FileLock(self.lock_file, timeout=5):
                circuit = self.load()
                if model_id not in circuit:
                    circuit[model_id] = {'failures': 0, 'cooldown_until': 0}
                circuit[model_id]['failures'] += 1
                if circuit[model_id]['failures'] >= self.max_failures:
                    circuit[model_id]['cooldown_until'] = time.time() + self.cooldown_seconds
                    circuit[model_id]['failures'] = 0
                    logger.warning(f"CIRCUIT BREAKER: {model_id} tripped! Cooldown: {self.cooldown_seconds}s.")
                self.save(circuit)
        except Timeout:
            logger.debug(f"Timeout acquiring lock. Could not record failure for {model_id}.")

    def record_success(self, model_id):
        try:
            with FileLock(self.lock_file, timeout=5):
                circuit = self.load()
                if model_id in circuit and circuit[model_id]['failures'] > 0:
                    circuit[model_id]['failures'] = 0
                self.save(circuit)
        except Timeout:
            pass


# "auto" / "auto-max" are kept as aliases of "auto-smart" for backwards compatibility.
_AUTO_ALIASES = {"auto": "smart", "auto-max": "smart", "auto-smart": "smart", "auto-fast": "fast"}


def parse_model(model_string, force_refresh_auto=False):
    if ":" in model_string:
        provider, model = model_string.split(":", 1)
        provider, model = provider.strip().lower(), model.strip()
    else:
        provider, model = "nvidia", model_string.strip()

    mode = _AUTO_ALIASES.get(model.lower())
    if mode:
        resolved = discover_best_model(provider, mode=mode, force_refresh=force_refresh_auto)
        if resolved:
            return provider, resolved
        logger.error(f"Auto-discovery unavailable for '{provider}' ({mode}) and no static fallback was given.")
        return provider, None

    return provider, model


def query_ai(models_list, prompt, cb: CircuitBreaker, max_retries=2, base_timeout=30, force_refresh_auto=False, stream_out=False):
    if isinstance(models_list, str):
        models_list = [m.strip() for m in models_list.split(',')]

    for current_model_str in models_list:
        provider, current_model = parse_model(current_model_str, force_refresh_auto=force_refresh_auto)
        if not current_model:
            continue
        # Circuit breaker tracks the *resolved* model, not the literal "auto" alias,
        # since "auto" can point at a different real model over time.
        resolved_key = f"{provider}:{current_model}"

        if not cb.check_health(resolved_key):
            logger.info(f"Health Check Failed: {resolved_key} is in cooldown. Skipping...")
            continue

        provider_config = PROVIDERS.get(provider)
        if not provider_config or not provider_config.get("api_key"):
            logger.error(f"Provider '{provider}' not configured or missing API key. Skipping...")
            continue

        is_nemotron = "nemotron" in current_model.lower()
        model_timeout = 90 if is_nemotron else base_timeout

        for attempt in range(max_retries):
            try:
                full_reasoning = ""
                full_content = ""
                if provider == "anthropic":
                    if not HAS_ANTHROPIC:
                        raise ImportError("Anthropic package is missing. 'pip install anthropic' required.")
                    client = anthropic.Anthropic(api_key=provider_config["api_key"], timeout=model_timeout, max_retries=0)
                    with client.messages.stream(
                        model=current_model, max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}], temperature=0.7
                    ) as stream:
                        for text in stream.text_stream:
                            if stream_out:
                                import sys
                                sys.stdout.write(text)
                                sys.stdout.flush()
                            full_content += text
                else:
                    client = OpenAI(
                        base_url=provider_config["base_url"], api_key=provider_config["api_key"], timeout=model_timeout, max_retries=0
                    )
                    extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if (provider == "nvidia" and "nemotron" in current_model.lower()) else {}
                    completion = client.chat.completions.create(
                        model=current_model, messages=[{"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=4096, extra_body=extra_body if extra_body else None, stream=True
                    )
                    for chunk in completion:
                        if not chunk.choices: continue
                        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                        if reasoning:
                            full_reasoning += reasoning
                            if stream_out:
                                import sys
                                sys.stderr.write(reasoning)
                                sys.stderr.flush()
                        content = chunk.choices[0].delta.content
                        if content:
                            if stream_out:
                                import sys
                                sys.stdout.write(content)
                                sys.stdout.flush()
                            full_content += content

                cb.record_success(resolved_key)
                output = ""
                if full_reasoning: output += f"--- REASONING ({resolved_key}) ---\n{full_reasoning}\n--- END REASONING ---\n\n"
                output += full_content
                return output

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Attempt {attempt+1} failed for {resolved_key}: {str(e)}")
                status_code = getattr(e, "status_code", None)
                if status_code in (404, 401, 403) or "404" in error_msg or "not found" in error_msg or "auth" in error_msg:
                    break
                if status_code != 429:
                    cb.record_failure(resolved_key)
                if status_code == 429:
                    break
                
                if attempt == max_retries - 1: break
                time.sleep((2 ** attempt) + random.uniform(0.1, 1.5))

    raise RuntimeError("All fallback models failed, timed out, or are in cooldown.")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Smart Router: A fault-tolerant CLI tool for LLM delegation.")
    parser.add_argument("-v", "--version", action="version", version=f"LLM Proxy CLI v{__version__}")
    parser.add_argument("-m", "--models", required=True, help="Comma-separated list of provider:model fallbacks (e.g. nvidia:nemotron,groq:llama3, or groq:auto-smart / groq:auto-fast).")
    parser.add_argument("-p", "--prompt", help="The prompt text to send to the model.")
    parser.add_argument("-f", "--file", help="Path to a text file containing the prompt.")
    parser.add_argument("--project", default="default", help="Project ID for isolating circuit breaker state.")
    parser.add_argument("--max-failures", type=int, default=2, help="Failures before tripping the circuit breaker.")
    parser.add_argument("--cooldown", type=int, default=120, help="Cooldown in seconds when circuit is tripped.")
    parser.add_argument("--refresh-models", action="store_true", help="Ignore the 24h auto-discovery cache and re-query provider model lists now.")
    args = parser.parse_args()

    prompt_text = ""
    if args.prompt:
        prompt_text = args.prompt
    elif args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            sys.exit(1)
    elif not sys.stdin.isatty():
        prompt_text = sys.stdin.read()
    else:
        parser.error("You must provide a prompt via -p, -f, or stdin (piped input).")

    if not prompt_text.strip():
        logger.error("Prompt cannot be empty.")
        sys.exit(1)

    cb = CircuitBreaker(args.project, args.max_failures, args.cooldown)
    try:
        response = query_ai(args.models, prompt_text, cb, force_refresh_auto=args.refresh_models, stream_out=True)
        if not sys.stdout.isatty():
            print(response)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
