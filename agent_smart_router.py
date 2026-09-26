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

__version__ = "0.2.0"

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Auto model discovery config ---
AUTO_CACHE_TTL = 24 * 60 * 60     # refresh discovery at most once per day
AUTO_DISCOVERY_TIMEOUT = 5        # seconds - keep the "auto" resolve snappy


def get_api_key(provider):
    keys_file = os.path.join(SCRIPT_DIR, "keys.json")
    if os.path.exists(keys_file):
        try:
            with open(keys_file, 'r', encoding='utf-8') as f:
                keys = json.load(f)
            env_name = f"{provider.upper()}_API_KEY"
            if keys.get(env_name):
                return keys[env_name]
        except Exception:
            pass
    return os.environ.get(f"{provider.upper()}_API_KEY")


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
# Usage: pass "provider:auto" instead of a hardcoded model name, e.g.
#   agent-smart-router -m "groq:auto,nvidia:auto" -p "..."
#
# The router hits the provider's /models endpoint, scores every candidate by
# parameter size, generation/version, and role keywords, and picks the best
# one. Results are cached locally for AUTO_CACHE_TTL seconds so normal calls
# never pay the discovery latency.
# ---------------------------------------------------------------------------

def _score_model_id(model_id):
    """Higher score = bigger / newer / more capable, based on the model id string alone."""
    name = model_id.lower()
    score = 0.0

    # Parameter size, e.g. "8b", "70b", "405b" -> biggest wins, weighted heavily.
    size_matches = re.findall(r'(\d+(?:\.\d+)?)\s*b(?!\w)', name)
    if size_matches:
        score += max(float(s) for s in size_matches) * 10
        name_wo_size = re.sub(r'\d+(?:\.\d+)?\s*b(?!\w)', '', name)
    else:
        name_wo_size = name

    # Any remaining digits are treated as generation/version numbers,
    # e.g. "llama-4", "gemini-2.5", "v3.3" -> higher wins.
    version_matches = re.findall(r'(\d+(?:\.\d+)?)', name_wo_size)
    if version_matches:
        score += max(float(v) for v in version_matches) * 5

    for kw, bonus in (("instruct", 3), ("versatile", 3), ("reasoning", 4), ("chat", 1)):
        if kw in name:
            score += bonus
    for kw, penalty in (("mini", -2), ("lite", -2), ("tiny", -3), ("preview", -1), ("deprecated", -100)):
        if kw in name:
            score += penalty

    return score


def _auto_cache_path(provider):
    return os.path.join(SCRIPT_DIR, f"auto_models_cache_{provider}.json")


def _load_auto_cache(provider):
    path = _auto_cache_path(provider)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if time.time() - data.get('timestamp', 0) < AUTO_CACHE_TTL:
                return data.get('best_model')
        except Exception:
            pass
    return None


def _save_auto_cache(provider, best_model):
    path = _auto_cache_path(provider)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'timestamp': time.time(), 'best_model': best_model}, f)
    except Exception:
        pass


def discover_best_model(provider, force_refresh=False):
    """Return the best available model id for `provider`, using a 24h local cache.
    Returns None on any failure so callers can fall back gracefully."""
    if not force_refresh:
        cached = _load_auto_cache(provider)
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

    if not model_ids:
        return None

    best_model = max(model_ids, key=_score_model_id)
    logger.info(f"Auto-discovery: picked '{best_model}' for '{provider}' out of {len(model_ids)} candidates.")
    _save_auto_cache(provider, best_model)
    return best_model


class CircuitBreaker:
    def __init__(self, project_id, max_failures=2, cooldown_seconds=120):
        self.max_failures = max_failures
        self.cooldown_seconds = cooldown_seconds
        safe_proj = "".join([c if c.isalnum() else "_" for c in project_id])
        self.circuit_file = os.path.join(SCRIPT_DIR, f"circuit_breaker_{safe_proj}.json")
        self.lock_file = os.path.join(SCRIPT_DIR, f"circuit_breaker_{safe_proj}.json.lock")

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


def parse_model(model_string, force_refresh_auto=False):
    if ":" in model_string:
        provider, model = model_string.split(":", 1)
        provider, model = provider.strip().lower(), model.strip()
    else:
        provider, model = "nvidia", model_string.strip()

    if model.lower() in ("auto", "auto-max"):
        resolved = discover_best_model(provider, force_refresh=force_refresh_auto)
        if resolved:
            return provider, resolved
        logger.error(f"Auto-discovery unavailable for '{provider}' and no static fallback was given.")
        return provider, model  # will fail fast in query_ai (model not found / no key)

    return provider, model


def query_ai(models_list, prompt, cb: CircuitBreaker, max_retries=2, base_timeout=30, force_refresh_auto=False):
    if isinstance(models_list, str):
        models_list = [m.strip() for m in models_list.split(',')]

    for current_model_str in models_list:
        provider, current_model = parse_model(current_model_str, force_refresh_auto=force_refresh_auto)
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
                    client = anthropic.Anthropic(api_key=provider_config["api_key"], timeout=model_timeout)
                    with client.messages.stream(
                        model=current_model, max_tokens=4096,
                        messages=[{"role": "user", "content": prompt}], temperature=0.7
                    ) as stream:
                        for text in stream.text_stream:
                            full_content += text
                else:
                    client = OpenAI(
                        base_url=provider_config["base_url"], api_key=provider_config["api_key"], timeout=model_timeout
                    )
                    extra_body = {"chat_template_kwargs": {"enable_thinking": True}} if (provider == "nvidia" and "nemotron" in current_model.lower()) else {}
                    completion = client.chat.completions.create(
                        model=current_model, messages=[{"role": "user", "content": prompt}],
                        temperature=0.7, max_tokens=4096, extra_body=extra_body if extra_body else None, stream=True
                    )
                    for chunk in completion:
                        if not chunk.choices: continue
                        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                        if reasoning: full_reasoning += reasoning
                        content = chunk.choices[0].delta.content
                        if content: full_content += content

                cb.record_success(resolved_key)
                output = ""
                if full_reasoning: output += f"--- REASONING ({resolved_key}) ---\n{full_reasoning}\n--- END REASONING ---\n\n"
                output += full_content
                return output

            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Attempt {attempt+1} failed for {resolved_key}: {str(e)}")
                cb.record_failure(resolved_key)
                if "404" in error_msg or "not found" in error_msg or "auth" in error_msg: break
                if attempt == max_retries - 1: break
                time.sleep((2 ** attempt) + random.uniform(0.1, 1.5))

    logger.error("All fallback models failed, timed out, or are in cooldown.")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Smart Router: A fault-tolerant CLI tool for LLM delegation.")
    parser.add_argument("-v", "--version", action="version", version=f"Smart Router v{__version__}")
    parser.add_argument("-m", "--models", required=True, help="Comma-separated list of provider:model fallbacks (e.g. nvidia:nemotron,groq:llama3, or groq:auto).")
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
    response = query_ai(args.models, prompt_text, cb, force_refresh_auto=args.refresh_models)

    # Print the final LLM response to stdout so it can be piped properly
    print(response)


if __name__ == "__main__":
    main()
