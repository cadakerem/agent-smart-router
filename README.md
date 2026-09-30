# Agent Smart Router (CLI Delegation Tool)

> A lightweight, fault-tolerant CLI tool for delegating LLM tasks to expert models across multiple providers (Nvidia NIM, Groq, OpenAI, Anthropic Claude, Gemini).

[![PyPI version](https://img.shields.io/pypi/v/agent-smart-router.svg)](https://pypi.org/project/agent-smart-router/)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![CI](https://github.com/cadakerem/agent-smart-router/actions/workflows/ci.yml/badge.svg)

## 🚀 Features

- **Multi-Provider Support:** Seamlessly route requests to `nvidia`, `groq`, `openai`, `anthropic`, or `gemini`.
- **Dynamic Model Discovery:** Never hardcode a model name again. Use `auto-smart` or `auto-fast` and the router will auto-select the best model, backed by an active Liveness Verification Ping to drop fake/gated candidates.
- **Automatic Fallbacks:** Provide a comma-separated list of models (or auto aliases). If one fails, it instantly falls back to the next.
- **Circuit Breaker:** Built-in health tracking and cooldowns to prevent spamming dead endpoints.
- **Reasoning Extraction:** Automatically extracts and formats hidden `<thought>` or `reasoning` blocks (e.g., from Nemotron or DeepSeek).
- **Streaming Native:** Built on the official OpenAI SDK for fast and reliable streaming chunks.


## 🔍 Auto Model Discovery

Instead of hardcoding a model name, you can let the router pick one for you dynamically:

```bash
agent-smart-router -m "groq:auto-smart" -p "Explain quantum entanglement."
agent-smart-router -m "groq:auto-fast" -p "Quick sanity check please."
```

- `auto-smart` (aliases: `auto`, `auto-max`) picks the biggest / most capable model currently served by the provider.
- `auto-fast` picks the smallest / snappiest one.

Both query the provider's `/models` endpoint, live-verify the top-scored candidates with a minimal chat request (so gated/TTS/embedding models are never selected), and cache the result for 24h in `auto_models_cache_<provider>_<mode>.json`. Force a fresh lookup with `--refresh-models`.


## ⚠️ Troubleshooting & Known Quirks

### Nvidia EULA (404 Not Found) Handling
Nvidia NIM requires users to manually accept the **End User License Agreement (EULA)** for certain models on their website before using them via API. If you haven't accepted the EULA for a dynamically discovered model, Nvidia returns a cryptic 404 Not Found error instead of a billing/auth error.

The Smart Router intercepts this behavior automatically and will print a clear warning:
[WARNING] Auto-discovery: Nvidia model 'ibm/granite...' requires EULA approval. Please visit https://build.nvidia.com to search and accept the terms for this model.

**How to Fix:**
1. Log into the [Nvidia Build Portal](https://build.nvidia.com).
2. Search for the exact model name shown in the warning.
3. Click to run a quick test prompt in their browser UI to accept the terms.
4. Run the router again—it will now seamlessly pick up the model.

**Bypassing Auto-Discovery (Manual Mode):**
If you don't want to deal with EULA approvals for new models, you can explicitly hardcode a model that you know already works for your account. This completely bypasses the auto-discovery mechanism:
`ash
agent-smart-router -m "nvidia:meta/llama-3.2-11b-vision-instruct,groq:auto-smart" -p "Hello"
`


## 📦 Installation

Since the package is published on PyPI, you can install it globally via `pip`:

```bash
pip install agent-smart-router
```

Alternatively, you can clone the repository for local development:
```bash
git clone https://github.com/cadakerem/agent-smart-router.git
cd agent-smart-router
pip install -e .
```

### Setup API Keys
The router checks for a `keys.json` file in your working directory, or falls back to system environment variables.

**Option 1: keys.json**
Create a `keys.json` file:
```json
{
  "NVIDIA_API_KEY": "nvapi-...",
  "GROQ_API_KEY": "gsk_...",
  "OPENAI_API_KEY": "sk-...",
  "ANTHROPIC_API_KEY": "sk-ant-...",
  "GEMINI_API_KEY": "AIza..."
}
```

**Option 2: Environment Variables**
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="AIza..."
export NVIDIA_API_KEY="nvapi-..."
export GROQ_API_KEY="gsk_..."
```

## 💻 Usage

Once installed, you can use the `agent-smart-router` command globally from your terminal! 
Provide your model fallback chain using `-m`, and your prompt using `-p`.

```bash
agent-smart-router -m "<provider:model1>,<provider:model2>" -p "<your prompt>"
```

You can also read prompts from a file or via standard input:
```bash
agent-smart-router -m "nvidia:nemotron,groq:llama3" -f prompt.txt
cat logs.txt | agent-smart-router -m "groq:llama3"
```

### Examples

**Heavy Coding Task (Nvidia Laguna -> Groq Fallback):**
```bash
agent-smart-router -m "nvidia:poolside/laguna-xs-2.1,groq:groq/compound" -p "Write a python script to parse logs."
```

**Custom Cooldown and Project ID:**
```bash
agent-smart-router -m "groq:llama3" -p "Hello" --project "agent-core" --max-failures 3 --cooldown 300
```

## 🏗️ Architecture Overview

The router uses a `FileLock`-backed JSON state (`circuit_breaker.json`) to track failures across concurrent runs. 
If an endpoint times out or returns a 5xx error more than `MAX_FAILURES` times, the circuit trips and forces the router to skip that endpoint for the next 120 seconds, immediately trying the next fallback model.

