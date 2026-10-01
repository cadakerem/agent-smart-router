# LLM Proxy CLI

![CI Status](https://img.shields.io/github/actions/workflow/status/cadakerem/llm-proxy-cli/ci.yml?branch=master&label=CI&logo=github)
![PyPI Version](https://img.shields.io/pypi/v/llm-proxy-cli?color=blue&logo=pypi)
![License](https://img.shields.io/github/license/cadakerem/llm-proxy-cli)

> A lightweight, fault-tolerant CLI tool for delegating LLM tasks to expert models across multiple providers (Nvidia NIM, Groq, OpenAI, Anthropic Claude, Gemini).


## ⚡ Features
- **Multi-Provider Support**: Seamlessly route requests to `nvidia`, `groq`, `openai`, `anthropic`, or `gemini`.
- **Dynamic Model Discovery**: Never hardcode a model name again. Use `auto-smart` or `auto-fast` and the router will auto-select the best model.
- **Active Liveness Verification**: Pings candidates with a minimal chat request to drop fake/gated models before they crash your task.
- **Automatic Fallbacks**: Provide a comma-separated list of models. If one fails, it instantly falls back to the next.
- **Circuit Breaker**: Built-in health tracking and cooldowns to prevent spamming dead endpoints.
- **Reasoning Extraction**: Automatically extracts `reasoning_content` from natively supported models (e.g., DeepSeek-R1 or Nemotron) and outputs them to stderr.
- **Streaming Native**: Built on the official OpenAI SDK for fast and reliable streaming chunks.

## 🏗️ Architecture & Under the Hood
- **Language**: Python 3
- **Libraries**: openai, filelock, anthropic
- **Design Pattern**: Circuit Breaker, Chain of Responsibility (Fallback Routing), and Dynamic Caching.

The router uses a `FileLock`-backed JSON state (`circuit_breaker.json`) to track failures across concurrent runs. 
If an endpoint times out or returns a 5xx error more than `MAX_FAILURES` times, the circuit trips and forces the router to skip that endpoint for the next 120 seconds, immediately trying the next fallback model.


## 📦 Installation

```bash
# Install via pip
pip install llm-proxy-cli

# Or for local development:
# git clone https://github.com/cadakerem/llm-proxy-cli.git
# cd llm-proxy-cli
# pip install -e .
```

## 🔑 Configuration & API Keys

The router looks for API keys in your environment variables or in ~/.config/llm-proxy-cli/keys.json.

Supported environment variables:
- NVIDIA_API_KEY
- GROQ_API_KEY
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GEMINI_API_KEY

## 💻 Usage

The tool is designed to **automatically** discover and use the best model without you having to memorize model names (using `auto-smart` and `auto-fast`).

### 1. Automatic Model Selection (Recommended)
Instead of guessing which model is currently the best or active on the API, simply use `auto-smart` (for complex coding/reasoning tasks) or `auto-fast` (for quick tasks).

```bash
# Auto-select the smartest model on Nvidia (e.g., Nemotron or Llama 3.1 405B)
llm-proxy-cli -m "nvidia:auto-smart" -p "Write a React button."

# Auto-select the fastest model on Groq
llm-proxy-cli -m "groq:auto-fast" -p "Summarize this text."
```

### 2. Chained Automatic Fallback
If Nvidia goes down or hits a rate limit, you can instantly fall back to Groq's best model by separating them with a comma:

```bash
llm-proxy-cli -m "nvidia:auto-smart,groq:auto-smart" -p "Refactor this python script."
```

### 3. Specific / Manual Model Selection
If you have a specific model you want to use, you can still hardcode it directly:

```bash
# Use Laguna, and fallback to a specific Groq model if it fails
llm-proxy-cli -m "nvidia:poolside/laguna-xs-2.1,groq:groq/compound" -p "Explain quantum entanglement."
```

### ⚠️ Troubleshooting & Known Quirks: Nvidia EULA (404 Not Found)
Nvidia NIM requires users to manually accept the **End User License Agreement (EULA)** for certain models on their website before using them via API. If you haven't accepted the EULA for a dynamically discovered model, Nvidia returns a cryptic `404 Not Found` error.
The Smart Router intercepts this behavior automatically and will print a clear warning.

**How to Fix:**
1. Log into the [Nvidia Build Portal](https://build.nvidia.com).
2. Search for the exact model name shown in the warning and click to run a quick test prompt to accept the terms.
3. Or bypass auto-discovery completely by explicitly hardcoding a model:
```bash
llm-proxy-cli -m "nvidia:meta/llama-3.2-11b-vision-instruct" -p "Hello"
```

## 🧑‍💻 Developer & Contributions
Developed by Kerem Barbaros Karnabat ([@cadakerem](https://github.com/cadakerem)).

> **Note on Repository Structure:** The core routing logic, dynamic model discovery, and circuit breaker patterns are entirely contained within `llm_proxy_cli.py` to ensure maximum portability. Unit tests are located in the `tests/` directory, and `SKILL.md` provides instructions for integrating this tool as a native AI agent skill.

Contributions, issues, and feature requests are welcome! Feel free to check the [Issues page](../../issues).

## 📜 License
This project is licensed under the [MIT License](LICENSE).
