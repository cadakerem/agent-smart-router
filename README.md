# Agent Smart Router

> A lightweight, fault-tolerant CLI tool for delegating LLM tasks to expert models across multiple providers (Nvidia NIM, Groq, OpenAI, Anthropic Claude, Gemini).

[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## ⚡ Features
- **Multi-Provider Support**: Seamlessly route requests to `nvidia`, `groq`, `openai`, `anthropic`, or `gemini`.
- **Dynamic Model Discovery**: Never hardcode a model name again. Use `auto-smart` or `auto-fast` and the router will auto-select the best model.
- **Active Liveness Verification**: Pings candidates with a minimal chat request to drop fake/gated models before they crash your task.
- **Automatic Fallbacks**: Provide a comma-separated list of models. If one fails, it instantly falls back to the next.
- **Circuit Breaker**: Built-in health tracking and cooldowns to prevent spamming dead endpoints.
- **Reasoning Extraction**: Automatically extracts and formats hidden `<thought>` or `reasoning` blocks (e.g., from Nemotron).
- **Streaming Native**: Built on the official OpenAI SDK for fast and reliable streaming chunks.

## 🏗️ Architecture & Under the Hood
- **Language**: Python 3
- **Libraries**: openai, filelock, anthropic
- **Design Pattern**: Circuit Breaker, Chain of Responsibility (Fallback Routing), and Dynamic Caching.

## 📦 Installation

```bash
# Install via pip
pip install agent-smart-router

# Or for local development:
# git clone https://github.com/cadakerem/agent-smart-router.git
# cd agent-smart-router
# pip install -e .
```

## 💻 Usage

```bash
# Example command
# Heavy Coding Task (Nvidia Laguna -> Groq Fallback)
agent-smart-router -m "nvidia:poolside/laguna-xs-2.1,groq:groq/compound" -p "Write a python script to parse logs."

# Auto Model Discovery
agent-smart-router -m "groq:auto-smart" -p "Explain quantum entanglement."
```

### ⚠️ Troubleshooting & Known Quirks: Nvidia EULA (404 Not Found)
Nvidia NIM requires users to manually accept the **End User License Agreement (EULA)** for certain models on their website before using them via API. If you haven't accepted the EULA for a dynamically discovered model, Nvidia returns a cryptic `404 Not Found` error.
The Smart Router intercepts this behavior automatically and will print a clear warning.

**How to Fix:**
1. Log into the [Nvidia Build Portal](https://build.nvidia.com).
2. Search for the exact model name shown in the warning and click to run a quick test prompt to accept the terms.
3. Or bypass auto-discovery completely by explicitly hardcoding a model:
```bash
agent-smart-router -m "nvidia:meta/llama-3.2-11b-vision-instruct" -p "Hello"
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

## 📜 License
This project is licensed under the [MIT License](LICENSE).
