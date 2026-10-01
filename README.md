# LLM Proxy CLI

> A lightweight, fault-tolerant CLI tool for delegating LLM tasks to expert models across multiple providers (Nvidia NIM, Groq, OpenAI, Anthropic Claude, Gemini).


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

## 💻 Usage

Araç, model isimlerini ezberlemene gerek kalmadan **otomatik** olarak en iyi modeli bulacak şekilde tasarlanmıştır (uto-smart ve uto-fast).

### 1. Otomatik Model Seçimi (Önerilen)
Hangi modelin en iyi olduğunu veya API\'da güncel olduğunu düşünmek yerine, sadece uto-smart (zorlu kodlama/mantık işleri için) veya uto-fast (hızlı işler için) kullanın.

`ash
# Nvidia üzerindeki en akıllı modeli otomatik seçer (Örn: Nemotron veya Llama 3.1 405B)
llm-proxy-cli -m "nvidia:auto-smart" -p "Bana bir React butonu yaz."

# Groq üzerindeki en hızlı modeli otomatik seçer
llm-proxy-cli -m "groq:auto-fast" -p "Bu metni özetle."
`

### 2. Zincirleme Otomatik Fallback (Yedekleme)
Nvidia çökerse veya kota dolarsa saniyesinde Groq\'un en iyi modeline geçiş yapsın isterseniz araya virgül koymanız yeterli:

`ash
llm-proxy-cli -m "nvidia:auto-smart,groq:auto-smart" -p "Python scriptini refactor et."
`

### 3. Spesifik / Manuel Model Kullanımı
Eğer özellikle kullanmak istediğiniz belirli bir model varsa, eskisi gibi doğrudan adını da yazabilirsiniz:

`ash
# Laguna modelini kullan, çökerse Groq\'taki özel bir modele geç
llm-proxy-cli -m "nvidia:poolside/laguna-xs-2.1,groq:groq/compound" -p "Kuantum dolanıklığını açıkla."
`

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
