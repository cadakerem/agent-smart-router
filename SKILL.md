---
name: smart-router
description: Delegate complex coding and reasoning tasks to expert models using automatic fallbacks.
---

# Smart Router Delegation Skill

When you need to perform heavy coding, architecture design, logic reasoning, or simply want to parallelize/offload work, you MUST delegate the task to expert models using the CLI router.

## Usage

```bash
python /path/to/llm_proxy_cli.py -m "<provider:model1,provider:model2>" -p "<your_prompt>"
```
You can also read from a file using `-f file.txt` or pipe content via standard input.
*Note: Always provide at least two models separated by a comma for automatic fallback.*

## Supported Providers
Prefix the model name with the provider (e.g., `nvidia:`, `groq:`, `openai:`). If no prefix is provided, it defaults to `nvidia`.
