# Architecture

RISA wraps an existing agent loop with runtime checks.

## Main layers

- `config.py` — runtime configuration
- `llm_client.py` — async LLM client
- `core/` — action authorization, observation quarantine, response-release protection, guard prompts
- `evaluation/offline/` — offline audit entry point
- `evaluation/online/` — online benchmark entry point

## Runtime flow

1. The agent proposes an action or a final answer.
2. High-impact actions are checked before execution.
3. Tool outputs are filtered before re-entering agent context.
4. Final answers are checked before release.
