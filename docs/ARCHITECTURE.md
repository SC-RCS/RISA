# Architecture

RISA is organized as a runtime wrapper around an existing agent loop rather than as a fine-tuned model or a replacement planner.

## Main layers

- `config.py`
  Loads environment-based runtime configuration.
- `llm_client.py`
  Wraps the OpenAI-compatible client and tracks token usage.
- `core/`
  Implements the core runtime controls:
  - source-aware action authorization
  - observation quarantine
  - response-release protection
  - shared guard prompts and JSON parsing helpers
- `evaluation/offline/`
  Contains the released offline audit entry script.
- `evaluation/online/`
  Contains the released AgentDojo benchmark entry script.

## Runtime flow

1. The agent proposes an action or a final answer.
2. RISA checks high-impact actions before execution.
3. Tool outputs are filtered before they re-enter agent context.
4. Final answers are checked before user release.

## Scope

This repository is a public code release. It is intended to expose the current framework structure and the main experiment entry points, not to bundle every private benchmark asset or backend dependency required for full historical reruns.
