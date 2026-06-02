# RISA

RISA is a runtime integrity defense framework for tool-using large language model agents. This repository is the public project codebase and exposes the current framework modules, shared runtime utilities, and the main offline and online experiment entry points.

## Repository layout

```text
config.py                     Environment-based runtime configuration
llm_client.py                 Shared OpenAI-compatible client wrapper
core/                         Main RISA runtime modules
evaluation/                   Public experiment entry scripts
docs/                         High-level architecture and evaluation notes
examples/                     Reserved space for lightweight public examples
results/                      Default output location for local runs
requirements.txt              Minimal Python dependencies for the public release
```

## Included components

- source-aware action authorization
- observation quarantine
- final-answer safeguard
- guarded executor
- offline audit entry point
- online AgentDojo benchmark entry point

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env` or your shell environment before running any model-backed script.

## Offline script

```bash
python evaluation/offline/run_offline_eval.py --dataset path/to/dataset.json --output results/offline.json
```

## Online script

```bash
python evaluation/online/run_agentdojo_benchmark.py --mode defense --suites banking,workspace --attacks important_instructions
```

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `core/README.md`
- `evaluation/README.md`

## Scope

This repository is a public framework release, not a complete historical rerun bundle. Some benchmark assets, provider credentials, and private runtime dependencies are intentionally not included here.
