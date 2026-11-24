# RISA

RISA is a runtime integrity defense framework for tool-using large language model agents.

## Layout

```text
config.py                  Runtime configuration
llm_client.py              Async LLM client wrapper
configs/                   Endpoint and policy configuration
core/                      Runtime modules
docs/                      Design notes
evaluation/                Evaluation entry points
examples/                  Usage examples
scripts/                   Utility scripts
requirements.txt           Python dependencies
```

## Components

- Source-aware action authorization
- Observation quarantine
- Final-answer safeguard
- Guarded executor
- Offline audit utilities
- Online benchmark runner

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Fill in the fields in `.env` or provide them through the environment.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/EVALUATION.md`
- `core/README.md`
- `evaluation/README.md`
