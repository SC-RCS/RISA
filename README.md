# RISA: Runtime Integrity Safeguards for Tool-Using LLM Agents

RISA is a runtime integrity framework for defending tool-using large language model agents against indirect prompt injection. It mediates the agent loop at three runtime boundaries: side-effect authorization, observation incorporation, and response release.

This repository provides the public code and processed-result snapshot supporting the manuscript **RISA: Runtime Integrity Enforcement for Tool-Using Large Language Model Agents**. It includes the core RISA runtime components, policy and experiment configuration snapshots, guard prompt snapshots, processed evaluation results, and sanitized trace examples.

## Repository layout

```text
config.py                         Runtime model/API configuration
llm_client.py                     Shared OpenAI-compatible client wrapper
evaluation/offline/               Offline audit entry point
evaluation/online/                AgentDojo online-evaluation entry point
risa/core/                        Core RISA runtime modules
configs/                          Configuration and manifest snapshots
prompts/                          Guard prompt snapshots
data/aggregate/                   Processed aggregate summaries
data/sample_level/                Processed sample-level outcomes
data/trace_samples/               Sanitized case examples
ARTIFACT_MANIFEST.md              Mapping between manuscript results and artifact files
DATA.md                           Data scope and exclusions
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the offline audit entry point on the included stratified sample:

```bash
python evaluation/offline/run_offline_eval.py --stratified --output results/offline_eval.json
```

Run the AgentDojo online entry point:

```bash
python evaluation/online/run_agentdojo_benchmark.py --mode defense
```

Online runs require an external AgentDojo installation and valid OpenAI-compatible API credentials. Processed result records are provided for inspection in `data/aggregate/` and `data/sample_level/`.

## License

MIT License.

