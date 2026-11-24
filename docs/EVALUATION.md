# Evaluation

The repository contains offline and online evaluation entry points.

## Offline

```bash
python evaluation/offline/run_offline_eval.py --dataset path/to/dataset.json --output results/offline.json
```

## Online

```bash
python evaluation/online/run_agentdojo_benchmark.py --mode defense --suites banking,workspace --attacks important_instructions
```

The online runner requires a compatible AgentDojo installation and API credentials supplied through the environment.
