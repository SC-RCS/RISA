# Evaluation

The repository exposes two main experiment entry points.

## Offline evaluation

`evaluation/offline/run_offline_eval.py`

This script is the released offline audit entry point for trajectory-style evaluation.

Example:

```bash
python evaluation/offline/run_offline_eval.py --dataset path/to/dataset.json --output results/offline.json
```

## Online evaluation

`evaluation/online/run_agentdojo_benchmark.py`

This script is the released online benchmark entry point for AgentDojo-style evaluation.

Example:

```bash
python evaluation/online/run_agentdojo_benchmark.py --mode defense --suites banking,workspace
```

## Notes

- Full reruns may require external benchmark assets and separate environment setup.
- The public repository is structured to expose the framework and experiment entry points even when all original private runtime dependencies are not bundled here.
