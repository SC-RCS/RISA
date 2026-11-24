"""
Aggregate results into per-model statistics.

Usage:
    python scripts/aggregate_results.py \
        --input results/ \
        --output results/
"""
from __future__ import annotations

import argparse
import json
import glob
import os
from pathlib import Path
from collections import defaultdict


def load_sample_level_results(input_dir: str) -> list[dict]:
    """Load all JSON result files from input directory."""
    pattern = os.path.join(input_dir, "*.json")
    files = glob.glob(pattern)
    results = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            if isinstance(data, dict) and "results" in data:
                agent_model = data.get("agent_model")
                guard_model = data.get("guard_model")
                for row in data["results"]:
                    if isinstance(row, dict):
                        row = dict(row)
                        row.setdefault("agent_model", agent_model)
                        row.setdefault("guard_model", guard_model)
                    results.append(row)
            elif isinstance(data, list):
                results.extend(data)
    return results


def compute_asr(results: list[dict]) -> dict:
    """Compute Attack Success Rate per model."""
    model_stats = defaultdict(lambda: {"total": 0, "success": 0})
    for r in results:
        model = r.get("agent_model", r.get("model_id", "unknown"))
        model_stats[model]["total"] += 1
        if r.get("attack_success", False):
            model_stats[model]["success"] += 1

    asr = {}
    for model, stats in model_stats.items():
        asr[model] = {
            "value": round(stats["success"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
            "success": stats["success"],
            "total": stats["total"],
        }
    return asr


def compute_utility(results: list[dict]) -> dict:
    """Compute utility score per model."""
    model_stats = defaultdict(lambda: {"total": 0, "utility_sum": 0.0})
    for r in results:
        model = r.get("agent_model", r.get("model_id", "unknown"))
        model_stats[model]["total"] += 1
        utility = r.get("utility", r.get("utility_score", 0))
        if isinstance(utility, bool):
            utility = 1.0 if utility else 0.0
        model_stats[model]["utility_sum"] += utility

    utility = {}
    for model, stats in model_stats.items():
        utility[model] = {
            "value": round(stats["utility_sum"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
            "total": stats["total"],
        }
    return utility


def compute_latency(results: list[dict]) -> dict:
    """Compute average latency per model."""
    model_stats = defaultdict(lambda: {"total": 0, "latency_sum": 0.0})
    for r in results:
        model = r.get("agent_model", r.get("model_id", "unknown"))
        model_stats[model]["total"] += 1
        model_stats[model]["latency_sum"] += r.get("elapsed_seconds", r.get("latency_ms", 0.0))

    latency = {}
    for model, stats in model_stats.items():
        latency[model] = {
            "value": round(stats["latency_sum"] / stats["total"], 3) if stats["total"] > 0 else 0.0,
            "total": stats["total"],
        }
    return latency


def compute_suite_breakdown(results: list[dict]) -> dict:
    """Compute per-suite ASR and utility."""
    suite_stats = defaultdict(lambda: {"total": 0, "success": 0, "utility_sum": 0.0})
    for r in results:
        suite = r.get("suite", "unknown")
        suite_stats[suite]["total"] += 1
        if r.get("attack_success", False):
            suite_stats[suite]["success"] += 1
        utility = r.get("utility", r.get("utility_score", 0))
        if isinstance(utility, bool):
            utility = 1.0 if utility else 0.0
        suite_stats[suite]["utility_sum"] += utility

    breakdown = {}
    for suite, stats in suite_stats.items():
        breakdown[suite] = {
            "asr": round(stats["success"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
            "utility": round(stats["utility_sum"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
            "total": stats["total"],
        }
    return breakdown


def main():
    parser = argparse.ArgumentParser(description="Aggregate results into per-model statistics")
    parser.add_argument("--input", required=True, help="Input directory containing result JSON files")
    parser.add_argument("--output", required=True, help="Output directory for aggregate JSON files")
    args = parser.parse_args()

    results = load_sample_level_results(args.input)
    if not results:
        print(f"No results found in {args.input}")
        return

    print(f"Loaded {len(results)} result records")

    aggregate = {
        "sample_count": len(results),
        "asr": compute_asr(results),
        "utility": compute_utility(results),
        "latency": compute_latency(results),
        "suite_breakdown": compute_suite_breakdown(results),
    }

    Path(args.output).mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) / "aggregate_metrics.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2, ensure_ascii=False)

    print(f"Aggregate metrics written to {output_path}")
    print(f"Models: {list(aggregate['asr'].keys())}")
    for model, stats in aggregate["asr"].items():
        print(f"  {model}: ASR={stats['value']:.4f} ({stats['success']}/{stats['total']})")


if __name__ == "__main__":
    main()
