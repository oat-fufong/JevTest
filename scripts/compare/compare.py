"""Build a side-by-side comparison of jev_stats.json and llm_stats.json.

Usage:
    python -m scripts.compare.compare
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def load(name):
    with open(_OUTPUTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def main():
    jev = load("jev_stats.json")
    llm = load("llm_stats.json")

    comparison = {
        "overall_accuracy": {
            "jev": jev["overall"]["accuracy"],
            "llm": llm["overall"]["accuracy"],
            "delta": jev["overall"]["accuracy"] - llm["overall"]["accuracy"],
        },
        "per_category_accuracy": {
            label: {
                "jev": jev["per_category"][label]["accuracy"],
                "llm": llm["per_category"][label]["accuracy"],
                "delta": (
                    jev["per_category"][label]["accuracy"]
                    - llm["per_category"][label]["accuracy"]
                ),
            }
            for label in ("ce", "gk", "chat")
        },
        "avg_latency_ms": {
            "jev": jev["avg_latency_ms"],
            "llm": llm["avg_latency_ms"],
            "jev_vs_llm_speedup": (
                llm["avg_latency_ms"] / jev["avg_latency_ms"]
                if jev["avg_latency_ms"]
                else None
            ),
        },
        "cost": {
            "jev_total_usd": jev["total_cost_usd"],
            "llm_total_usd": llm["total_cost_usd"],
            "jev_avg_usd": jev["avg_cost_usd"],
            "llm_avg_usd": llm["avg_cost_usd"],
            "llm_vs_jev_cost_multiple": (
                llm["total_cost_usd"] / jev["total_cost_usd"]
                if jev["total_cost_usd"]
                else None
            ),
        },
        "misclassified_ids": {
            "jev": sorted(r["id"] for r in jev["misclassified"]),
            "llm": sorted(r["id"] for r in llm["misclassified"]),
            "both_wrong": sorted(
                {r["id"] for r in jev["misclassified"]}
                & {r["id"] for r in llm["misclassified"]}
            ),
            "jev_only_wrong": sorted(
                {r["id"] for r in jev["misclassified"]}
                - {r["id"] for r in llm["misclassified"]}
            ),
            "llm_only_wrong": sorted(
                {r["id"] for r in llm["misclassified"]}
                - {r["id"] for r in jev["misclassified"]}
            ),
        },
    }

    out_path = _OUTPUTS_DIR / "comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print(json.dumps(comparison, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
