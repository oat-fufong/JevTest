"""Shared golden-set evaluation harness for classify_fn-style callables.

Used by scripts/jev/evaluate.py and scripts/llm/evaluate.py so both
report identical stats (accuracy, confusion matrix, latency, cost) and can
be compared directly.
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_GOLDEN_PATH = _ROOT / "golden_queries.json"
_OUTPUTS_DIR = _ROOT / "outputs"

with open(_GOLDEN_PATH, encoding="utf-8") as f:
    GOLDEN = json.load(f)


def evaluate(queries, classify_fn):
    """classify_fn(query) -> dict with at least {"predicted", "cost"};
    any extra keys (e.g. "probabilities") are carried through to results."""
    results = []
    total = len(queries)
    for i, entry in enumerate(queries, 1):
        query = entry["query"]
        label = entry["label"]
        start = time.monotonic()
        try:
            outcome = classify_fn(query)
        except Exception as e:
            outcome = {"predicted": f"ERROR({str(e)[:30]})", "cost": 0.0}
        elapsed_ms = (time.monotonic() - start) * 1000
        results.append(
            {
                **entry,  # full entry: id, query (untruncated), label, note, ...
                "correct": outcome["predicted"] == label,
                "elapsed_ms": elapsed_ms,
                **outcome,
            }
        )
        if i % 25 == 0 or i == total:
            print(f"  ... {i}/{total} queries classified", file=sys.stderr)
    return results


def build_stats(results):
    correct = Counter()
    total = Counter()
    confusion = defaultdict(int)

    for r in results:
        label = r["label"]
        predicted = r["predicted"]
        total[label] += 1
        confusion[(label, predicted)] += 1
        if predicted == label:
            correct[label] += 1

    overall_correct = sum(correct.values())
    overall_total = sum(total.values())
    overall = overall_correct / overall_total if overall_total else 0
    avg_ms = sum(r["elapsed_ms"] for r in results) / len(results) if results else 0
    total_cost = sum(r.get("cost", 0.0) for r in results)

    return {
        "per_category": {
            label: {
                "correct": correct[label],
                "total": total[label],
                "accuracy": (correct[label] / total[label]) if total[label] else 0,
            }
            for label in ("ce", "gk", "chat")
        },
        "overall": {
            "correct": overall_correct,
            "total": overall_total,
            "accuracy": overall,
        },
        "avg_latency_ms": avg_ms,
        "total_cost_usd": total_cost,
        "avg_cost_usd": total_cost / len(results) if results else 0,
        "confusion_matrix": {
            f"{actual}->{predicted}": count
            for (actual, predicted), count in sorted(confusion.items())
            if count > 0
        },
        "misclassified": [
            {k: r[k] for k in r if k not in ("elapsed_ms", "correct")}
            for r in results
            if not r["correct"]
        ],
    }


def report(results, title, full=True):
    stats = build_stats(results)
    correct = {k: v["correct"] for k, v in stats["per_category"].items()}
    total = {k: v["total"] for k, v in stats["per_category"].items()}
    overall = stats["overall"]["accuracy"]

    lines = ["=" * 60, title, "=" * 60]
    for label in ("ce", "gk", "chat"):
        n, c = total[label], correct[label]
        rate = c / n if n else 0
        lines.append(f"  {label:4s}: {c:3d}/{n:3d} ({rate:.1%})")

    lines.append(
        f"  Overall: {sum(correct.values()):3d}/"
        f"{sum(total.values()):3d} ({overall:.1%})"
    )
    lines.append(f"  Avg latency: {stats['avg_latency_ms']:.0f}ms")
    lines.append(
        f"  Total cost: ${stats['total_cost_usd']:.6f} "
        f"(${stats['avg_cost_usd']:.6f}/query)"
    )
    lines.append("")
    lines.append("Confusion matrix (actual -> predicted):")
    for key, count in stats["confusion_matrix"].items():
        actual, predicted = key.split("->")
        marker = " OK" if actual == predicted else ""
        lines.append(f"  {actual:4s} -> {predicted:4s}: {count:3d}{marker}")
    lines.append("=" * 60)

    print("\n".join(lines))

    if full:
        misclassified = stats["misclassified"]
        if misclassified:
            print(f"\nMisclassified queries ({len(misclassified)} total):")
            for r in misclassified:
                extra = {k: v for k, v in r.items() if k not in ("id", "label", "predicted", "query")}
                print(f"  #{r['id']:>3} {r['label']:4s}->{r['predicted']:4s} {extra} | {r['query']}")
        else:
            print("\nAll queries classified correctly!")

    return stats


def run(classify_fn, title, output_prefix, model_env_note=""):
    """Full pipeline: evaluate, report, write outputs/{prefix}_{outputs,stats}.json."""
    summary_only = "--summary" in sys.argv
    print(model_env_note, file=sys.stderr)
    print(f"Queries: {len(GOLDEN)}", file=sys.stderr)
    print()

    results = evaluate(GOLDEN, classify_fn)
    stats = report(results, title, full=not summary_only)

    _OUTPUTS_DIR.mkdir(exist_ok=True)
    outputs_path = _OUTPUTS_DIR / f"{output_prefix}_outputs.json"
    stats_path = _OUTPUTS_DIR / f"{output_prefix}_stats.json"

    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {outputs_path.name} and {stats_path.name}", file=sys.stderr)
