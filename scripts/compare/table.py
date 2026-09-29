"""Print (and save) a markdown comparison table of jev_stats.json vs llm_stats.json.

Usage:
    python -m scripts.compare.table
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_OUTPUTS_DIR = Path(__file__).resolve().parent.parent.parent / "outputs"


def load(name):
    with open(_OUTPUTS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def pct(stats, label=None):
    s = stats["overall"] if label is None else stats["per_category"][label]
    return f"{s['accuracy']:.1%} ({s['correct']}/{s['total']})"


def main():
    jev = load("jev_stats.json")
    llm = load("llm_stats.json")

    speedup = llm["avg_latency_ms"] / jev["avg_latency_ms"]
    cost_multiple = llm["total_cost_usd"] / jev["total_cost_usd"]

    rows = [
        ("Overall accuracy", pct(jev), pct(llm)),
        (
            "ce / gk / chat",
            " / ".join(f"{jev['per_category'][k]['accuracy']:.1%}" for k in ("ce", "gk", "chat")),
            " / ".join(f"{llm['per_category'][k]['accuracy']:.1%}" for k in ("ce", "gk", "chat")),
        ),
        ("Avg latency", f"{jev['avg_latency_ms']:.0f}ms", f"{llm['avg_latency_ms']:.0f}ms"),
        ("Total cost", f"${jev['total_cost_usd']:.6f}", f"${llm['total_cost_usd']:.6f}"),
        ("Cost per query", f"${jev['avg_cost_usd']:.6f}", f"${llm['avg_cost_usd']:.6f}"),
    ]

    lines = [
        "| | Jev | LLM (gemini-2.5-flash) |",
        "|---|---|---|",
        *(f"| {name} | {j} | {l} |" for name, j, l in rows),
        "",
        f"Jev is {cost_multiple:.1f}x cheaper and {speedup:.2f}x faster per call.",
    ]
    table = "\n".join(lines)

    print(table)
    out_path = _OUTPUTS_DIR / "comparison_table.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
