"""Print (and save) a markdown comparison table of the LLM judge vs Jev-as-judge.

Usage:
    python -m scripts.compare.judge_table [--conflicts real|synthetic]
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
    conflicts_mode = "real"
    if "--conflicts" in sys.argv:
        conflicts_mode = sys.argv[sys.argv.index("--conflicts") + 1]

    jev = load(f"judge_jev_{conflicts_mode}_stats.json")
    llm = load(f"judge_llm_{conflicts_mode}_stats.json")

    speedup = llm["avg_latency_ms"] / jev["avg_latency_ms"]
    cost_multiple = llm["total_cost_usd"] / jev["total_cost_usd"]

    def pct(stats):
        o = stats["overall"]
        return f"{o['accuracy']:.1%} ({o['correct']}/{o['total']})"

    rows = [
        ("Overall accuracy", pct(jev), pct(llm)),
        ("Avg latency", f"{jev['avg_latency_ms']:.0f}ms", f"{llm['avg_latency_ms']:.0f}ms"),
        ("Total cost", f"${jev['total_cost_usd']:.6f}", f"${llm['total_cost_usd']:.6f}"),
        ("Cost per query", f"${jev['avg_cost_usd']:.6f}", f"${llm['avg_cost_usd']:.6f}"),
    ]

    lines = [
        f"| ({conflicts_mode} conflicts) | Jev (as judge) | LLM judge (gemini-2.5-pro) |",
        "|---|---|---|",
        *(f"| {name} | {j} | {l} |" for name, j, l in rows),
        "",
        f"Jev is {cost_multiple:.1f}x cheaper and {speedup:.2f}x faster per call, on the same {jev['overall']['total']}-query {conflicts_mode}-conflict set.",
    ]
    table = "\n".join(lines)

    print(table)
    out_path = _OUTPUTS_DIR / f"judge_comparison_table_{conflicts_mode}.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
