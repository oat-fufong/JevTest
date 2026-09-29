"""Evaluate the LLM classifier (ragsha-agent's current approach) against
the same golden set, for direct comparison with evaluate_jev_classifier.py.

Usage:
    python -m scripts.llm.evaluate [--summary]
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from scripts import eval_common
from scripts.llm.classifier import ROUTER_MODEL, classify_via_llm


def classify_fn(query: str) -> dict:
    response = classify_via_llm(query)
    content = response["choices"][0]["message"]["content"].strip().lower()
    if content in ("ce", "gk", "chat"):
        predicted = content
    else:
        predicted = next(
            (cat for cat in ("ce", "gk", "chat") if cat in content),
            f"UNKNOWN({content[:30]})",
        )
    return {"predicted": predicted, "cost": response["usage"]["cost"]}


if __name__ == "__main__":
    eval_common.run(
        classify_fn,
        title="LLM Classifier — Golden Set Report",
        output_prefix="llm",
        model_env_note=f"Model: {ROUTER_MODEL}",
    )
