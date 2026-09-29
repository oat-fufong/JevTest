"""Evaluate the Jev classifier against ragsha-agent's golden set.

Usage:
    python -m scripts.jev.evaluate [--summary]
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from scripts import eval_common
from scripts.jev.classifier import classify_via_jev


def classify_fn(query: str) -> dict:
    response = classify_via_jev(query)
    answer = response["answers"]["category"]
    return {
        "predicted": answer["choice"],
        "answer": answer,  # full answer object: type, choice, probabilities, confidence
        "cost": response["usage"]["cost"],
        "usage": response["usage"],  # input_tokens, output_tokens, cost
        "model": response.get("model"),
        "response_id": response.get("id"),
        "provider": response.get("provider"),
    }


if __name__ == "__main__":
    eval_common.run(
        classify_fn,
        title="Jev Classifier — Golden Set Report",
        output_prefix="jev",
        model_env_note=f"Model: {os.environ.get('JEV_MODEL', '~typesafe/jev-latest')}",
    )
