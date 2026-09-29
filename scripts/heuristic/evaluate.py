"""Evaluate ragsha-agent's keyword heuristic against the golden set.

Imports the real _classify_heuristic from the ragsha-agent submodule
(rather than reimplementing its keyword lists), so this measures the
actual production function. No network calls: cost is always $0.

Usage:
    python -m scripts.heuristic.evaluate [--summary]
"""

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_RAGSHA_AGENT = Path(__file__).resolve().parent.parent.parent / "ragsha-agent"
sys.path.insert(0, str(_RAGSHA_AGENT))

from router.classifier import _classify_heuristic  # noqa: E402

from scripts import eval_common  # noqa: E402


def classify_fn(query: str) -> dict:
    predicted, method = _classify_heuristic(query)
    return {"predicted": predicted, "method": method, "cost": 0.0}


if __name__ == "__main__":
    eval_common.run(
        classify_fn,
        title="Heuristic Classifier — Golden Set Report",
        output_prefix="heuristic",
        model_env_note="Model: _classify_heuristic (ragsha-agent/router/classifier.py, keyword-based, no network)",
    )
