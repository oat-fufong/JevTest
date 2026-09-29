"""
Jev-based query classifier for ragsha-agent's router — prototype.

Stands in for router/classifier.py's `_classify_via_llm`: same three
categories (ce = manual RAG, gk = maintenance log RAG, chat), but decided
by Jev's `choice` primitive instead of a chat-completion prompt.
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
JEV_MODEL = os.environ.get("JEV_MODEL", "~typesafe/jev-latest")

_CRITERIA = {
    "ce": (
        "Product manuals, specs, alarm/error/fault codes, how-to procedures, "
        "wiring, troubleshooting, technical definitions ('what is X'), equipment "
        "selection, installation/commissioning guides, testing procedures. Also "
        "general 'how things work' inverter/solar questions and specific product "
        "model behavior (e.g. Sun2000, Huawei, ABB PVS)."
    ),
    "gk": (
        "Project-specific data: maintenance records, failure statistics, project "
        "health trends, KPIs, equipment history, maintenance schedules. Triggered "
        "by a specific project name (e.g. อ่างทอง, CPF22) or a maintenance record "
        "ID (e.g. DA11961), not a general technical question."
    ),
    "chat": (
        "Small talk, greetings, personal questions, non-solar topics (IT, "
        "plumbing, music), jokes, general knowledge unrelated to solar/PV."
    ),
}

_INSTRUCTIONS = (
    "Classify this solar/PV support query into exactly one category. "
    "Alarm/error/fault codes and 'how to fix/check' always go to ce, never gk, "
    "even if a project name is also mentioned incidentally."
)


def classify_via_jev(query: str, history: str = "") -> dict:
    """Classify a query into ce/gk/chat using Jev's choice primitive.

    Returns the full API response: {"answers": {...}, "usage": {...}, ...}.
    """
    state = (
        query
        if not history.strip()
        else f"Conversation history:\n{history}\n\nCurrent query: {query}"
    )

    response = requests.post(
        url="https://openrouter.ai/api/alpha/decisions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps(
            {
                "model": JEV_MODEL,
                "state": state,
                "questions": {
                    "category": {
                        "type": "choice",
                        "instructions": _INSTRUCTIONS,
                        "criteria": _CRITERIA,
                    }
                },
            }
        ),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def classify(query: str, history: str = "") -> str:
    """Convenience wrapper returning just the winning category string."""
    return classify_via_jev(query, history)["answers"]["category"]["choice"]


if __name__ == "__main__":
    for q in [
        "ABB Alarm E032",
        "โครงการ CPF22 DC ศูนย์กระจายสินค้า ล้างแผงล่าสุดวันไหน",
        "tell me a joke",
    ]:
        result = classify_via_jev(q)
        print(f"{q!r} -> {result['answers']['category']} | cost=${result['usage']['cost']}")
