"""
Jev exploration via OpenRouter's alpha decisions endpoint.

Fill in .env (copy from .env.example) before running:
  OPENROUTER_API_KEY from https://openrouter.ai/keys
"""

import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
JEV_MODEL = os.environ.get("JEV_MODEL", "~typesafe/jev-latest")


def ask(state: str, questions: dict) -> dict:
    response = requests.post(
        url="https://openrouter.ai/api/alpha/decisions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        data=json.dumps({"model": JEV_MODEL, "state": state, "questions": questions}),
    )
    response.raise_for_status()
    return response.json()["answers"]


def main():
    # One state, all three primitive types at once.
    answers = ask(
        state="What is the statute of limitations for breach of contract?",
        questions={
            "domain": {
                "type": "choice",
                "instructions": "Which area does this question fall under?",
                "criteria": {
                    "civil": "Civil law matters",
                    "criminal": "Criminal law matters",
                    "administrative": "Administrative/regulatory matters",
                    "unclear": "Not enough information to tell",
                },
            },
            "clarity": {
                "type": "score",
                "instructions": "How clearly specified is the question?",
                "criteria": ["vague", "somewhat clear", "clear", "very precise"],
            },
            "is_urgent": {
                "type": "noul",
                "instructions": "Does this message convey urgency?",
                "criteria": {
                    "true": "Explicitly time-sensitive",
                    "false": "No urgency expressed",
                },
            },
        },
    )
    print(json.dumps(answers, indent=2))


if __name__ == "__main__":
    main()
