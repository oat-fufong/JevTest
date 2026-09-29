"""Evaluate a judge backend (LLM or Jev) on forced conflicts.

The judge only ever runs when the heuristic and router classifier
disagree, so there is no natural "judge on every query" baseline. This
forces a conflict for each sampled query: the real heuristic result,
paired with a deliberately wrong second label, mimicking the shape of
input the judge sees in production. The same conflict set (deterministic
given --n) is used for both backends, so their stats are comparable.

--backend llm (default): the real judge path — _build_judge_prompt from
    the ragsha-agent submodule (so the prompt matches production) sent to
    JUDGE_MODEL (e.g. google/gemini-2.5-pro) via chat completions.
--backend jev: the same conflict framing, answered by Jev's typed Choice
    instead, reusing _JEV_CRITERIA from the same submodule.

Both request real per-call cost via OpenRouter's usage.include, since
_classify_judge / _classify_via_jev in the repo don't request it.

Costs real money per call — defaults to a small sample, not the full
golden set.

Usage:
    python -m scripts.judge.evaluate --backend llm [--n 12] [--summary]
    python -m scripts.judge.evaluate --backend jev [--n 12] [--summary]
"""

import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_RAGSHA_AGENT = Path(__file__).resolve().parent.parent.parent / "ragsha-agent"
sys.path.insert(0, str(_RAGSHA_AGENT))

import requests  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from router.classifier import (  # noqa: E402
    _JEV_CRITERIA,
    _build_judge_prompt,
    _classify_heuristic,
)

from scripts import eval_common  # noqa: E402

load_dotenv(_RAGSHA_AGENT / ".env")

API_KEY = os.environ["LLM_API_KEY"]
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "google/gemini-2.5-pro")
JEV_MODEL = os.environ.get("JEV_MODEL", "typesafe/jev-1.13")
JEV_API_URL = os.environ.get("JEV_API_URL", "https://openrouter.ai/api/alpha/decisions")


def call_judge_llm(query: str, heuristic_result: str, llm_result: str) -> tuple[str, float]:
    prompt = _build_judge_prompt(query, heuristic_result, llm_result, "")
    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
        "usage": {"include": True},
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ragsha-router",
    }
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers, data=json.dumps(payload), timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip().lower()
    predicted = content if content in ("ce", "gk", "chat") else next(
        (cat for cat in ("ce", "gk", "chat") if cat in content), f"UNKNOWN({content[:30]})"
    )
    return predicted, data["usage"]["cost"]


def call_judge_jev(query: str, heuristic_result: str, llm_result: str) -> tuple[str, float]:
    instructions = (
        "Two classifiers disagree about this query's category. "
        f"Heuristic says: {heuristic_result}. LLM says: {llm_result}. "
        "Decide which category is correct, using the definitions below. "
        "You are not limited to the two suggestions if both are wrong."
    )
    response = requests.post(
        url=JEV_API_URL,
        headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        data=json.dumps({
            "model": JEV_MODEL,
            "state": query,
            "questions": {"category": {
                "type": "choice", "instructions": instructions, "criteria": _JEV_CRITERIA,
            }},
        }),
        timeout=20,
    )
    response.raise_for_status()
    data = response.json()
    return data["answers"]["category"]["choice"], data["usage"]["cost"]


def build_synthetic_conflict_set(golden: list, n: int) -> list:
    """Pair each sampled query with its real heuristic result and a
    deliberately wrong alternate label, so the two disagree.

    Biased: takes queries in file order (golden_queries.json happens to
    start with a run of ce queries) and always picks the first category
    in ("ce", "gk", "chat") that isn't the heuristic result as the "wrong"
    side, so whenever the heuristic says ce, forced_llm_result is always
    gk, never chat. Every case here also has the heuristic already right.
    Kept for comparison; prefer build_real_conflict_set.
    """
    conflicts = []
    for entry in golden:
        heuristic_result, _ = _classify_heuristic(entry["query"])
        forced_wrong = next(c for c in ("ce", "gk", "chat") if c != heuristic_result)
        conflicts.append({**entry, "heuristic_result": heuristic_result, "forced_llm_result": forced_wrong})
        if len(conflicts) == n:
            break
    return conflicts


def build_real_conflict_set(golden: list, n: int | None) -> list:
    """Queries where the heuristic and a real LLM router run actually
    disagreed with each other, reusing outputs/llm_outputs.json (an
    existing scripts/llm/evaluate.py run) instead of paying for new LLM
    router calls. Whichever side is "right" varies per case, unlike the
    synthetic set where the heuristic is always right.
    """
    llm_path = Path(__file__).resolve().parent.parent.parent / "outputs" / "llm_outputs.json"
    if not llm_path.exists():
        raise FileNotFoundError(
            f"{llm_path} not found — run `python -m scripts.llm.evaluate` first "
            "so there is a real LLM router result to compare the heuristic against."
        )
    llm_by_id = {r["id"]: r["predicted"] for r in json.loads(llm_path.read_text(encoding="utf-8"))}

    conflicts = []
    for entry in golden:
        heuristic_result, _ = _classify_heuristic(entry["query"])
        llm_result = llm_by_id.get(entry["id"])
        if llm_result is not None and llm_result != heuristic_result:
            conflicts.append({**entry, "heuristic_result": heuristic_result, "forced_llm_result": llm_result})
        if n is not None and len(conflicts) == n:
            break
    return conflicts


def main():
    backend = "llm"
    if "--backend" in sys.argv:
        backend = sys.argv[sys.argv.index("--backend") + 1]
    if backend not in ("llm", "jev"):
        print(f"--backend must be 'llm' or 'jev', got {backend!r}", file=sys.stderr)
        sys.exit(1)

    conflicts_mode = "real"
    if "--conflicts" in sys.argv:
        conflicts_mode = sys.argv[sys.argv.index("--conflicts") + 1]
    if conflicts_mode not in ("real", "synthetic"):
        print(f"--conflicts must be 'real' or 'synthetic', got {conflicts_mode!r}", file=sys.stderr)
        sys.exit(1)

    n = None
    if "--n" in sys.argv:
        n = int(sys.argv[sys.argv.index("--n") + 1])
    elif conflicts_mode == "synthetic":
        n = 12  # unbounded would run all 135; keep the old default sample size
    summary_only = "--summary" in sys.argv

    golden = json.loads((Path(__file__).resolve().parent.parent.parent / "golden_queries.json").read_text(encoding="utf-8"))
    conflicts = (
        build_real_conflict_set(golden, n)
        if conflicts_mode == "real"
        else build_synthetic_conflict_set(golden, n)
    )
    if not conflicts:
        print("No conflicts to judge (0 found).", file=sys.stderr)
        sys.exit(0)
    by_query = {c["query"]: c for c in conflicts}
    call_judge = call_judge_llm if backend == "llm" else call_judge_jev

    def classify_fn(query: str) -> dict:
        c = by_query[query]
        predicted, cost = call_judge(query, c["heuristic_result"], c["forced_llm_result"])
        return {
            "predicted": predicted,
            "cost": cost,
            "heuristic_result": c["heuristic_result"],
            "forced_llm_result": c["forced_llm_result"],
        }

    model_label = JUDGE_MODEL if backend == "llm" else JEV_MODEL
    print(f"Backend: {backend} ({model_label}), judge on {conflicts_mode} conflicts", file=sys.stderr)
    print(f"Queries: {len(conflicts)}", file=sys.stderr)
    print()

    results = eval_common.evaluate(conflicts, classify_fn)
    stats = eval_common.report(results, title=f"Judge ({backend}, {conflicts_mode}) — Conflict Report", full=not summary_only)

    outputs_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    prefix = f"judge_{backend}_{conflicts_mode}"
    (outputs_dir / f"{prefix}_outputs.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (outputs_dir / f"{prefix}_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {prefix}_outputs.json and {prefix}_stats.json", file=sys.stderr)


if __name__ == "__main__":
    main()
