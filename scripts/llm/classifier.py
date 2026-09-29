"""
LLM-based query classifier — mirrors ragsha-agent's router LLM step
(router/classifier.py::_classify_via_llm / scripts/evaluate_llm_classifier.py)
exactly, so its cost/accuracy is directly comparable to jev_classifier.py.

Same system prompt, same default model (google/gemini-2.5-flash), same
max_tokens=32. Requests OpenRouter's usage.cost so real spend is measured
per call instead of estimated from a pricing table.
"""

import json
import os
import sys

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
ROUTER_MODEL = os.environ.get("ROUTER_MODEL", "google/gemini-2.5-flash")

# Copied verbatim from ragsha-agent/router/classifier.py
_SYSTEM_PROMPT = """You are a query router. Classify each query into exactly one category.

Reply with ONLY the category name (ce, gk, or chat) — nothing else.

Categories:
- "ce": Product manuals, specs, alarm codes, how-to procedures, wiring, troubleshooting, technical definitions, equipment selection, installation guides, testing procedures
- "gk": Project-specific data, maintenance records, failure statistics, project health trends, KPIs, equipment history, specific project names (อ่างทอง, CPF), maintenance schedules
- "chat": Small talk, greetings, personal questions, non-solar topics (IT, plumbing, music), jokes, general knowledge

Rules:
1. Alarm codes, error codes, fault codes, "how to" fix/check → ce
2. Product model specs or behavior (Sun2000, Huawei 50KTL, ABB PVS) → ce
3. Specific PROJECT (e.g., "โครงการอ่างทอง", "CPF22") or maintenance record → gk
4. "What is X" about a technical term → ce (definition, not maintenance log)
5. General inverter/solar "how things work" → ce, not gk

Examples:
"Inverter HUAWEI SUN2000-36KTL-M3 สามารถใช้งานร่วมกับ Smart Logger 1000 ได้หรือไหม" → ce
"inverter sun2000 40ktl code 768 เกิดจากอะไรครับ" → ce
"Inverter sungrow 60ktl alarm code012 แก้ไขอย่างไร" → ce
"ABB Alarm E032" → ce
"แผง360 มีVOCเท่าไหร่" → ce
"ค่า Irradiance คือค่าอะไร" → ce
"สายPV 4 sq.mm ทดแรงดันได้เท่าไหร่" → ce
"คู่มือ Inverter HUAWEI" → ce
"วิธีติดตั้ง และ Setting Smartloger SUN3000" → ce
"Inverter Self Derating คือ" → ce
"การตรวจสอบลงกราวด์ของโซลาร์เซลล์" → ce
"โครงการ CPF22 DC ศูนย์กระจายสินค้า ล้างแผงล่าสุดวันไหน" → gk
"ขอ spec inverter โครงการอ่างทอง 1" → gk
"หาเคส DA11961" → gk
"daily report โหลดข้อมูลกราฟขึ้นErrorเกิดจากสาเหตุใด" → gk
"tell me a joke" → chat
"ไก่กับไข่อะไรเกิดก่อนกัน" → chat
"computer ขึ้น PowerShell เกิดจากอะไร" → chat
"How can i impress my husband?" → chat
"AI จะครองโลก ตามคำทำนายจริงไหม" → chat"""


def classify_via_llm(query: str, history: str = "") -> dict:
    """Classify a query into ce/gk/chat via chat completions.

    Returns the full API response: {"choices": [...], "usage": {...}, ...}.
    """
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history.strip():
        messages.append({"role": "user", "content": history})
    messages.append({"role": "user", "content": query})

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "ragsha-router",
        },
        data=json.dumps(
            {
                "model": ROUTER_MODEL,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 32,
                "usage": {"include": True},
            }
        ),
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def classify(query: str, history: str = "") -> str:
    """Convenience wrapper returning just the winning category string."""
    data = classify_via_llm(query, history)
    content = data["choices"][0]["message"]["content"].strip().lower()
    if content in ("ce", "gk", "chat"):
        return content
    for cat in ("ce", "gk", "chat"):
        if cat in content:
            return cat
    return f"UNKNOWN({content[:30]})"


if __name__ == "__main__":
    for q in [
        "ABB Alarm E032",
        "โครงการ CPF22 DC ศูนย์กระจายสินค้า ล้างแผงล่าสุดวันไหน",
        "tell me a joke",
    ]:
        result = classify_via_llm(q)
        content = result["choices"][0]["message"]["content"]
        print(f"{q!r} -> {content!r} | cost=${result['usage']['cost']}")
