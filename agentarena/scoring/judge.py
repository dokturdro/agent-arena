"""LLM-as-judge, with the standard hygiene:

* rubric grading against a reference answer (not open-ended "is this good?")
* pairwise comparisons run twice with positions swapped; inconsistent verdicts
  become ties, which cancels position bias
* the judge must come from a different model family than the contestants it grades
* agreement with human labels is measured (Cohen's kappa) before trusting it
"""

from __future__ import annotations

import json
import re
from typing import Any

from agentarena.agents.base import Model
from agentarena.schemas import Message

RUBRIC = """You are grading an answer to a research question against a reference answer.

Question: {question}
Reference answer: {reference}
Candidate answer: {answer}
Candidate citations: {citations}

Score 1-5:
5 = correct and fully supported by relevant quotes
4 = correct, citations weak or partly irrelevant
3 = partially correct or hedged between options
2 = incorrect but on topic
1 = incorrect, missing, or unsupported claims
"correct" is true only if the candidate's answer matches the reference in substance
(ignore formatting; a reference of UNKNOWN means the candidate must say the information is unavailable).

Reply with JSON only: {{"score": <1-5>, "correct": <true|false>, "rationale": "<one sentence>"}}"""

PAIRWISE = """Two agents answered the same research question. Decide which answer is better: more
correct first, then better supported by its citations.

Question: {question}
Reference answer: {reference}

[Answer 1]
{a}

[Answer 2]
{b}

Reply with JSON only: {{"winner": "1" | "2" | "tie", "rationale": "<one sentence>"}}"""


def _parse_json(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"judge returned no JSON: {text[:200]!r}")
    return json.loads(m.group(0))


async def _ask(judge: Model, prompt: str) -> dict[str, Any]:
    comp = await judge.complete([Message(role="user", content=prompt)], [], max_tokens=1024)
    return _parse_json(comp.text)


def format_answer(answer: str, citations: list[dict[str, Any]]) -> str:
    cites = "\n".join(f'- {c.get("doc_id")}: "{c.get("quote")}"' for c in citations) or "- (none)"
    return f"Answer: {answer}\nCitations:\n{cites}"


async def rubric_judge(judge: Model, question: str, reference: str, answer: str,
                       citations: list[dict[str, Any]]) -> dict[str, Any]:
    out = await _ask(judge, RUBRIC.format(question=question, reference=reference, answer=answer,
                                          citations=json.dumps(citations)))
    return {"score": int(out["score"]), "correct": bool(out["correct"]), "rationale": out.get("rationale", "")}


async def pairwise_judge(judge: Model, question: str, reference: str, a: str, b: str) -> float:
    """Returns 1 if a wins, 0 if b wins, 0.5 for a tie or an inconsistent verdict."""
    first = (await _ask(judge, PAIRWISE.format(question=question, reference=reference, a=a, b=b)))["winner"]
    second = (await _ask(judge, PAIRWISE.format(question=question, reference=reference, a=b, b=a)))["winner"]
    v1 = {"1": 1.0, "2": 0.0}.get(str(first), 0.5)
    v2 = {"1": 0.0, "2": 1.0}.get(str(second), 0.5)  # swapped back to a's perspective
    return v1 if v1 == v2 else 0.5


def cohen_kappa(y1: list[bool], y2: list[bool]) -> float:
    n = len(y1)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(y1, y2)) / n
    p1, p2 = sum(y1) / n, sum(y2) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
