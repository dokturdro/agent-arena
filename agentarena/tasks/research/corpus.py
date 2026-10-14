"""Synthetic research corpus about fictional companies, plus BM25 search tools.

The facts are invented per seed, so a model cannot answer from memorised world
knowledge and has to find the evidence. The corpus deliberately contains
look-alike company names, superseded facts and an explicit correction notice.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentarena.tools import Environment, Tool, ToolError, obj

PREFIXES = ["Veltrix", "Orbana", "Quillon", "Sarnex", "Tavera", "Lumora", "Kestrel", "Vantor", "Mirelo", "Zephra",
            "Corvid", "Halcyra"]
SUFFIXES = ["Dynamics", "Systems", "Labs", "Holdings", "Robotics", "Analytics", "Foods", "Energy"]
CITIES = ["Port Aldine", "New Carrow", "Eastholm", "Brightwater", "Velmora", "Lindqvist Bay", "Ostrava Falls"]
FIRST = ["Ines", "Tomas", "Priya", "Kofi", "Mara", "Elias", "Yuki", "Sven", "Leila", "Ravi", "Ana", "Jonah"]
LAST = ["Halvorsen", "Achebe", "Moreno", "Tanaka", "Okafor", "Lindgren", "Novak", "Castell", "Iyer", "Brandt"]
YEARS = [2022, 2023, 2024]

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


@dataclass
class Company:
    name: str
    founded: int
    hq: str
    ceo: str
    revenue: dict[int, float]  # $M
    former_ceo: str | None = None  # an older article still names this person
    acquired_by: str | None = None


@dataclass
class Corpus:
    companies: dict[str, Company] = field(default_factory=dict)
    docs: dict[str, dict[str, str]] = field(default_factory=dict)  # id -> {title, date, text}

    @classmethod
    def generate(cls, rng: random.Random) -> Corpus:
        c = cls()
        prefixes = rng.sample(PREFIXES, 7)
        people = [f"{f} {l}" for f in FIRST for l in LAST]
        rng.shuffle(people)
        names = [f"{p} {rng.choice(SUFFIXES)}" for p in prefixes]
        # Look-alike: same prefix, different suffix, different facts.
        lookalike_src = names[0].split()[0]
        other_suffix = rng.choice([s for s in SUFFIXES if s != names[0].split()[1]])
        names.append(f"{lookalike_src} {other_suffix}")
        for name in names:
            c.companies[name] = Company(
                name=name,
                founded=rng.randint(1975, 2016),
                hq=rng.choice(CITIES),
                ceo=people.pop(),
                revenue={y: round(rng.uniform(40, 900), 1) for y in YEARS},
            )
        cos = list(c.companies.values())
        cos[1].acquired_by = cos[2].name
        cos[3].acquired_by = cos[4].name
        cos[5].former_ceo, cos[5].ceo = cos[5].ceo, people.pop()  # superseded by a correction
        c._write_docs()
        return c

    def _write_docs(self) -> None:
        n = 0

        def add(title: str, date: str, text: str) -> str:
            nonlocal n
            n += 1
            doc_id = f"doc-{n:03d}"
            self.docs[doc_id] = {"title": title, "date": date, "text": text}
            return doc_id

        self.gold: dict[str, dict[str, str]] = {}
        for co in self.companies.values():
            g = self.gold.setdefault(co.name, {})
            g["profile"] = add(
                f"{co.name}: company profile",
                "2025-01-15",
                f"{co.name} was founded in {co.founded} and is headquartered in {co.hq}. "
                + (f"The company is led by chief executive {co.former_ceo}. " if co.former_ceo
                   else f"Its chief executive officer is {co.ceo}. ")
                + f"{co.name} operates in several regional markets.",
            )
            g["financials"] = add(
                f"{co.name} annual results",
                "2025-03-02",
                " ".join(f"In fiscal {y}, {co.name} reported revenue of ${co.revenue[y]} million." for y in YEARS),
            )
            if co.former_ceo:
                g["correction"] = add(
                    f"Correction: leadership at {co.name}",
                    "2025-06-20",
                    f"Correction: an earlier profile misstated who leads {co.name}. {co.former_ceo} stepped down in "
                    f"2024; the current chief executive officer of {co.name} is {co.ceo}.",
                )
            if co.acquired_by:
                g["acquisition"] = add(
                    f"{co.acquired_by} completes acquisition",
                    "2025-02-10",
                    f"{co.acquired_by} announced that it has completed its acquisition of {co.name}. "
                    f"{co.name} will continue to operate under its own brand.",
                )
        add("Industry outlook 2025", "2025-01-05",
            "Analysts expect consolidation across the sector, with several mid-sized firms changing hands. "
            "Revenue growth is projected to slow in most regional markets.")

    # ---- search -----------------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """Okapi BM25 over title + text."""
        q = tokenize(query)
        if not q:
            return []
        docs = {i: tokenize(d["title"] + " " + d["text"]) for i, d in self.docs.items()}
        avgdl = sum(map(len, docs.values())) / len(docs)
        df = Counter(t for toks in docs.values() for t in set(toks))
        N, k1, b = len(docs), 1.5, 0.75
        scores = {}
        for i, toks in docs.items():
            tf = Counter(toks)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(toks) / avgdl))
            if s > 0:
                scores[i] = s
        top = sorted(scores, key=lambda i: (-scores[i], i))[:k]
        return [{"doc_id": i, "title": self.docs[i]["title"], "date": self.docs[i]["date"],
                 "snippet": self.docs[i]["text"][:120]} for i in top]

    def tools(self, env: Environment) -> list[Tool]:
        def search_docs(query: str) -> list[dict[str, Any]]:
            return self.search(query)

        def read_doc(doc_id: str) -> dict[str, str]:
            if doc_id not in self.docs:
                raise ToolError(f"no document {doc_id}")
            return {"doc_id": doc_id} | self.docs[doc_id]

        def submit_answer(answer: str, citations: list[dict[str, str]]) -> str:
            env.submission = {"answer": answer, "citations": citations}
            return "Answer submitted."

        s = {"type": "string"}
        citation = obj({"doc_id": s, "quote": {"type": "string", "description": "Exact sentence copied from the doc"}})
        return [
            Tool("search_docs", "Keyword search over the document collection. Returns the top 5 hits.",
                 obj({"query": s}), search_docs),
            Tool("read_doc", "Read the full text of a document.", obj({"doc_id": s}), read_doc),
            Tool("submit_answer", "Submit your final answer with supporting citations. Ends the task.",
                 obj({"answer": s, "citations": {"type": "array", "items": citation}}), submit_answer),
        ]

    def env(self) -> Environment:
        env = Environment(tools=[], state=self)
        env.tools = self.tools(env)
        env.__post_init__()
        return env
