"""Research task templates: answer a question from the corpus, with verifiable citations."""

from __future__ import annotations

import re

from agentarena.schemas import Score, Trajectory
from agentarena.tasks.base import OraclePlan, Task
from agentarena.tasks.research.corpus import YEARS, Corpus, normalize
from agentarena.tools import Environment

SYSTEM = (
    "You are a research agent. Answer the user's question using only the document collection, which you "
    "can search and read with tools. Documents can be outdated or contradict each other; prefer the most "
    "recent authoritative statement. When you know the answer, call submit_answer exactly once. The "
    "answer field must be short: just the name, year or number (no units or extra words). Support it with "
    "citations, quoting exact sentences from the documents you used. If the documents do not contain the "
    "answer, submit the answer UNKNOWN."
)

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


class ResearchTask(Task):
    domain = "research"
    system_prompt = SYSTEM
    # Subclasses set: question, answer (str | float), gold (doc ids), evidence (quote per gold doc)
    numeric_tolerance = 0.005  # relative
    forbidden: list[str] = []

    def setup(self) -> None:
        self.corpus = Corpus.generate(self.rng)
        self.cos = list(self.corpus.companies.values())
        self.forbidden = []
        self.configure()
        self.prompt = self.question

    def configure(self) -> None:
        raise NotImplementedError

    def build_env(self) -> Environment:
        return self.corpus.env()  # corpus is read-only, safe to share

    def sentence(self, doc_id: str, must_contain: str) -> str:
        text = self.corpus.docs[doc_id]["text"]
        return next(s for s in re.split(r"(?<=\.)\s+", text) if must_contain in s)

    def answer_correct(self, answer: str) -> bool:
        if isinstance(self.answer, (int, float)) and not isinstance(self.answer, bool):
            # Only the first number counts, so listing several candidates can't game the check.
            nums = [float(x) for x in _NUM.findall(answer.replace(",", ""))][:1]
            return any(abs(n - self.answer) <= max(abs(self.answer) * self.numeric_tolerance, 1e-9) for n in nums)
        # Parenthetical justification ("X ($659M vs Y's $57M)") is not part of the answer itself.
        core = re.sub(r"\([^)]*\)", " ", answer)
        got, gold = normalize(core), normalize(str(self.answer))
        if got == gold:
            return True
        # Allow a short answer containing the gold span, unless it also names a forbidden
        # alternative (hedging like "X or Y" must fail).
        return gold in got and len(got) <= len(gold) + 25 and not any(normalize(f) in got for f in self.forbidden)

    def score(self, env: Environment, traj: Trajectory) -> Score:
        sub = env.submission or {"answer": traj.final_answer, "citations": []}
        answer, cites = str(sub.get("answer", "")), list(sub.get("citations") or [])
        correct = self.answer_correct(answer)

        valid = []
        for c in cites:
            doc = self.corpus.docs.get(str(c.get("doc_id", "")))
            quote = normalize(str(c.get("quote", "")))
            valid.append(bool(doc and quote and len(quote) >= 12 and quote in normalize(doc["text"])))
        cited_ok = {str(c.get("doc_id")) for c, v in zip(cites, valid) if v}
        precision = sum(valid) / len(valid) if valid else (1.0 if not self.gold else 0.0)
        recall = len(cited_ok & set(self.gold)) / len(self.gold) if self.gold else 1.0
        grounded = not self.gold or bool(cited_ok & set(self.gold))
        return Score(
            success=correct and grounded,
            score=0.6 * correct + 0.2 * precision + 0.2 * recall,
            details={
                "answer": answer,
                "expected": self.answer,
                "correct": correct,
                "citation_precision": precision,
                "gold_recall": recall,
                "submitted": env.submission is not None,
            },
        )

    def oracle(self, env: Environment) -> OraclePlan:
        calls = [("search_docs", {"query": self.question})]
        calls += [("read_doc", {"doc_id": d}) for d in self.gold]
        cites = [{"doc_id": d, "quote": q} for d, q in zip(self.gold, self.evidence)]
        calls.append(("submit_answer", {"answer": str(self.answer), "citations": cites}))
        return OraclePlan(calls, str(self.answer))


class FoundingYear(ResearchTask):
    template = "founding_year"

    def configure(self) -> None:
        co = self.rng.choice(self.cos)
        self.question = f"In what year was {co.name} founded?"
        self.answer = co.founded
        self.gold = [self.corpus.gold[co.name]["profile"]]
        self.evidence = [self.sentence(self.gold[0], "founded")]


class LookalikeHQ(ResearchTask):
    """Two companies share a prefix; the agent must not mix them up."""

    template = "lookalike_hq"

    def configure(self) -> None:
        a, b = self.cos[0], self.cos[-1]
        if a.hq == b.hq:
            b.hq = next(c for c in sorted({x.hq for x in self.cos} | {"Eastholm", "Velmora"}) if c != a.hq)
            self.corpus.write_docs()
        co = self.rng.choice([a, b])
        other = b if co is a else a
        self.question = f"In which city is {co.name} headquartered?"
        self.answer = co.hq
        self.forbidden = [other.hq]
        self.gold = [self.corpus.gold[co.name]["profile"]]
        self.evidence = [self.sentence(self.gold[0], "headquartered")]


class AcquirerCEO(ResearchTask):
    """Multi-hop: find the acquirer, then the acquirer's CEO."""

    template = "acquirer_ceo"

    def configure(self) -> None:
        target = self.rng.choice([c for c in self.cos if c.acquired_by])
        acq = self.corpus.companies[target.acquired_by]
        self.question = f"Who is the chief executive officer of the company that acquired {target.name}?"
        self.answer = acq.ceo
        self.gold = [self.corpus.gold[target.name]["acquisition"], self.corpus.gold[acq.name]["profile"]]
        self.evidence = [self.sentence(self.gold[0], "acquisition"), self.sentence(self.gold[1], "chief executive")]


class CompareRevenue(ResearchTask):
    template = "compare_revenue"

    def configure(self) -> None:
        a, b = self.rng.sample(self.cos[:-1], 2)
        year = self.rng.choice(YEARS)
        hi, lo = (a, b) if a.revenue[year] > b.revenue[year] else (b, a)
        self.question = f"Which company reported higher revenue in fiscal {year}: {a.name} or {b.name}?"
        self.answer = hi.name
        self.forbidden = [lo.name]
        self.gold = [self.corpus.gold[a.name]["financials"], self.corpus.gold[b.name]["financials"]]
        self.evidence = [self.sentence(self.gold[0], str(year)), self.sentence(self.gold[1], str(year))]


class TotalRevenue(ResearchTask):
    """Aggregation across three facts in one document."""

    template = "total_revenue"

    def configure(self) -> None:
        co = self.rng.choice(self.cos)
        self.question = (
            f"What was the total revenue of {co.name} across fiscal 2022, 2023 and 2024, in millions of dollars?"
        )
        self.answer = round(sum(co.revenue.values()), 1)
        self.gold = [self.corpus.gold[co.name]["financials"]]
        self.evidence = [self.sentence(self.gold[0], "2022")]


class YearsBetween(ResearchTask):
    template = "years_between_founding"

    def configure(self) -> None:
        a, b = self.rng.sample(self.cos, 2)
        if a.founded == b.founded:
            b.founded += 3
            self.corpus.write_docs()
        old, new = (a, b) if a.founded < b.founded else (b, a)
        self.question = f"How many years after {old.name} was {new.name} founded?"
        self.answer = new.founded - old.founded
        self.gold = [self.corpus.gold[old.name]["profile"], self.corpus.gold[new.name]["profile"]]
        self.evidence = [self.sentence(d, "founded") for d in self.gold]


class CorrectedCEO(ResearchTask):
    """Conflicting sources: the profile is outdated and a later correction supersedes it."""

    template = "corrected_ceo"

    def configure(self) -> None:
        co = next(c for c in self.cos if c.former_ceo)
        self.question = f"Who is the current chief executive officer of {co.name}?"
        self.answer = co.ceo
        self.forbidden = [co.former_ceo]
        self.gold = [self.corpus.gold[co.name]["correction"]]
        self.evidence = [self.sentence(self.gold[0], "current chief executive")]


class Unanswerable(ResearchTask):
    """The corpus has no data for the asked year: the right answer is UNKNOWN."""

    template = "unanswerable"

    def configure(self) -> None:
        co = self.rng.choice(self.cos)
        year = self.rng.choice([2018, 2019, 2020])
        self.question = f"What revenue did {co.name} report in fiscal {year}, in millions of dollars?"
        self.answer = "UNKNOWN"
        self.gold, self.evidence = [], []


TEMPLATES: list[type[ResearchTask]] = [
    FoundingYear,
    LookalikeHQ,
    AcquirerCEO,
    CompareRevenue,
    TotalRevenue,
    YearsBetween,
    CorrectedCEO,
    Unanswerable,
]
