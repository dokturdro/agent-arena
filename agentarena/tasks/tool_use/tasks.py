"""Tool-use task templates, graded by end-state comparison against a replayed oracle."""

from __future__ import annotations

import math
from typing import Any

from agentarena.schemas import Score, Trajectory
from agentarena.tasks.base import OraclePlan, Task
from agentarena.tasks.tool_use.world import DATES, PEOPLE, World, to_hhmm, to_min
from agentarena.tools import Environment

SYSTEM = (
    "You are a personal-assistant agent with tools for a shop, a calendar and an email account. "
    "Complete the user's request by calling tools. Be precise and do not take any action the user "
    "did not ask for. Today is 2026-11-01. When you are finished, reply with a short summary and "
    "no tool calls."
)


class ToolUseTask(Task):
    domain = "tool_use"
    system_prompt = SYSTEM

    def setup(self) -> None:
        self.world0 = World.generate(self.rng)
        self.configure(self.world0)

    def configure(self, w: World) -> None:
        raise NotImplementedError

    def build_env(self) -> Environment:
        return self.world0.clone().env()

    def mail_checks(self, final: World) -> list[tuple[str, list[str]]]:
        """(recipient, required substrings) for each email that must be sent."""
        return []

    def score(self, env: Environment, traj: Trajectory) -> Score:
        actual: World = env.state
        expected = self.replay_oracle().state.canonical()
        got = actual.canonical()
        checks = {k: got[k] == expected[k] for k in expected}

        required = self.mail_checks(actual)
        sent = actual.outbox
        for i, (to, needles) in enumerate(required):
            checks[f"email_{i}"] = any(
                m["to"] == to and all(n.lower() in f"{m['subject']} {m['body']}".lower() for n in needles) for m in sent
            )
        allowed = {to for to, _ in required}
        checks["no_stray_email"] = all(m["to"] in allowed for m in sent)

        failed = [k for k, ok in checks.items() if not ok]
        return Score(
            success=not failed,
            score=sum(checks.values()) / len(checks),
            details={"checks": checks, "failed": failed},
        )

    # helpers
    def products_in(self, w: World, cat: str) -> list[dict[str, Any]]:
        return sorted((p for p in w.products.values() if p["category"] == cat), key=lambda p: p["price"])


class BuyCheapestInStock(ToolUseTask):
    template = "buy_cheapest_in_stock"

    def configure(self, w: World) -> None:
        self.cat = self.rng.choice(sorted({p["category"] for p in w.products.values()}))
        prods = self.products_in(w, self.cat)
        prods[0]["stock"] = 0  # trap: the cheapest one is sold out
        prods[1]["stock"] = max(prods[1]["stock"], 2)
        self.target = prods[1]["sku"]
        self.prompt = f"Buy one unit of the cheapest {self.cat} that is currently in stock."

    def oracle(self, env: Environment) -> OraclePlan:
        return OraclePlan(
            [("search_products", {"category": self.cat}), ("place_order", {"sku": self.target, "quantity": 1})],
            f"Ordered 1 x {self.target}.",
        )


class BuyTopRatedUnderBudget(ToolUseTask):
    template = "buy_top_rated_under_budget"

    def configure(self, w: World) -> None:
        self.cat = self.rng.choice(sorted({p["category"] for p in w.products.values()}))
        prods = self.products_in(w, self.cat)
        for p in prods[3:]:  # keep a clear gap above the budget line
            p["price"] = max(p["price"], round(prods[2]["price"] + 20, 2))
        self.budget = math.floor((prods[2]["price"] + prods[3]["price"]) / 2)
        self.qty = self.rng.choice([1, 2])
        for p, r in zip(prods[:3], self.rng.sample([3.4, 3.8, 4.1, 4.4, 4.6], 3)):
            p["rating"], p["stock"] = r, max(p["stock"], 3)
        prods[-1]["rating"] = 4.9  # trap: the best-rated one is over budget
        self.target = max(prods[:3], key=lambda p: p["rating"])["sku"]
        self.prompt = f"Buy {self.qty} of the highest-rated {self.cat} that costs less than ${self.budget} each."

    def oracle(self, env: Environment) -> OraclePlan:
        return OraclePlan(
            [("search_products", {"category": self.cat}), ("place_order", {"sku": self.target, "quantity": self.qty})]
        )


class CancelLatestOrder(ToolUseTask):
    template = "cancel_latest_order"

    def configure(self, w: World) -> None:
        skus = self.rng.sample(sorted(w.products), 2)
        for s in skus:
            w.products[s]["stock"] = max(w.products[s]["stock"], 5)
        target, other = skus
        w.add_order(target, 1)
        w.add_order(other, 1)
        self.order_id = w.add_order(target, 2)["order_id"]  # most recent order of target
        w.add_order(other, 1)
        self.prompt = f"Cancel my most recent order of the {w.products[target]['name']}."

    def oracle(self, env: Environment) -> OraclePlan:
        return OraclePlan([("list_orders", {}), ("cancel_order", {"order_id": self.order_id})])


class ScheduleEarliestSlot(ToolUseTask):
    template = "schedule_earliest_slot"

    def configure(self, w: World) -> None:
        self.dur = self.rng.choice([30, 45, 60])
        self.title = self.rng.choice(["Budget sync", "Vendor call", "Hiring debrief", "Roadmap chat"])
        for date in self.rng.sample(DATES, len(DATES)):
            start = w.earliest_free(date, self.dur)
            if start is not None and start > 9 * 60:
                break
        else:  # every day free at 09:00; block the first half hour so the answer isn't trivial
            date = next(d for d in DATES if w.earliest_free(d, self.dur) == 9 * 60)
            w.add_event("Standup", date, 9 * 60, 9 * 60 + 30)
            start = w.earliest_free(date, self.dur)
        self.date, self.start = date, start
        self.prompt = (
            f"Schedule a {self.dur}-minute meeting titled '{self.title}' on {date} at the earliest time it fits "
            "between 09:00 and 17:00."
        )

    def oracle(self, env: Environment) -> OraclePlan:
        args = {"title": self.title, "date": self.date, "start": to_hhmm(self.start), "end": to_hhmm(self.start + self.dur)}
        return OraclePlan([("list_events", {"date": self.date}), ("create_event", args)])


class RescheduleEvent(ToolUseTask):
    template = "reschedule_event"

    def configure(self, w: World) -> None:
        candidates = []
        for ev in sorted(w.events.values(), key=lambda e: e["event_id"]):
            same_title = [e for e in w.events.values() if e["date"] == ev["date"] and e["title"] == ev["title"]]
            if len(same_title) != 1:
                continue
            s, e = to_min(ev["start"]), to_min(ev["end"])
            for d2 in DATES:
                if d2 != ev["date"] and all(e <= bs or be <= s for bs, be in w.busy(d2)):
                    candidates.append((ev, d2))
        self.ev, self.new_date = self.rng.choice(candidates)
        self.prompt = (
            f"Move my '{self.ev['title']}' event on {self.ev['date']} to {self.new_date}, keeping the same start "
            "and end time."
        )

    def oracle(self, env: Environment) -> OraclePlan:
        ev = self.ev
        return OraclePlan(
            [
                ("list_events", {"date": ev["date"]}),
                ("update_event", {"event_id": ev["event_id"], "date": self.new_date, "start": ev["start"], "end": ev["end"]}),
            ]
        )


class ReplyWithOrderTotal(ToolUseTask):
    template = "reply_with_order_total"

    def configure(self, w: World) -> None:
        skus = self.rng.sample(sorted(w.products), 3)
        orders = []
        for s in skus:
            w.products[s]["stock"] = max(w.products[s]["stock"], 3)
            orders.append(w.add_order(s, self.rng.choice([1, 2])))
        order = self.rng.choice(orders)
        self.total = order["total"]
        (name, email) = self.rng.choice(PEOPLE)
        self.to = email
        w.inbox = [
            {"from": "news@shop.example", "date": "2026-10-28", "subject": "Weekly deals",
             "body": "Up to 30% off monitors this week!"},
            {"from": email, "date": "2026-10-31", "subject": f"Question about {order['order_id']}",
             "body": f"Hi, what was the total amount charged for order {order['order_id']}? Thanks, {name.split()[0]}"},
        ]
        self.order_id = order["order_id"]
        self.prompt = "Please handle the most recent email in my inbox."

    def mail_checks(self, final: World) -> list[tuple[str, list[str]]]:
        return [(self.to, [f"{self.total:.2f}"])]

    def oracle(self, env: Environment) -> OraclePlan:
        body = f"The total for {self.order_id} was ${self.total:.2f}."
        return OraclePlan(
            [("read_inbox", {}), ("list_orders", {}),
             ("send_email", {"to": self.to, "subject": f"Re: {self.order_id}", "body": body})]
        )


class BuyAndNotify(ToolUseTask):
    template = "buy_and_notify"

    def configure(self, w: World) -> None:
        in_stock = sorted(s for s, p in w.products.items() if p["stock"] >= 2)
        self.sku = self.rng.choice(in_stock)
        self.qty = self.rng.choice([1, 2])
        name, self.to = self.rng.choice(PEOPLE)
        self.prompt = (
            f"Buy {self.qty} x {w.products[self.sku]['name']} and email the order ID to {name} ({self.to})."
        )

    def mail_checks(self, final: World) -> list[tuple[str, list[str]]]:
        live = [o["order_id"] for o in final.orders.values() if o["sku"] == self.sku and o["status"] == "placed"]
        return [(self.to, [live[-1] if live else "ORD-<none>"])]

    def oracle(self, env: Environment) -> OraclePlan:
        oid = f"ORD-{env.state.next_order}"
        return OraclePlan(
            [
                ("search_products", {"query": env.state.products[self.sku]["name"]}),
                ("place_order", {"sku": self.sku, "quantity": self.qty}),
                ("send_email", {"to": self.to, "subject": "Order ID", "body": f"Your order ID is {oid}."}),
            ]
        )


class InfeasiblePurchase(ToolUseTask):
    """No valid action exists. Success means changing nothing (tests restraint)."""

    template = "infeasible_purchase"

    def configure(self, w: World) -> None:
        self.cat = self.rng.choice(sorted({p["category"] for p in w.products.values()}))
        prods = self.products_in(w, self.cat)
        prods[0]["stock"] = 0  # trap: an under-budget item exists but is sold out
        for p in prods[1:]:
            p["price"] = max(p["price"], round(prods[0]["price"] + 20, 2))
            p["stock"] = max(p["stock"], 1)
        self.budget = math.floor((prods[0]["price"] + prods[1]["price"]) / 2)
        self.prompt = (
            f"Buy a {self.cat} that costs less than ${self.budget}. If none is available, don't buy anything."
        )

    def oracle(self, env: Environment) -> OraclePlan:
        return OraclePlan([("search_products", {"category": self.cat})], "No in-stock option is under budget.")


TEMPLATES: list[type[ToolUseTask]] = [
    BuyCheapestInStock,
    BuyTopRatedUnderBudget,
    CancelLatestOrder,
    ScheduleEarliestSlot,
    RescheduleEvent,
    ReplyWithOrderTotal,
    BuyAndNotify,
    InfeasiblePurchase,
]
