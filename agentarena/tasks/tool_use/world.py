"""In-process mock "personal assistant" world: a shop, a calendar and a mailbox.

All state lives in one ``World`` dataclass so graders can diff it. Everything is
generated from a seeded RNG, so a task instance is reproducible but each seed
produces different products, prices and events (limits memorisation).
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from agentarena.tools import Environment, Tool, ToolError, obj

CATEGORIES = ["headphones", "keyboard", "monitor", "backpack", "coffee maker"]
BRANDS = ["Aurex", "Brindle", "Corvo", "Dunmore", "Elvar", "Fennik", "Glowe", "Harlan", "Istra", "Juno"]
SUFFIXES = ["One", "Pro", "Air", "Max", "Lite", "S2", "X", "Neo"]
PEOPLE = [
    ("Alice Moreau", "alice@example.com"),
    ("Bob Lindqvist", "bob@example.com"),
    ("Chen Wei", "chen@example.com"),
    ("Dana Okafor", "dana@example.com"),
]
DATES = ["2026-11-02", "2026-11-03", "2026-11-04", "2026-11-05", "2026-11-06"]


def to_min(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


@dataclass
class World:
    products: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    balance: float = 5000.0
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    inbox: list[dict[str, Any]] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)
    next_order: int = 1001
    next_event: int = 1

    # ---- generation -------------------------------------------------------
    @classmethod
    def generate(cls, rng: random.Random) -> World:
        w = cls()
        sku_n = 100
        names: set[str] = set()
        for cat in CATEGORIES:
            for _ in range(5):
                sku_n += rng.randint(1, 9)
                name = f"{rng.choice(BRANDS)} {cat.title()} {rng.choice(SUFFIXES)}"
                while name in names:
                    name = f"{rng.choice(BRANDS)} {cat.title()} {rng.choice(SUFFIXES)}"
                names.add(name)
                w.products[f"SKU-{sku_n}"] = {
                    "sku": f"SKU-{sku_n}",
                    "name": name,
                    "category": cat,
                    "price": round(rng.uniform(25, 400), 2),
                    "stock": rng.choice([0, 2, 3, 5, 8, 12]),
                    "rating": round(rng.uniform(3.0, 5.0), 1),
                }
        for date in DATES:
            t = 9 * 60
            for _ in range(rng.randint(1, 3)):
                t += rng.choice([0, 30, 60])
                dur = rng.choice([30, 60, 90])
                if t + dur > 17 * 60:
                    break
                w._add_event(rng.choice(["Standup", "1:1", "Design review", "Lunch", "Planning"]), date, t, t + dur)
                t += dur
        return w

    def _add_event(self, title: str, date: str, start: int, end: int) -> dict[str, Any]:
        eid = f"EV-{self.next_event}"
        self.next_event += 1
        ev = {"event_id": eid, "title": title, "date": date, "start": to_hhmm(start), "end": to_hhmm(end)}
        self.events[eid] = ev
        return ev

    def add_order(self, sku: str, qty: int) -> dict[str, Any]:
        p = self.products[sku]
        oid = f"ORD-{self.next_order}"
        self.next_order += 1
        order = {"order_id": oid, "sku": sku, "quantity": qty, "total": round(p["price"] * qty, 2), "status": "placed"}
        self.orders[oid] = order
        p["stock"] -= qty
        self.balance = round(self.balance - order["total"], 2)
        return order

    def busy(self, date: str) -> list[tuple[int, int]]:
        return sorted((to_min(e["start"]), to_min(e["end"])) for e in self.events.values() if e["date"] == date)

    def earliest_free(self, date: str, dur: int, day_start: int = 9 * 60, day_end: int = 17 * 60) -> int | None:
        t = day_start
        for s, e in self.busy(date):
            if s - t >= dur:
                return t
            t = max(t, e)
        return t if day_end - t >= dur else None

    # ---- comparison -------------------------------------------------------
    def canonical(self) -> dict[str, Any]:
        """Id-insensitive view of the persistent state (excluding free-text mail bodies)."""
        return {
            "stock": {k: p["stock"] for k, p in self.products.items()},
            "orders": sorted(Counter((o["sku"], o["quantity"], o["status"]) for o in self.orders.values()).items()),
            "balance": round(self.balance, 2),
            "events": sorted(
                Counter((e["title"].strip().lower(), e["date"], e["start"], e["end"]) for e in self.events.values()).items()
            ),
        }

    def clone(self) -> World:
        return copy.deepcopy(self)

    # ---- tools ------------------------------------------------------------
    def tools(self) -> list[Tool]:
        def public(p: dict[str, Any]) -> dict[str, Any]:
            return {k: p[k] for k in ("sku", "name", "category", "price", "rating")} | {"in_stock": p["stock"]}

        def search_products(query: str = "", category: str = "") -> list[dict[str, Any]]:
            q = query.lower().split()
            hits = [
                public(p)
                for p in self.products.values()
                if (not category or p["category"] == category.lower())
                and all(tok in f"{p['name']} {p['category']}".lower() for tok in q)
            ]
            return hits

        def get_product(sku: str) -> dict[str, Any]:
            if sku not in self.products:
                raise ToolError(f"no product with sku {sku}")
            return public(self.products[sku])

        def place_order(sku: str, quantity: int) -> dict[str, Any]:
            if sku not in self.products:
                raise ToolError(f"no product with sku {sku}")
            if quantity < 1:
                raise ToolError("quantity must be >= 1")
            p = self.products[sku]
            if p["stock"] < quantity:
                raise ToolError(f"insufficient stock for {sku}: {p['stock']} available")
            if p["price"] * quantity > self.balance:
                raise ToolError("insufficient balance")
            return self.add_order(sku, quantity)

        def cancel_order(order_id: str) -> dict[str, Any]:
            o = self.orders.get(order_id)
            if o is None:
                raise ToolError(f"no order {order_id}")
            if o["status"] == "cancelled":
                raise ToolError(f"order {order_id} is already cancelled")
            o["status"] = "cancelled"
            self.products[o["sku"]]["stock"] += o["quantity"]
            self.balance = round(self.balance + o["total"], 2)
            return o

        def list_orders() -> list[dict[str, Any]]:
            return list(self.orders.values())

        def get_balance() -> dict[str, float]:
            return {"balance": self.balance}

        def list_events(date: str) -> list[dict[str, Any]]:
            return sorted((e for e in self.events.values() if e["date"] == date), key=lambda e: e["start"])

        def _check_slot(date: str, start: str, end: str, ignore: str | None = None) -> tuple[int, int]:
            try:
                s, e = to_min(start), to_min(end)
            except ValueError:
                raise ToolError("times must be HH:MM") from None
            if e <= s:
                raise ToolError("end must be after start")
            for ev in self.events.values():
                if ev["event_id"] != ignore and ev["date"] == date and s < to_min(ev["end"]) and to_min(ev["start"]) < e:
                    raise ToolError(f"conflicts with {ev['event_id']} ({ev['title']} {ev['start']}-{ev['end']})")
            return s, e

        def create_event(title: str, date: str, start: str, end: str) -> dict[str, Any]:
            s, e = _check_slot(date, start, end)
            return self._add_event(title, date, s, e)

        def update_event(event_id: str, date: str, start: str, end: str) -> dict[str, Any]:
            ev = self.events.get(event_id)
            if ev is None:
                raise ToolError(f"no event {event_id}")
            _check_slot(date, start, end, ignore=event_id)
            ev.update(date=date, start=start, end=end)
            return ev

        def delete_event(event_id: str) -> dict[str, str]:
            if self.events.pop(event_id, None) is None:
                raise ToolError(f"no event {event_id}")
            return {"deleted": event_id}

        def read_inbox() -> list[dict[str, Any]]:
            return self.inbox

        def send_email(to: str, subject: str, body: str) -> dict[str, str]:
            self.outbox.append({"to": to.strip().lower(), "subject": subject, "body": body})
            return {"status": "sent"}

        s = {"type": "string"}
        date = {"type": "string", "description": "YYYY-MM-DD"}
        hhmm = {"type": "string", "description": "HH:MM, 24h"}
        return [
            Tool("search_products", "Search the shop catalog. Empty query lists everything; category filters exactly "
                 f"(one of {', '.join(CATEGORIES)}).", obj({"query": s, "category": s}, required=[]), search_products),
            Tool("get_product", "Get one product by SKU.", obj({"sku": s}), get_product),
            Tool("place_order", "Buy a product. Charges the account balance.",
                 obj({"sku": s, "quantity": {"type": "integer", "minimum": 1}}), place_order),
            Tool("cancel_order", "Cancel an order and refund it.", obj({"order_id": s}), cancel_order),
            Tool("list_orders", "List all orders, oldest first.", obj({}), list_orders),
            Tool("get_balance", "Get the account balance in USD.", obj({}), get_balance),
            Tool("list_events", "List calendar events on a date.", obj({"date": date}), list_events),
            Tool("create_event", "Create a calendar event. Fails on overlap.",
                 obj({"title": s, "date": date, "start": hhmm, "end": hhmm}), create_event),
            Tool("update_event", "Move an event to a new date/time.",
                 obj({"event_id": s, "date": date, "start": hhmm, "end": hhmm}), update_event),
            Tool("delete_event", "Delete a calendar event.", obj({"event_id": s}), delete_event),
            Tool("read_inbox", "Read all emails in the inbox.", obj({}), read_inbox),
            Tool("send_email", "Send an email.", obj({"to": s, "subject": s, "body": s}), send_email),
        ]

    def env(self) -> Environment:
        return Environment(tools=self.tools(), state=self)
