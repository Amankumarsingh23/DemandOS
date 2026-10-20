"""Immutable observable-state API; no demand labels, prices or hidden shortages."""
from dataclasses import dataclass
from datetime import date

from demandos.inventory.core import integer, order_units


@dataclass(frozen=True)
class Order:
    order_id: str
    issue_day: int
    arrival_day: int
    quantity: int
    lead_time: int
    supplier_shock_id: str

    def __post_init__(self):
        for name in ["arrival_day", "quantity", "lead_time"]:
            integer(getattr(self, name), name, 1)
        integer(self.issue_day, "issue_day")
        if not self.order_id or self.arrival_day != self.issue_day+self.lead_time:
            raise ValueError("Order needs an ID and arrival_day=issue_day+lead_time")


@dataclass(frozen=True)
class Observation:
    day_index: int
    calendar_day: int
    simulation_date: date
    on_hand: int
    backlog: int
    on_order: int
    pipeline: tuple[Order, ...]
    fulfilled_sales_history: tuple[int, ...]

    @property
    def position(self):
        return self.on_hand+self.on_order-self.backlog


@dataclass(frozen=True)
class Action:
    quantity: int
    reason: str
    threshold: int
    target: int
    action_date: date


@dataclass(frozen=True)
class Policy:
    rule: str
    reorder_point: int
    target: int
    fixed_quantity: int = 20
    review_days: int = 7
    first_review_day: int = 1
    multiple: int = 1
    minimum: int = 0

    def __post_init__(self):
        if self.rule not in {"rQ", "rS", "RS"}:
            raise ValueError("Expected daily-review rQ/rS or periodic RS")
        for name in ["reorder_point", "target", "fixed_quantity", "minimum"]:
            integer(getattr(self, name), name)
        for name in ["review_days", "first_review_day", "multiple"]:
            integer(getattr(self, name), name, 1)

    def decide(self, observation):
        """At day CLOSE, after current demand. Frozen origin targets throughout."""
        if self.rule == "RS":
            review = observation.calendar_day >= self.first_review_day and (observation.calendar_day-self.first_review_day) % self.review_days == 0
            quantity = max(0, self.target-observation.position) if review else 0
            reason = "scheduled_review" if review else "not_review_day"
        else:
            trigger = observation.position <= self.reorder_point
            quantity = (self.fixed_quantity if self.rule == "rQ" else max(0, self.target-observation.position)) if trigger else 0
            reason = "position_at_or_below_r" if trigger else "position_above_r"
        return Action(order_units(quantity, self.multiple, self.minimum), reason,
                      self.reorder_point, self.target, observation.simulation_date)
