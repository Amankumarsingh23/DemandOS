"""Event-accurate daily engine, FIFO backlog, supplier queue and conservation gates."""
from collections import deque
from dataclasses import dataclass, asdict
from datetime import timedelta

import numpy as np
import pandas as pd

from demandos.inventory.core import counts, integer, Distribution
from demandos.inventory.scenarios import seed_for
from .policies import Observation, Order

STATUS = "SIMULATED_SALES_AS_DEMAND"
RUN_KEYS = ["series_id", "origin_date", "policy_id", "cost_scenario_id", "lead_scenario_id", "demand_scenario_id", "replication_id"]


@dataclass(frozen=True)
class Costs:
    holding: float
    lost_penalty: float
    backlog_daily: float
    order_fee: float
    purchase: float
    terminal_value: float = 0.

    def __post_init__(self):
        if not np.isfinite(list(asdict(self).values())).all() or min(asdict(self).values()) < 0 or self.terminal_value > self.purchase:
            raise ValueError("Finite nonnegative costs and terminal value <= purchase required")

    def book(self, stock, lost, backlog, ordered, terminal=False):
        stock, lost, backlog, ordered = [integer(x) for x in [stock, lost, backlog, ordered]]
        items = dict(holding_cost=self.holding*stock, lost_cost=self.lost_penalty*lost,
                     backlog_cost=self.backlog_daily*backlog, ordering_cost=self.order_fee*(ordered > 0),
                     procurement_cost=self.purchase*ordered,
                     terminal_credit=self.terminal_value*stock if terminal else 0.)
        items["operational_cost"] = sum(v for k, v in items.items() if k != "terminal_credit")-items["terminal_credit"]
        return items


@dataclass(frozen=True)
class Supplier:
    """Common randomness keyed by series/origin/replication/calendar issue day.

    Neither policy ID, cost regime nor order count enters the stream. One order
    per day per run; policies ordering on different days see different shocks.
    Weekend delay is an explicitly time-dependent sensitivity assumption.
    """
    days: tuple[int, ...]
    probabilities: tuple[float, ...]
    seed: int
    stream_id: str
    weekend_delay: int = 0

    def __post_init__(self):
        if not self.days or len(set(self.days)) != len(self.days):
            raise ValueError("Distinct positive lead supports required")
        for day in self.days: integer(day, minimum=1)
        Distribution(self.days, self.probabilities)
        integer(self.seed); integer(self.weekend_delay)

    def at(self, calendar_day, issue_date):
        integer(calendar_day)
        key = seed_for(self.seed, self.stream_id, calendar_day)
        uniform = np.random.default_rng(key).random()
        lead = int(self.days[min(np.searchsorted(np.cumsum(self.probabilities), uniform, side="right"), len(self.days)-1)])
        if pd.Timestamp(issue_date).dayofweek >= 5:
            lead += self.weekend_delay
        return lead, str(key), float(uniform)


def steps(demand, origin, costs, supplier, *, on_hand=10, initial_orders=(), mode="lost_sales",
             initial_backlog=0, warmup=0, calendar_offset=0):
    """All evaluator demand stays here; policy sees only immutable observations.

    Receive -> FIFO old backlog -> serve NEW demand -> unmet lost/backlogged ->
    post-demand IP -> decide -> queue delivery at issue_day+L -> book costs.
    L=1 arrives next day OPEN. Procurement is booked at ORDER, including any
    end-of-horizon pipeline; initial open orders were paid before the run.
    """
    y = counts(demand)
    if y.ndim != 1 or mode not in {"lost_sales", "backorders"}:
        raise ValueError("One daily integer path and explicit fulfillment mode required")
    on_hand, initial_backlog, warmup, calendar_offset = [integer(x) for x in [on_hand, initial_backlog, warmup, calendar_offset]]
    if warmup >= len(y) or (mode == "lost_sales" and initial_backlog):
        raise ValueError("Need scored days; lost sales cannot carry backlog")
    origin = pd.Timestamp(origin)
    if pd.isna(origin) or origin != origin.normalize(): raise ValueError("Daily forecast origin required")
    initial_orders = tuple(initial_orders)
    if any(not isinstance(x, Order) or x.issue_day != 0 for x in initial_orders):
        raise ValueError("Initial orders must have issue_day=0 and a known relative ETA")
    queue = {x.order_id: x for x in initial_orders}
    if len(queue) != len(initial_orders): raise ValueError("Duplicate initial order IDs")
    pending = deque([(0, initial_backlog)] if initial_backlog else [])
    backlog, first_stock = initial_backlog, on_hand
    initial_on_order = sum(x.quantity for x in initial_orders)
    daily, events, fulfilled_history = [], [], []
    all_orders = dict(queue)
    delivered = set()
    for order in initial_orders:
        events.append(dict(event="initial_outstanding", event_day=0, **asdict(order)))
    cumulative_received = cumulative_served = cumulative_new = cumulative_lost = cumulative_ordered = 0
    next_id = 1
    for day, demand_today in enumerate(y.astype(int), 1):
        date = (origin+pd.Timedelta(days=calendar_offset+day)).date()
        opening, old_backlog = on_hand, backlog
        opening_on_order = sum(x.quantity for x in queue.values())
        arrivals = sorted((x for x in queue.values() if x.arrival_day == day), key=lambda x: x.order_id)
        if any(x.arrival_day < day for x in queue.values()): raise AssertionError("Overdue order disappeared from arrivals")
        received = sum(x.quantity for x in arrivals)
        for order in arrivals:
            if order.order_id in delivered: raise AssertionError("Double delivery")
            delivered.add(order.order_id); del queue[order.order_id]
            events.append(dict(event="received", event_day=day, **asdict(order)))
        on_hand += received
        fulfilled_old, waiting = 0, 0
        while pending and on_hand:
            issue, quantity = pending[0]
            served = min(on_hand, quantity)
            on_hand -= served; backlog -= served; fulfilled_old += served; waiting += served*(day-issue)
            if served == quantity: pending.popleft()
            else: pending[0] = (issue, quantity-served)
        immediate = min(int(demand_today), on_hand)
        on_hand -= immediate
        unmet = int(demand_today)-immediate
        lost, created = (unmet, 0) if mode == "lost_sales" else (0, unmet)
        if created: pending.append((day, created)); backlog += created
        before_order = sum(x.quantity for x in queue.values())
        fulfilled_history.append(immediate)
        observation = Observation(day, calendar_offset+day, date, on_hand, backlog, before_order,
                                  tuple(sorted(queue.values(), key=lambda x: (x.arrival_day, x.order_id))), tuple(fulfilled_history))
        action = yield observation
        quantity = integer(action.quantity)
        if action.action_date != date: raise ValueError("Action timestamp differs from observable decision date")
        lead, shock_id, uniform = supplier.at(calendar_offset+day, date)
        if quantity:
            order_id = f"order_{next_id:05d}"
            if order_id in all_orders: raise ValueError("Order ID collision")
            order = Order(order_id, day, day+lead, quantity, lead, shock_id)
            queue[order_id] = order; all_orders[order_id] = order; next_id += 1
            events.append(dict(event="placed", event_day=day, **asdict(order)))
        ending_on_order = sum(x.quantity for x in queue.values())
        current_served = immediate+fulfilled_old
        # DAILY and CUMULATIVE hard conservation, integer/nonnegative state.
        if opening+received != current_served+on_hand or old_backlog+demand_today != fulfilled_old+immediate+backlog+lost:
            raise AssertionError("Daily stock/demand conservation failure")
        if demand_today != immediate+lost+created or backlog != sum(q for _, q in pending):
            raise AssertionError("New demand/backlog conservation failure")
        if ending_on_order != opening_on_order-received+quantity or on_hand < 0 or backlog < 0 or (mode == "lost_sales" and backlog):
            raise AssertionError("Queue/nonnegative state failure")
        if observation.position+quantity != on_hand+ending_on_order-backlog:
            raise AssertionError("Inventory position/action reconciliation failure")
        cumulative_received += received; cumulative_served += current_served
        cumulative_new += int(demand_today); cumulative_lost += lost; cumulative_ordered += quantity
        if first_stock+cumulative_received != cumulative_served+on_hand:
            raise AssertionError("Cumulative stock conservation failure")
        if initial_backlog+cumulative_new != cumulative_served+cumulative_lost+backlog:
            raise AssertionError("Cumulative demand conservation failure")
        if initial_on_order+cumulative_ordered != cumulative_received+ending_on_order:
            raise AssertionError("Cumulative order conservation failure")
        row = dict(day_index=day, calendar_day=calendar_offset+day, simulation_date=pd.Timestamp(date),
            scored=day > warmup, mode=mode, opening_stock=opening, opening_backlog=old_backlog,
            opening_on_order=opening_on_order, arrivals=received, demand_proxy=int(demand_today),
            immediate_served=immediate, old_backlog_fulfilled=fulfilled_old, new_backlog=created, lost_units=lost,
            ending_stock=on_hand, ending_backlog=backlog, on_order_before_action=before_order,
            ending_on_order=ending_on_order, position_before_order=observation.position,
            position_after_order=on_hand+ending_on_order-backlog, ordered=quantity,
            action_reason=action.reason, action_threshold=action.threshold, action_target=action.target,
            supplier_shock_id=shock_id, supplier_uniform=uniform, sampled_lead=lead,
            no_shortage=unmet == 0, fulfilled_backorder_wait_unit_days=waiting,
            stock_conservation_gap=0, demand_conservation_gap=0, queue_conservation_gap=0, status=STATUS,
            **costs.book(on_hand, lost, backlog, quantity, terminal=day == len(y)))
        daily.append(row)
    if set(all_orders) != set(queue) | delivered or set(queue) & delivered:
        raise AssertionError("Unreconciled order identities")
    for order in queue.values(): events.append(dict(event="terminal_outstanding", event_day=len(y), **asdict(order)))
    return pd.DataFrame(daily), pd.DataFrame(events)


def simulate(demand, origin, policy, costs, supplier, *, on_hand=10, initial_orders=(), mode="lost_sales",
             initial_backlog=0, warmup=0, calendar_offset=0):
    """Preserved full-path API; drive the same validated timeline one day at a time."""
    timeline = steps(demand, origin, costs, supplier, on_hand=on_hand, initial_orders=initial_orders,
                     mode=mode, initial_backlog=initial_backlog, warmup=warmup, calendar_offset=calendar_offset)
    try:
        observation = next(timeline)
        while True:
            observation = timeline.send(policy.decide(observation))
    except StopIteration as finished:
        return finished.value
