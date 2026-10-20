"""Discrete stock-target economics and explicit single-order lost-sales timing."""
from dataclasses import dataclass
from math import ceil
from statistics import NormalDist

import numpy as np


def integer(value, name="quantity", minimum=0):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value != int(value) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def counts(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Finite nonnegative integer demand required")
    return values


@dataclass(frozen=True)
class Economics:
    price: float
    purchase: float
    salvage: float
    shortage_penalty: float = 0.
    overage_penalty: float = 0.

    def __post_init__(self):
        if not np.isfinite(list(vars(self).values())).all() or min(vars(self).values()) < 0:
            raise ValueError("Nonnegative finite hypothetical monetary parameters required")
        if self.cu < 0 or self.co < 0 or self.cu+self.co <= 0:
            raise ValueError("Require Cu, Co >= 0 and Cu+Co > 0")

    @property
    def cu(self):
        return self.price-self.purchase+self.shortage_penalty

    @property
    def co(self):
        return self.purchase-self.salvage+self.overage_penalty

    @property
    def fractile(self):
        return self.cu/(self.cu+self.co)

    def cost(self, stock, demand):
        stock = counts(stock)
        demand = counts(demand)
        return self.cu*np.maximum(demand-stock, 0)+self.co*np.maximum(stock-demand, 0)


class Distribution:
    """Finite integer demand, normalized validated weights, left inverse CDF.

    tau=0 returns zero (smallest nonnegative integer); tau=1 returns max
    positive-probability support. Duplicate support values are combined.
    """
    def __init__(self, values, probabilities=None):
        values = counts(values)
        if values.ndim != 1:
            raise ValueError("One-dimensional demand support required")
        p = np.full(len(values), 1/len(values)) if probabilities is None else np.asarray(probabilities, float)
        if p.shape != values.shape or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1., atol=1e-12, rtol=0):
            raise ValueError("Probabilities must be finite, nonnegative and sum to one")
        self.values, inverse = np.unique(values[p > 0], return_inverse=True)
        self.p = np.bincount(inverse, weights=p[p > 0])
        self.p /= self.p.sum()
        self.cumulative = np.cumsum(self.p)
        self.cumulative[-1] = 1.

    def cdf(self, quantity):
        if not np.isfinite(quantity):
            raise ValueError("Finite CDF argument required")
        return float(self.p[self.values <= quantity].sum())

    def quantile(self, tau):
        if not np.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError("Quantile level must be in [0,1]")
        if tau == 0:
            return 0
        # Suppress only machine precision summation error at exact CDF ties.
        index = np.searchsorted(self.cumulative, tau-8*np.finfo(float).eps, side="left")
        return int(self.values[min(index, len(self.values)-1)])

    @property
    def mean(self):
        return float(self.p@self.values)

    @property
    def variance(self):
        return float(self.p@((self.values-self.mean)**2))

    def metrics(self, stock, economics):
        stock = integer(stock)
        shortage = float(self.p@np.maximum(self.values-stock, 0))
        leftover = float(self.p@np.maximum(stock-self.values, 0))
        return dict(expected_cost=economics.cu*shortage+economics.co*leftover,
                    shortage=shortage, leftover=leftover, csl=self.cdf(stock),
                    fill_rate=1-shortage/self.mean if self.mean else None)

    def curve(self, economics):
        q = np.unique(np.r_[0, self.values]).astype(int)
        mass = np.r_[0., np.cumsum(self.p)]
        weighted = np.r_[0., np.cumsum(self.p*self.values)]
        i = np.searchsorted(self.values, q, side="right")
        shortage = self.mean-weighted[i]-q*(1-mass[i])
        leftover = q*mass[i]-weighted[i]
        return q, economics.cu*shortage+economics.co*leftover


def service(demand, served, probabilities=None):
    demand, served = counts(demand), counts(served)
    if demand.shape != served.shape or demand.ndim != 1 or (served > demand).any():
        raise ValueError("Aligned cycle totals with served <= demand required")
    if probabilities is None:
        p = np.full(len(demand), 1/len(demand))
    else:
        # Validate without removing zero-weight cycles or changing their order.
        Distribution(np.arange(len(demand)), probabilities)
        p = np.asarray(probabilities, float)
    total = float(p@demand)
    return dict(csl=float(p@(served == demand)), fill_rate=float(p@served/total) if total else None)


def lead_moments(mean, variance, leads, probabilities):
    """IID daily demand independent of lead time; this is an explicit assumption."""
    if not np.isfinite([mean, variance]).all() or min(mean, variance) < 0:
        raise ValueError("Invalid daily moments")
    d = Distribution(leads, probabilities)
    return d.mean*mean, d.mean*variance+d.variance*mean**2


def covariance_variance(covariance):
    c = np.asarray(covariance, float)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or not np.isfinite(c).all() or not np.allclose(c, c.T) or np.linalg.eigvalsh(c).min() < -1e-9:
        raise ValueError("Finite symmetric positive semidefinite covariance required")
    return float(c.sum())


def normal_rop(mean, sigma, lead, alpha):
    integer(lead, "lead", 1)
    if not np.isfinite([mean, sigma, alpha]).all() or min(mean, sigma) < 0 or not 0 < alpha < 1:
        raise ValueError("Invalid explicitly selected IID normal approximation")
    continuous = lead*mean+NormalDist().inv_cdf(alpha)*sigma*np.sqrt(lead)
    return continuous, max(0, ceil(continuous))


def inventory_position(on_hand, on_order, backorders=0):
    return integer(on_hand)+integer(on_order)-integer(backorders)


def order_units(value, multiple=1, minimum=0):
    multiple, minimum = integer(multiple, "multiple", 1), integer(minimum)
    if not np.isfinite(value):
        raise ValueError("Finite order required")
    return 0 if value <= 0 else int(ceil(max(value, minimum)/multiple)*multiple)


def policy(position, rop, target, fixed_quantity, multiple=1, minimum=0):
    if not np.isfinite(position) or position != int(position):
        raise ValueError("Integer inventory position required")
    integer(rop); integer(target); integer(fixed_quantity)
    trigger = position <= rop
    up_to = order_units(target-position, multiple, minimum)
    return dict(trigger=trigger, periodic_up_to=up_to,
                continuous_up_to=up_to if trigger else 0,
                continuous_fixed=order_units(fixed_quantity, multiple, minimum) if trigger else 0)


def cycle(paths, lead, review, on_hand, on_order, order, trace=False):
    """One origin decision, lost sales, no reordering or backlog.

    E is day-close; consume days 1..L using existing on-hand. Both assumed open
    orders and the new order arrive at day L+1 OPEN, before its consumption.
    Finish at day L+R CLOSE. Orders never serve earlier shortages.
    Returns vector cycle totals and sum of daily closing stock (unit-days).
    """
    y = counts(paths)
    lead, review = integer(lead, "lead", 1), integer(review, "review", 1)
    if y.ndim != 2 or y.shape[1] < lead+review:
        raise ValueError("Complete lead+review daily path required; no truncation")
    stock = np.full(len(y), integer(on_hand), dtype=float)
    integer(on_order); integer(order)
    served, exposure = np.zeros(len(y)), np.zeros(len(y))
    daily = []
    for day in range(lead+review):
        opening = stock.copy() if trace else None
        if day == lead:
            stock += on_order+order
        consumption = np.minimum(stock, y[:, day])
        served += consumption
        stock -= consumption
        exposure += stock
        if trace:
            for sample in range(len(y)):
                daily.append(dict(sample_id=sample, day=day+1, opening_stock=float(opening[sample]),
                    receipt=on_order+order if day == lead else 0, demand=float(y[sample, day]),
                    served=float(consumption[sample]), lost_sales=float(y[sample, day]-consumption[sample]),
                    closing_stock=float(stock[sample]), status="EXACT_SYNTHETIC_TOY"))
    demand = y[:, :lead+review].sum(axis=1)
    result = dict(demand=demand, served=served, shortage=demand-served, leftover=stock, stock_unit_days=exposure)
    if trace:
        result["daily"] = daily
    return result
