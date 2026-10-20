"""Sparse LP/MILP allocation, explicit status handling and independent feasibility."""
from dataclasses import dataclass, field, replace
import hashlib
from time import perf_counter

import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, vstack

from demandos.inventory.core import integer

TOL = 1e-7


@dataclass(frozen=True)
class Candidates:
    keys: tuple[str, ...]
    request: np.ndarray
    purchase: np.ndarray
    volume: np.ndarray
    benefit: np.ndarray
    suppliers: tuple[str, ...]
    stores: tuple[str, ...]
    fee: np.ndarray
    pack: np.ndarray
    moq: np.ndarray
    minimum: np.ndarray
    item_cap: np.ndarray | None = None

    def __post_init__(self):
        n = len(self.keys)
        if not n or len(set(self.keys)) != n or tuple(sorted(self.keys)) != self.keys or len(self.suppliers) != n or len(self.stores) != n:
            raise ValueError("Unique stable sorted candidate keys and aligned groups required")
        for name in ["request", "purchase", "volume", "benefit", "fee", "pack", "moq", "minimum"]:
            a = np.asarray(getattr(self, name), float)
            if a.shape != (n,) or not np.isfinite(a).all() or (a < 0).any(): raise ValueError(f"Invalid {name}")
            object.__setattr__(self, name, a.copy())
        for name in ["request", "pack", "moq", "minimum"]:
            if np.any(getattr(self, name) != np.floor(getattr(self, name))): raise ValueError("Integer request/pack/MOQ/minimum input required; no implicit rounding")
        if (self.pack < 1).any() or (self.purchase <= 0).any() or (self.volume <= 0).any(): raise ValueError("Positive purchase, volume, case packs required")
        cap=self.request.copy() if self.item_cap is None else np.asarray(self.item_cap,float).copy()
        if cap.shape!=(n,) or not np.isfinite(cap).all() or (cap<0).any() or (cap!=np.floor(cap)).any():raise ValueError("Integer nonnegative per-item supplier caps required")
        object.__setattr__(self,"item_cap",cap)


@dataclass(frozen=True)
class Resources:
    budget: float
    flow_volume: float
    suppliers: dict
    receiving: dict = field(default_factory=dict)

    def __post_init__(self):
        numbers = [self.budget, self.flow_volume, *self.suppliers.values(), *self.receiving.values()]
        if not np.isfinite(numbers).all() or min(numbers) < 0: raise ValueError("Finite nonnegative hard resource limits required")


@dataclass(frozen=True)
class Fairness:
    minimum_share: float = 0.
    maximum_disparity: float = 1.
    soft: bool = False
    penalty: float = 1000.

    def __post_init__(self):
        if not 0 <= self.minimum_share <= 1 or not 0 <= self.maximum_disparity <= 1 or not np.isfinite(self.penalty) or self.penalty <= 0:
            raise ValueError("Invalid explicit fairness preference")


def resource_use(c, q):
    q = np.asarray(q, float)
    return dict(budget=float(c.purchase@q+c.fee@(q > TOL)), flow_volume=float(c.volume@q),
        suppliers={s:float(q[np.array(c.suppliers)==s].sum()) for s in sorted(set(c.suppliers))},
        receiving={s:float(q[np.array(c.stores)==s].sum()) for s in sorted(set(c.stores))})


def fair_slacks(c, q, fairness):
    shares = {}
    for store in sorted(set(c.stores)):
        mask = np.array(c.stores)==store; requested = c.request[mask].sum()
        if requested > 0: shares[store] = float(np.sum(q[mask])/requested)
    shortfall = {s:max(0., fairness.minimum_share-v) for s,v in shares.items()}
    disparity = max(shares.values(),default=0.)-min(shares.values(),default=0.)
    return dict(shares=shares, minimum_shortfall=shortfall, disparity=disparity,
                disparity_slack=max(0.,disparity-fairness.maximum_disparity))


def verify(c, resources, q, fairness=Fairness(), *, integer_orders=True):
    """Check original business constraints, independently of solver matrices."""
    q = np.asarray(q, float)
    if q.shape != c.request.shape or not np.isfinite(q).all() or (q < -TOL).any() or (q > c.request+TOL).any() or (q>c.item_cap+TOL).any() or (q < c.minimum-TOL).any():
        raise AssertionError("Allocation violates request/minimum bounds")
    if integer_orders and (np.abs(q-np.rint(q)) > TOL).any(): raise AssertionError("Fractional allocation cannot be implemented")
    if integer_orders and (np.abs(q/c.pack-np.rint(q/c.pack)) > TOL).any(): raise AssertionError("Case-pack violation")
    if integer_orders and np.any((q > TOL)&(q < c.moq-TOL)): raise AssertionError("Activated MOQ violation")
    use = resource_use(c,q)
    if use["budget"] > resources.budget+TOL or use["flow_volume"] > resources.flow_volume+TOL: raise AssertionError("Hard budget/flow capacity violation")
    for family in ["suppliers","receiving"]:
        for group, limit in getattr(resources,family).items():
            if use[family].get(group,0.) > limit+TOL: raise AssertionError(f"Hard {family} capacity violation")
    fair = fair_slacks(c,q,fairness)
    if not fairness.soft and (max(fair["minimum_shortfall"].values(),default=0.)>TOL or fair["disparity_slack"]>TOL): raise AssertionError("Hard fairness violation")
    return use, fair


def basic_lp(benefit, matrix, limits, bounds=None):
    """Continuous maximization; economic RHS dual = NEGATIVE minimization marginal."""
    result = linprog(-np.asarray(benefit,float), A_ub=csr_matrix(matrix), b_ub=limits,
                     bounds=(0,None) if bounds is None else bounds, method="highs",
                     options={"presolve":True,"primal_feasibility_tolerance":TOL,"dual_feasibility_tolerance":TOL})
    return dict(status=status(result), x=result.x, benefit=-result.fun if result.x is not None else None,
                shadow_prices=-result.ineqlin.marginals if result.status==0 else None,
                slack=result.ineqlin.residual if result.status==0 else None)


def status(result):
    if result.status==0:
        gap=getattr(result,"mip_gap",0.)
        return "OPTIMAL" if gap is None or gap<=TOL else "FEASIBLE_NOT_PROVEN_OPTIMAL"
    if result.status==1: return "FEASIBLE_NOT_PROVEN_OPTIMAL" if result.x is not None else "TIME_LIMIT_NO_INCUMBENT"
    return {2:"INFEASIBLE",3:"UNBOUNDED"}.get(result.status,"NUMERICAL_ERROR")


def diagnostics(c,r):
    forced=np.where(c.minimum>0,np.maximum(np.ceil(c.minimum/c.pack),np.ceil(c.moq/c.pack))*c.pack,0.)
    use=resource_use(c,forced);violations=[]
    if use["budget"]>r.budget+TOL:violations.append("budget")
    if use["flow_volume"]>r.flow_volume+TOL:violations.append("warehouse_procurement_flow")
    for family in ["suppliers","receiving"]:
        violations.extend(family+":"+name for name,limit in getattr(r,family).items() if use[family].get(name,0)>limit+TOL)
    return dict(forced_minimum_budget=use["budget"],forced_minimum_flow_volume=use["flow_volume"],
                budget=r.budget,flow_capacity=r.flow_volume,violating_families=violations)


def build(c, r, fairness=Fairness(), *, demand=None, probabilities=None, stock=None, cu=40., co=40.):
    """n=case counts; z=binary activation; optional scenario shortage/excess recourse.

    q=pack*n is a SINGLE first-stage vector across all scenarios. No scenario
    purchases or hidden future labels. Marginal mismatch includes Cu/Co only.
    """
    n=len(c.keys); scenario=demand is not None
    if scenario:
        d=np.asarray(demand,float); p=np.asarray(probabilities,float); stock=np.asarray(stock,float)
        if d.ndim!=2 or d.shape[0]!=n or stock.shape!=(n,) or p.shape!=(d.shape[1],) or not np.isfinite(d).all() or (d<0).any() or not np.isfinite(stock).all() or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-12,rtol=0):
            raise ValueError("Aligned finite demand, predecision stock and normalized scenario probabilities required")
        if min(cu,co)<0 or cu+co<=0 or not np.isfinite([cu,co]).all(): raise ValueError("Invalid mismatch costs")
        w=d.shape[1]
    else: w=0
    fair_rows=[]
    groups=[]
    for store in sorted(set(c.stores)):
        mask=np.array(c.stores)==store; total=c.request[mask].sum()
        if total>0: groups.append((store,mask.astype(float)/total))
    for name,share in groups:
        if fairness.minimum_share>0: fair_rows.append(("fair_min:"+name,-share,-fairness.minimum_share))
    if fairness.maximum_disparity<1:
        for i,(a,share_a) in enumerate(groups):
            for b,share_b in groups[i+1:]:
                fair_rows.extend([("fair_disparity:"+a+":"+b,share_a-share_b,fairness.maximum_disparity),
                                  ("fair_disparity:"+b+":"+a,share_b-share_a,fairness.maximum_disparity)])
    slack_count=len(fair_rows) if fairness.soft else 0
    variables=2*n+2*n*w+slack_count
    objective=np.zeros(variables); low=np.zeros(variables); upper=np.full(variables,np.inf); integral=np.zeros(variables,int)
    upper[:n]=np.floor(np.minimum(c.request,c.item_cap)/c.pack); upper[n:2*n]=1
    # An activated positive order must meet BOTH one case and MOQ.
    minimum_cases=np.maximum(1,np.ceil(c.moq/c.pack))
    upper[:n]=np.where(upper[:n]>=minimum_cases,upper[:n],0)
    low[:n]=np.ceil(c.minimum/c.pack); integral[:2*n]=1
    if scenario:
        objective[2*n:2*n+n*w]=np.tile(p,n)*cu
        objective[2*n+n*w:2*n+2*n*w]=np.tile(p,n)*co
    else: objective[:n]=-c.benefit*c.pack
    if slack_count: objective[-slack_count:]=fairness.penalty
    entries=[]; rhs=[]; names=[]
    def row(name,coefficients,limit):
        names.append(name);rhs.append(float(limit));entries.append(coefficients)
    row("budget",{**{i:c.purchase[i]*c.pack[i] for i in range(n)},**{n+i:c.fee[i] for i in range(n)}},r.budget)
    row("warehouse_procurement_flow",{i:c.volume[i]*c.pack[i] for i in range(n)},r.flow_volume)
    for family in ["suppliers","receiving"]:
        candidate_groups=c.suppliers if family=="suppliers" else c.stores
        for name,limit in sorted(getattr(r,family).items()):
            row(family+":"+name,{i:c.pack[i] for i in range(n) if candidate_groups[i]==name},limit)
    for i in range(n):
        row("activation:"+c.keys[i],{i:1,n+i:-upper[i]},0)
        row("moq:"+c.keys[i],{i:-1,n+i:minimum_cases[i]},0)
    if scenario:
        for i in range(n):
            for j in range(w):
                row(f"short:{i}:{j}",{i:-c.pack[i],2*n+i*w+j:-1},stock[i]-d[i,j])
                row(f"excess:{i}:{j}",{i:c.pack[i],2*n+n*w+i*w+j:-1},d[i,j]-stock[i])
    for index,(name,coeff,limit) in enumerate(fair_rows):
        values={i:coeff[i]*c.pack[i] for i in range(n) if coeff[i]}
        if fairness.soft: values[variables-slack_count+index]=-1
        row(name,values,limit)
    rr=[];cc=[];vv=[]
    for i,values in enumerate(entries):
        for j,value in values.items():
            if value: rr.append(i);cc.append(j);vv.append(value)
    matrix=csr_matrix((vv,(rr,cc)),shape=(len(rhs),variables))
    return dict(c=objective,A=matrix,b=np.asarray(rhs),low=low,upper=upper,integral=integral,n=n,names=names,
                objective_type="expected_mismatch_CuCo_no_procurement_overlay" if scenario else "assumed_linear_benefit",
                forced_minimum_budget=float(c.purchase@c.minimum+c.fee@(c.minimum>0)))


def solve(c, resources, fairness=Fairness(), *, demand=None, probabilities=None, stock=None, cu=40.,co=40.,time_limit=3.,relax=False):
    start=perf_counter();m=build(c,resources,fairness,demand=demand,probabilities=probabilities,stock=stock,cu=cu,co=co)
    if (m["low"]>m["upper"]).any():
        return dict(status="INFEASIBLE",quantity=None,objective=None,mip_gap=None,
            diagnostic={**diagnostics(c,resources),"message":"Forced minimum exceeds pack-adjusted request"},runtime=perf_counter()-start)
    if relax:
        result=linprog(m["c"],A_ub=m["A"],b_ub=m["b"],bounds=list(zip(m["low"],m["upper"])),method="highs",options={"presolve":True,"time_limit":time_limit,"primal_feasibility_tolerance":TOL,"dual_feasibility_tolerance":TOL})
    else:
        result=milp(m["c"],integrality=m["integral"],bounds=Bounds(m["low"],m["upper"]),constraints=LinearConstraint(m["A"],-np.inf,m["b"]),options={"presolve":True,"time_limit":time_limit,"mip_rel_gap":0.})
    state=status(result); x=result.x
    if x is not None:
        residual=np.max(m["A"]@x-m["b"],initial=0.)
        if residual>TOL or np.any(x<m["low"]-TOL) or np.any(x>m["upper"]+TOL) or (not relax and np.max(np.abs(x[:2*m['n']]-np.rint(x[:2*m['n']])),initial=0)>TOL):
            state="NUMERICAL_ERROR";x=None
    q=None if x is None else x[:m["n"]]*c.pack
    if q is not None and not relax:
        q=np.rint(q).astype(int); verify(c,resources,q,fairness)
    # Stable sorted variables/matrices and deterministic HiGHS traversal are the
    # tie policy. No economic objective perturbation or claim of unique optimum.
    digest=hashlib.sha256(m["A"].data.tobytes()+m["A"].indices.tobytes()+m["A"].indptr.tobytes()+m["c"].tobytes()+m["b"].tobytes()).hexdigest()
    return dict(status=state,quantity=q,objective=float(m["c"]@x) if x is not None else None,
        mip_gap=getattr(result,"mip_gap",None),dual_bound=getattr(result,"mip_dual_bound",None),
        shadow_prices=(-result.ineqlin.marginals).tolist() if relax and x is not None and state=="OPTIMAL" and demand is None else None,
        minimization_rhs_marginals=result.ineqlin.marginals.tolist() if relax and x is not None and state=="OPTIMAL" else None,
        matrix=dict(shape=list(m["A"].shape),nnz=m["A"].nnz,sha256=digest,row_names=m["names"],
                    objective=m["objective_type"],tie_policy="stable_sorted_keys_and_deterministic_HiGHS",case_upper=m["upper"][:m["n"]].tolist()),
        diagnostic=dict(message=str(result.message),**diagnostics(c,resources)),runtime=perf_counter()-start)


def baseline(c,resources,method):
    q=np.zeros(len(c.keys),int)
    if method=="proportional":
        def rounded(alpha):
            trial=(np.floor(np.minimum(c.request*alpha,c.item_cap)/c.pack)*c.pack).astype(int)
            trial[trial<c.moq]=0
            return trial
        try:
            full=rounded(1.);verify(c,resources,full)
        except AssertionError:pass
        else:return full
        # Actual cash is discontinuous at activation thresholds. Search the
        # common scaling factor with fees, MOQ and all shared resources included.
        lo,hi=0.,1.
        for _ in range(48):
            alpha=(lo+hi)/2;trial=rounded(alpha)
            try:verify(c,resources,trial)
            except AssertionError:hi=alpha
            else:lo=alpha;q=trial
        verify(c,resources,q)
        return q
    if method!="greedy": raise ValueError("Unknown feasible baseline")
    order=sorted(range(len(q)),key=lambda i:(-c.benefit[i]/c.purchase[i],c.keys[i]))
    resource_only=replace(c,moq=np.zeros(len(q)))
    for i in order:
        upper=int(min(c.request[i],c.item_cap[i])//c.pack[i]); minimum=max(1,int(np.ceil(c.moq[i]/c.pack[i])))
        lo,hi=0,upper
        while lo<hi:
            mid=(lo+hi+1)//2;trial=q.copy();trial[i]=mid*int(c.pack[i])
            try:verify(resource_only,resources,trial)
            except AssertionError:hi=mid-1
            else:lo=mid
        if lo>=minimum:q[i]=lo*int(c.pack[i])
    verify(c,resources,q);return q
