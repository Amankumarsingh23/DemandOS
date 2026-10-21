"""Risk epigraphs over the unchanged Season 20 sparse allocation feasible set."""
from dataclasses import dataclass
from time import perf_counter
import hashlib
import warnings

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, hstack, vstack

from demandos.optimization.model import build, solve as nominal_solve, verify, status, Fairness, TOL, fair_slacks
from .metrics import tail, distribution, worst_probability


@dataclass(frozen=True)
class Scenarios:
    demand: np.ndarray  # series x mutually exclusive JOINT scenarios
    probabilities: np.ndarray
    ids: tuple[str, ...]

    def __post_init__(self):
        d = np.asarray(self.demand, float)
        p = np.asarray(self.probabilities, float)
        if d.ndim != 2 or not d.shape[0] or not d.shape[1] or not np.isfinite(d).all() or (d < 0).any() or (d != np.floor(d)).any():
            raise ValueError("Finite nonnegative integer scenario demand required")
        distribution(np.zeros(d.shape[1]), p)
        if len(self.ids) != d.shape[1] or len(set(self.ids)) != len(self.ids) or any(not isinstance(k,str) or not k for k in self.ids):
            raise ValueError("Unique aligned scenario identifiers required")
        object.__setattr__(self, "demand", d.copy())
        object.__setattr__(self, "probabilities", p.copy())
        object.__setattr__(self, "ids", tuple(self.ids))
        self.demand.setflags(write=False)
        self.probabilities.setflags(write=False)


def losses(q, stock, scenarios, cu, co):
    x = np.asarray(stock)[:, None]+np.asarray(q)[:, None]-scenarios.demand
    return (co*np.maximum(x, 0)+cu*np.maximum(-x, 0)).sum(axis=0)


def solve(c, resources, scenarios, stock, *, method="mean", alpha=.9, penalty=0., risk_limit=None,
          fairness=Fairness(), cu=40., co=40., interval=(.05,.20), time_limit=3.):
    """One first-stage order vector; shortage/excess are cost accounting recourse.

    mean_cvar = E[L]+penalty*CVaR_alpha(L). Fairness preference costs, if
    configured soft, are included identically in every scenario and oracle.
    They remain separately labeled from economic mismatch and operating costs.
    """
    start = perf_counter()
    if method not in {"mean","mean_cvar","ceiling","minimax","regret","dro"}:
        raise ValueError("Unknown finite-scenario objective")
    if not np.isfinite(penalty) or penalty < 0 or not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Finite nonnegative risk penalty and positive time limit required")
    tail([0.], [1.], alpha)
    if method == "ceiling" and (risk_limit is None or not np.isfinite(risk_limit)):
        raise ValueError("Explicit finite hard CVaR limit required")
    d, p = scenarios.demand, scenarios.probabilities
    n, w = d.shape
    base = build(c, resources, fairness, demand=d, probabilities=p, stock=stock, cu=cu, co=co)
    size = len(base["c"])
    # Normalize monetary rows/objective to avoid large, degenerate epigraphs.
    # All published losses, eta, bounds and objectives are converted back.
    cost_scale = max(1.,cu,co,fairness.penalty if fairness.soft else 0.)
    loss_matrix = np.zeros((w, size))
    for j in range(w):
        loss_matrix[j, 2*n+np.arange(n)*w+j] = cu
        loss_matrix[j, 2*n+n*w+np.arange(n)*w+j] = co
    # Existing builder appends only fairness slacks after the mismatch auxiliaries.
    loss_matrix[:, 2*n+2*n*w:] = base["c"][2*n+2*n*w:]
    loss_matrix /= cost_scale
    oracle = None
    oracle_records = []
    if method == "regret":
        oracle = []
        for j in range(w):
            r = nominal_solve(c, resources, fairness, demand=d[:, j:j+1], probabilities=[1.], stock=stock, cu=cu, co=co, time_limit=time_limit)
            oracle_records.append({k:r.get(k) for k in ["status","objective","mip_gap","runtime","dual_bound"]})
            if r["status"] != "OPTIMAL":
                return dict(status="ORACLE_NOT_PROVEN_OPTIMAL", quantity=None, objective=None,
                            mip_gap=None, runtime=perf_counter()-start, oracle_solves=oracle_records)
            oracle.append(r["objective"])
        oracle = np.asarray(oracle)/cost_scale
    extra = w+1 if method in {"mean_cvar","ceiling"} else (1 if method in {"minimax","regret","dro"} else 0)
    total = size+extra
    objective = np.zeros(total)
    low, upper = np.r_[base["low"], np.zeros(extra)], np.r_[base["upper"], np.full(extra,np.inf)]
    matrix = hstack([base["A"], csr_matrix((base["A"].shape[0], extra))], format="csr")
    rhs = list(base["b"])
    rows = []
    def add(coeff, bound):
        rows.append(coeff); rhs.append(float(bound))
    if method in {"mean","mean_cvar","ceiling"}:
        objective[:size] = p@loss_matrix
    if method in {"mean_cvar","ceiling"}:
        eta = size
        low[eta] = -np.inf
        for j in range(w):
            row = np.zeros(total); row[:size] = loss_matrix[j]; row[eta] = -1; row[eta+1+j] = -1
            add(row, 0)
        cvar_row = np.zeros(total); cvar_row[eta] = 1; cvar_row[eta+1:] = p/(1-alpha)
        if method == "ceiling": add(cvar_row, risk_limit/cost_scale)
        else: objective += penalty*cvar_row
    if method in {"minimax","regret"}:
        objective[size] = 1; low[size] = -np.inf
        for j in range(w):
            row = np.r_[loss_matrix[j], -1.]
            add(row, 0 if oracle is None else oracle[j])
    if method == "dro":
        worst_probability(0, 1, interval)
        if w % 2: raise ValueError("DRO requires equal-sized normal/shock support families")
        objective[size] = 1; low[size] = -np.inf
        for probability in interval:
            weights = np.r_[np.full(w//2,(1-probability)/(w//2)), np.full(w//2,probability/(w//2))]
            add(np.r_[weights@loss_matrix,-1.], 0)
    if rows: matrix = vstack([matrix, csr_matrix(np.asarray(rows))], format="csr")
    if (low > upper).any():
        return dict(status="INFEASIBLE",quantity=None,objective=None,mip_gap=None,runtime=perf_counter()-start,oracle_solves=oracle_records)
    # SciPy forwards these native HiGHS controls. Its default MIP feasibility
    # tolerance (1e-6) is looser than our independent 1e-7 business-unit gate.
    # Suppress only the documented forwarding notice, not numerical warnings.
    solver_started=perf_counter()
    attempts=[]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Unrecognized options detected:.*",category=RuntimeWarning)
        result = milp(objective, integrality=np.r_[base["integral"],np.zeros(extra,int)],
                      bounds=Bounds(low,upper), constraints=LinearConstraint(matrix,-np.inf,rhs),
                      options={"presolve":True,"time_limit":time_limit,"mip_rel_gap":0.,
                               "mip_feasibility_tolerance":1e-7,"primal_feasibility_tolerance":1e-7})
        attempts.append(dict(status=status(result),message=str(result.message),runtime=perf_counter()-solver_started))
        # HiGHS 1.12 may fail transforming a presolved integer solution back to
        # the original degenerate regret epigraph. Retry the IDENTICAL model
        # without presolve, within the original total wall-time budget.
        if result.status == 4:
            remaining=time_limit-(perf_counter()-solver_started)
            if remaining>0:
                retry_started=perf_counter()
                result=milp(objective,integrality=np.r_[base["integral"],np.zeros(extra,int)],
                            bounds=Bounds(low,upper),constraints=LinearConstraint(matrix,-np.inf,rhs),
                            options={"presolve":False,"time_limit":remaining,"mip_rel_gap":0.,
                                     "mip_feasibility_tolerance":1e-7,"primal_feasibility_tolerance":1e-7})
                attempts.append(dict(status=status(result),message=str(result.message),runtime=perf_counter()-retry_started))
    state, x = status(result), result.x
    residuals = None
    raw_residuals = None
    if x is not None:
        raw_residuals = dict(matrix=float((matrix@x-np.asarray(rhs)).max(initial=0)),
                            lower=float((low-x).max(initial=0)),upper=float((x-upper).max(initial=0)))
        # Reconstruct ONLY continuous cost auxiliaries from the issued integer
        # quantities. Degenerate monetary epigraphs can carry sub-micro roundoff.
        # No resource constraint or order is relaxed/clipped; verify q first.
        if np.abs(x[:2*n]-np.rint(x[:2*n])).max(initial=0) <= TOL:
            polished=x.copy();polished[:2*n]=np.rint(x[:2*n])
            q=polished[:n]*c.pack
            try: verify(c,resources,q,fairness)
            except AssertionError: pass
            else:
                position=np.asarray(stock)[:,None]+q[:,None]
                polished[2*n:2*n+n*w]=np.maximum(d-position,0).ravel()
                polished[2*n+n*w:2*n+2*n*w]=np.maximum(position-d,0).ravel()
                if size>2*n+2*n*w:
                    polished[2*n+2*n*w:size]=0
                    fair_rows=[i for i,name in enumerate(base["names"]) if name.startswith("fair_")]
                    residual=base["A"]@polished[:size]-base["b"]
                    polished[2*n+2*n*w:size]=np.maximum(residual[fair_rows],0)
                exact_losses=loss_matrix@polished[:size]
                if method in {"mean_cvar","ceiling"}:
                    polished[size]=tail(exact_losses,p,alpha)["var"]
                    polished[size+1:]=np.maximum(exact_losses-polished[size],0)
                elif method in {"minimax","regret"}:
                    polished[size]=np.max(exact_losses if oracle is None else exact_losses-oracle)
                elif method=="dro":
                    polished[size]=worst_probability(exact_losses[:w//2].mean(),exact_losses[w//2:].mean(),interval)["cost"]
                if state!="OPTIMAL" or np.isclose(objective@polished,objective@x,atol=1e-5,rtol=1e-8):
                    x=polished
        residuals = dict(matrix=float((matrix@x-np.asarray(rhs)).max(initial=0)),
                         lower=float((low-x).max(initial=0)),upper=float((x-upper).max(initial=0)),
                         integrality=float(np.abs(x[:2*n]-np.rint(x[:2*n])).max(initial=0)))
        if (matrix@x-np.asarray(rhs)).max(initial=0) > TOL or (x < low-TOL).any() or (x > upper+TOL).any() or np.abs(x[:2*n]-np.rint(x[:2*n])).max(initial=0) > TOL:
            state, x = "NUMERICAL_ERROR", None
    record = dict(status=state, quantity=None, objective=None, mip_gap=getattr(result,"mip_gap",None),
                  dual_bound=None if getattr(result,"mip_dual_bound",None) is None else result.mip_dual_bound*cost_scale,runtime=perf_counter()-start,
                  oracle_solves=oracle_records,solver_attempts=attempts,message=str(result.message),residuals=residuals,raw_residuals=raw_residuals,
                  matrix=dict(shape=list(matrix.shape),nnz=matrix.nnz,cost_scale=cost_scale,sha256=hashlib.sha256(matrix.data.tobytes()+matrix.indices.tobytes()+matrix.indptr.tobytes()+objective.tobytes()+np.asarray(rhs).tobytes()+low.tobytes()+upper.tobytes()).hexdigest()))
    if x is None: return record
    q = np.rint(x[:n]*c.pack).astype(int)
    _, fair = verify(c,resources,q,fairness)
    preference = fairness.penalty*(sum(fair["minimum_shortfall"].values())+fair["disparity_slack"])
    # Pairwise disparity slack in the builder can contain several positive pairs.
    if fairness.soft:
        shares = list(fair["shares"].values())
        preference = fairness.penalty*(sum(fair["minimum_shortfall"].values())+
            sum(max(0,abs(a-b)-fairness.maximum_disparity) for i,a in enumerate(shares) for b in shares[i+1:]))
    raw = losses(q,stock,scenarios,cu,co); scores = tail(raw+preference,p,alpha)
    exact = scores["expected"]
    if method == "mean_cvar": exact += penalty*scores["cvar"]
    if method == "minimax": exact = float((raw+preference).max())
    if method == "regret": exact = float((raw+preference-oracle*cost_scale).max())
    if method == "dro": exact = worst_probability((raw+preference)[:w//2].mean(),(raw+preference)[w//2:].mean(),interval)["cost"]
    if not np.isclose((objective@x)*cost_scale,exact,atol=1e-5,rtol=1e-8):
        raise AssertionError("Risk objective disagrees with independently scored allocation")
    if method == "ceiling" and scores["cvar"] > risk_limit+1e-6:
        raise AssertionError("Hard CVaR ceiling violated")
    # Report the gap for the reconstructed feasible incumbent, not only the
    # solver's pre-reconstruction cost auxiliaries.
    if record["dual_bound"] is not None:
        verified_gap=max(0.,(exact-record["dual_bound"])/max(1.,abs(exact)))
        record["mip_gap"]=max(record["mip_gap"] or 0.,verified_gap)
        if record["status"]=="OPTIMAL" and record["mip_gap"]>TOL:
            record["status"]="FEASIBLE_NOT_PROVEN_OPTIMAL"
    record.update(quantity=q,objective=float((objective@x)*cost_scale),losses=raw.tolist(),preference_cost=preference,
                  scores=scores,oracle=None if oracle is None else (oracle*cost_scale).tolist(),
                  regrets=None if oracle is None else (raw+preference-oracle*cost_scale).tolist(),
                  eta=float(x[size]*cost_scale) if method in {"mean_cvar","ceiling"} else None,
                  excess=(x[size+1:]*cost_scale).tolist() if method in {"mean_cvar","ceiling"} else None,
                  activation=np.rint(x[n:2*n]).astype(int).tolist())
    return record
