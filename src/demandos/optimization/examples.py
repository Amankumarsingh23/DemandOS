"""Exact SYNTHETIC hand models, brute-force reference checks and EVPI."""
from itertools import product
import numpy as np
from .model import Candidates,Resources,Fairness,basic_lp,solve,build


def candidates(request,purchase,benefit,*,pack=None,moq=None,minimum=None,stores=None):
    n=len(request)
    return Candidates(tuple(chr(65+i) for i in range(n)),np.array(request),np.array(purchase),np.ones(n),np.array(benefit),
        tuple("supplier" for _ in range(n)),tuple(stores or ["store"]*n),np.zeros(n),np.array(pack or [1]*n),np.array(moq or [0]*n),np.array(minimum or [0]*n))


def run_examples():
    a=np.array([[4,2],[1,1]],float);b=np.array([20,8],float)
    lp=basic_lp([8,5],a,b);cap=basic_lp([8,5],np.r_[a,[[0,1]]],np.r_[b,4])
    np.testing.assert_allclose(lp["x"],[2,6]);np.testing.assert_allclose(lp["shadow_prices"],[1.5,2])
    assert abs(lp["benefit"]-46)<1e-7 and abs(cap["benefit"]-44)<1e-7
    np.testing.assert_allclose(cap["x"],[3,4])
    # Enumerate vertices from every pair of active boundaries, including axes.
    boundaries=np.r_[a,[[1,0],[0,1]]];limits=np.r_[b,0,0];vertices=[]
    for i in range(4):
        for j in range(i+1,4):
            mat=boundaries[[i,j]]
            if abs(np.linalg.det(mat))<1e-12:continue
            q=np.linalg.solve(mat,limits[[i,j]])
            if min(q)>=-1e-9 and np.all(a@q<=b+1e-9):vertices.append(float(np.array([8,5])@q))
    assert abs(max(vertices)-46)<1e-7
    c=candidates([1,1,1],[6,5,4],[10,9,7]);r=Resources(10,100,{"supplier":100})
    knapsack=solve(c,r);np.testing.assert_array_equal(knapsack["quantity"],[1,0,1]);assert abs(knapsack["objective"]+17)<1e-7
    brute=max(np.array([10,9,7])@x for x in product([0,1],repeat=3) if np.array([6,5,4])@x<=10)
    assert brute==17
    c2=candidates([1,1],[6,5],[10,9]);integer=solve(c2,r);relaxed=solve(c2,r,relax=True)
    assert abs(integer["objective"]+10)<1e-7 and abs(relaxed["objective"]+17.3333333333333)<1e-7
    assert 6+5>10
    infeasible=solve(candidates([10,10],[1,1],[1,1],minimum=[5,5]),Resources(100,8,{"supplier":100}))
    assert infeasible["status"]=="INFEASIBLE" and infeasible["quantity"] is None
    cc=candidates([10,10],[1,1],[1,1]);rr=Resources(10,100,{"supplier":100});d=np.array([[0,10],[10,0]])
    shared=solve(cc,rr,demand=d,probabilities=[.5,.5],stock=[0,0],cu=9,co=1)
    perfect=[solve(cc,rr,demand=d[:,i:i+1],probabilities=[1.],stock=[0,0],cu=9,co=1)["objective"] for i in range(2)]
    evpi=shared["objective"]-np.mean(perfect);assert evpi>=-1e-7 and abs(evpi-50)<1e-7
    more=solve(c2,Resources(11,100,{"supplier":100}))
    return dict(label="EXACT_SYNTHETIC_SCENARIO",lp=dict(quantities=lp["x"].tolist(),value=lp["benefit"],duals=lp["shadow_prices"].tolist(),dual_value=float(lp["shadow_prices"]@b),supplier_cap_quantities=cap["x"].tolist(),supplier_cap_value=cap["benefit"]),
        knapsack=dict(quantity=knapsack["quantity"].tolist(),value=-knapsack["objective"],brute_force=brute),
        integrality=dict(integer_value=-integer["objective"],relaxed_value=-relaxed["objective"],gap=integer["objective"]-relaxed["objective"],integer_budget_plus_1_benefit=-more["objective"]),
        infeasible=infeasible["status"],evpi=dict(nonanticipative=shared["objective"],perfect=float(np.mean(perfect)),evpi=evpi),
        matrix=build(cc,rr,demand=d,probabilities=[.5,.5],stock=[0,0],cu=9,co=1))
