import numpy as np
from scipy.optimize import minimize
y=np.arange(1,6.0)
q0=np.array([.45,.04,.04,.12,.35]); pi0=np.array([.10,.10,.20,.30,.30]); q=np.array([.40,.08,.07,.15,.30])
def mu(g0):
    a=pi0/g0; p=a*q; p/=p.sum(); return y@p
rng=np.random.default_rng(1)
for etabar0 in (.05,):
  for kap in (.03,.01,.001,.0001):
    best=[9,-9]
    for sign,idx in ((1,0),(-1,1)):
        for _ in range(400):
            x0=q0-rng.dirichlet(np.ones(5)*.3)*etabar0*rng.random()
            x0=np.clip(x0,1e-9,q0)
            cons=[{'type':'ineq','fun':lambda g: g.sum()-(1-etabar0)}]+[{'type':'ineq','fun':(lambda g,h=h: g[h]-kap*g.sum())} for h in range(5)]
            r=minimize(lambda g: sign*mu(np.maximum(g,1e-12)),x0,bounds=[(1e-9,q0[h]) for h in range(5)],constraints=cons,method='SLSQP')
            if r.success:
                v=sign*r.fun
                best[idx]=min(best[idx],v) if sign==1 else max(best[idx],v)
    print(f"eta0cap={etabar0} kappa_s={kap}: target-mean range [{best[0]:.4f},{best[1]:.4f}]  (clean-anchor value {mu(q0):.4f})")
# eta0 cap below the thinnest category: kappa-free
for etabar0 in (.01,.02,.03,.035,.039):
    best=9
    for _ in range(300):
        x0=np.clip(q0-rng.dirichlet(np.ones(5)*.3)*etabar0*rng.random(),1e-9,q0)
        cons=[{'type':'ineq','fun':lambda g: g.sum()-(1-etabar0)}]
        r=minimize(lambda g: mu(np.maximum(g,1e-12)),x0,bounds=[(1e-9,q0[h]) for h in range(5)],constraints=cons,method='SLSQP')
        if r.success: best=min(best,r.fun)
    print(f"no kappa, eta0cap={etabar0}: lower bound of target mean = {best:.3f}")
