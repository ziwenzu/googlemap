import numpy as np
from scipy.optimize import minimize
rng=np.random.default_rng(7)
y=np.arange(1,6.0)
PI=np.array([[.42,.22,.16,.12,.08],[.2]*5,[.08,.12,.16,.22,.42]])
RHO=np.array([[.12,.10,.08,.06,.04],[.04,.06,.08,.10,.12],[.04,.06,.08,.10,.12]])
s=PI*RHO; s/=s.sum(1,keepdims=True)
def f(x,c,d,G,L):
    # x: 15 params in [0,1]: common log-weights (scaled by log G), drift c, drift d (scaled by log L)
    wb=np.exp(x[:5]*np.log(G)); dc=np.exp(x[5:10]*np.log(L)) if L>1 else np.ones(5); dd=np.exp(x[10:]*np.log(L)) if L>1 else np.ones(5)
    pc=s[c]*wb*dc; pc/=pc.sum(); pd=s[d]*wb*dd; pd/=pd.sum()
    return y@pc-y@pd
for (c,d) in [(2,1),(1,0)]:
  for G,L in [(3,1.0),(3,1.5),(10,1.0),(10,1.25)]:
    best=9
    for _ in range(300):
        x0=rng.random(15)
        r=minimize(f,x0,args=(c,d,G,L),bounds=[(0,1)]*15,method='L-BFGS-B')
        best=min(best,r.fun)
    print(f"cells {c+1} vs {d+1}, Gamma={G}, Lam={L}: continuous multi-start min = {best:+.4f}")
