import numpy as np
from check_01_replicate_table2 import fit, PI, RHO
rng=np.random.default_rng(3)
v=(PI*RHO).sum(1); q=PI*RHO/v[:,None]
for N in (5,40,100,500):
  for Gam in (1,2,3,5):
    w=[]
    for b in range(300):
        counts=rng.poisson(q*N)
        _,_,_,m=fit(counts,Gam,0.0)
        w.append((m[:,1]-m[:,0]).mean())
    print(f"N={N:4d} Gamma={Gam}: mean width of mean interval (1-5 scale) = {np.mean(w):.2f}")
# many-cell Bonferroni effect: same cell, but alpha split over J cells
from scipy.stats import beta
def fit_alpha(counts,gamma,alpha_per_margin):
    import check_01_replicate_table2 as r
    C,K=counts.shape
    out=[]
    for c in range(C):
        n=counts[c].sum()
        iv=np.array([r.cp(x,n,alpha_per_margin) for x in counts[c]])
        lo,hi=iv[:,0],iv[:,1]
        lo2=np.maximum.reduce([lo,1-(hi.sum()-hi),np.zeros(K)]); hi2=np.minimum.reduce([hi,1-(lo.sum()-lo),np.ones(K)])
        Fl=[r.cdf_map(max(lo2[:k].sum(),1-hi2[k:].sum(),0),gamma,0,False) for k in range(1,K)]
        Fu=[r.cdf_map(min(hi2[:k].sum(),1-lo2[k:].sum(),1),gamma,0,True) for k in range(1,K)]
        out.append((5-sum(Fu),5-sum(Fl)))
    return np.array(out)
print()
for J in (3,100,1000,10000):
  for N in (40,500):
    w=[]
    for b in range(200):
        counts=rng.poisson(q*N)
        m=fit_alpha(counts,1.0,.05/(J*5))
        w.append((m[:,1]-m[:,0]).mean())
    print(f"J={J:6d} cells, N={N}: sampling-only (Gamma=1) mean width = {np.mean(w):.2f}")
