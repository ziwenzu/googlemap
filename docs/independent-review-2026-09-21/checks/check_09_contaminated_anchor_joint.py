import numpy as np
Q1,Q2=.65,.5; cap=.2; cap0=.1; eps=1/15
def p1min(e1,r): return ((Q1-e1)/r)/((1-Q1)+(Q1-e1)/r)
def p2max(e2,r): return (Q2/r)/((1-Q2-e2)+Q2/r)
xs=np.linspace(0,cap0,101)
for J in (3,4,5,6,8,50):
    B=eps*J
    best=9
    for x in xs:                       # anchor contamination eta0=x, removed from category 2 (r<1) or category 1 (r>1)
        for r in ((.5-x)/.5, .5/(.5-x)):
            left=B-x
            if left<0: continue
            for e1 in np.linspace(0,min(cap,left),201):
                e2=min(cap,left-e1)
                best=min(best,p1min(e1,r)-p2max(e2,r))
    # product of marginals: each marginal optimised separately (own anchor configuration, own budget)
    m1=min(p1min(min(cap,B-x),r) for x in xs if B-x>=0 for r in ((.5-x)/.5,.5/(.5-x)))
    m2=max(p2max(min(cap,B-x),r) for x in xs if B-x>=0 for r in ((.5-x)/.5,.5/(.5-x)))
    print(f"J={J:3d}: joint sharp inf(p1-p2)={best:+.4f}   product of marginal intervals={m1-m2:+.4f}")
