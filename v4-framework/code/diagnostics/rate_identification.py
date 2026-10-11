"""Does the rating RATE (ratings per encounter), regressed on the index, identify the voice function?
Requires known encounter volumes U_j and loudness a_j unrelated to theta_j."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # vrindex.py lives one level up
import numpy as np
from scipy.optimize import least_squares
from vrindex import family_probs
rng=np.random.default_rng(3); nu=np.arange(1,6.)
c=np.log(np.array([.05,.08,.17,.35,.35])); b=np.array([15,2,1,1.5,5.])*1e-3
for J,sd_a in ((3000,0.5),(30000,0.5),(30000,1.0)):
    theta=0.45*rng.standard_normal(J); U=rng.integers(20000,400000,J).astype(float); a=np.exp(sd_a*rng.standard_normal(J)-sd_a**2/2)
    pi=family_probs(theta,c,nu); lam=U[:,None]*pi*a[:,None]*b[None,:]; X=rng.poisson(lam); N=X.sum(1)
    psi=c+np.log(b); psi=psi-psi[0]
    rate=N/U
    P=np.exp(psi[None,:]+theta[:,None]*nu[None,:]).sum(1)
    def resid(par):
        A=np.exp(par[0]); cc=np.concatenate([[0.],par[1:]])
        Z=np.exp(cc[None,:]+theta[:,None]*nu[None,:]).sum(1)
        return np.log(rate+1e-12)-np.log(A*P/Z)
    r=least_squares(resid, np.concatenate([[np.log(rate.mean())],psi[1:]]))   # start from "no selection": c = psi
    c_hat=np.concatenate([[0.],r.x[1:]]); b_hat=np.exp(psi-c_hat); b_hat/=b_hat[2]
    print(f"J={J}, sd(log a)={sd_a}: true b/b3 = {np.round(b/b[2],2)} ; estimated = {np.round(b_hat,2)}")
