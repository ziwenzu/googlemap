"""Is the informative-volume excess a distortion of the index scale (through psi-hat)?"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # vrindex.py lives one level up
import numpy as np, sys
from multiprocessing import Pool
from vrindex import RatingScaleEB, family_probs
NU = np.arange(1, 6.0); C = np.log([.05, .08, .17, .35, .35]); bJ = np.array([15, 2, 1, 1.5, 5.]); TAU = .45
PSI = C + np.log(bJ); PSI = PSI - PSI[0]

def resid(psi_hat):
    d = psi_hat - PSI
    A = np.column_stack([np.ones(5), NU])
    return d - A @ np.linalg.lstsq(A, d, rcond=None)[0]

def one(args):
    rep, endo = args
    rng = np.random.default_rng([41, rep]); J = 2000
    theta = TAU * rng.standard_normal(J)
    loud = np.exp(np.log(6) + 1.6 * rng.standard_normal(J))
    pi = family_probs(theta, C, NU); vis = pi * bJ; vis /= vis.sum(1, keepdims=True)
    g = (pi * bJ).sum(1) / (family_probs(np.zeros(1), C, NU)[0] * bJ).sum()
    N = rng.poisson(loud * (g if endo else 1.0)); X = np.array([rng.multinomial(n, v) for n, v in zip(N, vis)]); ok = N > 0
    m = RatingScaleEB().fit(X[ok], se=False)
    th = theta[ok]
    slope = np.polyfit(th, m.theta_, 1)[0]            # shrinkage-attenuated; compare across designs
    # un-shrunk slope among well-measured units
    big = X[ok].sum(1) >= 50
    mle = m.theta_mle(X[ok][big]); fin = np.isfinite(mle)
    slope_big = np.polyfit(th[big][fin], mle[fin], 1)[0]
    return dict(endo=endo, tau=m.tau_, resid=resid(m.psi_), slope_big=slope_big)

if __name__ == "__main__":
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    with Pool(2) as p:
        res = p.map(one, [(r, e) for e in (False, True) for r in range(R)])
    for e in (False, True):
        sel = [r for r in res if r["endo"] == e]
        print("endogenous" if e else "exogenous ", "tau %.4f" % np.mean([r["tau"] for r in sel]),
              "| slope of unshrunk index on truth (N>=50): %.4f" % np.mean([r["slope_big"] for r in sel]),
              "| nonlinear part of psi-hat minus psi:", np.round(np.mean([r["resid"] for r in sel], 0), 4))
