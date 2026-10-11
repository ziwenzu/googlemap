"""How much of the prototype's abstention is due to the summation-by-parts relaxation?
Replace the outer mean bound by the exact LP (within-cell sharp bound, projected over the same
Clopper-Pearson box ∩ simplex), keeping everything else identical to the manuscript's design."""
import numpy as np
from scipy.optimize import linprog
from scipy.stats import beta

PI = np.array([[.42, .22, .16, .12, .08], [.20] * 5, [.08, .12, .16, .22, .42]])
RHO = np.array([[.12, .10, .08, .06, .04], [.04, .06, .08, .10, .12], [.04, .06, .08, .10, .12]])
y = np.arange(1, 6.0); K = 5; MU = PI @ y


def cp(x, n, a):
    if n == 0: return 0.0, 1.0
    return (0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)), (1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x))


def lp_mean(lo, hi, gamma, e, sense):
    """variables: zq(K), zg(K), zy(K), tau ; minimize sense*sum(y*zy) s.t. sum(zy)=1,
    lo*tau<=zq<=hi*tau, sum zq = tau, 0<=zg<=zq, sum zg >= (1-e) tau, zg<=zy<=gamma*zg."""
    n = 3 * K + 1; c = np.zeros(n); c[2 * K:3 * K] = sense * y
    A, b = [], []
    def row(): return np.zeros(n)
    for h in range(K):
        r = row(); r[h] = -1; r[-1] = lo[h]; A.append(r); b.append(0)          # zq>=lo tau
        r = row(); r[h] = 1; r[-1] = -hi[h]; A.append(r); b.append(0)          # zq<=hi tau
        r = row(); r[K + h] = 1; r[h] = -1; A.append(r); b.append(0)           # zg<=zq
        r = row(); r[K + h] = 1; r[2 * K + h] = -1; A.append(r); b.append(0)   # zg<=zy
        r = row(); r[2 * K + h] = 1; r[K + h] = -gamma; A.append(r); b.append(0)  # zy<=gamma zg
    r = row(); r[K:2 * K] = -1; r[-1] = 1 - e; A.append(r); b.append(0)
    Aeq = [np.concatenate([np.zeros(2 * K), np.ones(K), [0]]), np.concatenate([np.ones(K), np.zeros(2 * K), [-1]])]
    res = linprog(c, A_ub=np.array(A), b_ub=b, A_eq=np.array(Aeq), b_eq=[1, 0], bounds=[(0, None)] * n, method="highs")
    return sense * res.fun


def cdf_map(Q, gamma, eta, upper):
    if upper:
        S = min(1.0, Q / (1 - eta)); return gamma * S / (gamma * S + 1 - S)
    S = max(0.0, (Q - eta) / (1 - eta)); return S / (S + gamma * (1 - S))


def run(expected, regime, B, seed):
    rng = np.random.default_rng(seed)
    eta = np.zeros(3) if regime == "clean" else np.array([0, .12, 0])
    cap = 0.0 if regime == "clean" else .12
    v = (PI * RHO).sum(1); s = PI * RHO / v[:, None]
    f = np.array([[.2] * 5, [1, 0, 0, 0, 0], [.2] * 5]); q = (1 - eta)[:, None] * s + eta[:, None] * f
    out = []
    for _ in range(B):
        counts = rng.poisson(q * expected); m_outer = np.zeros((3, 2)); m_sharp = np.zeros((3, 2))
        for c in range(3):
            n = counts[c].sum(); iv = np.array([cp(x, n, .05 / 15) for x in counts[c]])
            lo, hi = iv[:, 0], iv[:, 1]
            lo2 = np.maximum.reduce([lo, 1 - (hi.sum() - hi), np.zeros(K)]); hi2 = np.minimum.reduce([hi, 1 - (lo.sum() - lo), np.ones(K)])
            Fl = [cdf_map(max(lo2[:k].sum(), 1 - hi2[k:].sum(), 0), 3, cap, False) for k in range(1, K)]
            Fu = [cdf_map(min(hi2[:k].sum(), 1 - lo2[k:].sum(), 1), 3, cap, True) for k in range(1, K)]
            m_outer[c] = [5 - sum(Fu), 5 - sum(Fl)]
            m_sharp[c] = [lp_mean(lo2, hi2, 3, cap, +1), lp_mean(lo2, hi2, 3, cap, -1)]
        def summarize(m):
            res = 0; false = 0
            for c in range(3):
                for d in range(3):
                    if c != d and m[c, 0] - m[d, 1] > 1e-9:
                        res += 1; false |= MU[c] <= MU[d]
            return 1 - res / 3, (m[:, 1] - m[:, 0]).mean(), false
        out.append(summarize(m_outer) + summarize(m_sharp) + (np.all(m_sharp[:, 0] >= m_outer[:, 0] - 1e-7) and np.all(m_sharp[:, 1] <= m_outer[:, 1] + 1e-7),))
    return np.array(out, float).mean(0)


if __name__ == "__main__":
    print("regime N | OUTER: abstention width false | LP-SHARP(within cell): abstention width false | sharp inside outer")
    for regime in ("clean", "concentrated"):
        for N in (100, 500, 2000, 10000):
            print(regime, N, np.round(run(N, regime, 300, 11 + N), 3))
