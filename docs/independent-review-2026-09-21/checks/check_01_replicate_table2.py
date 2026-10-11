"""Independent Python re-implementation of the manuscript's prototype Monte Carlo
(simulation/run-prototype-study.R + packages/tracebounds/R/inference.R).
Different RNG, so numbers should agree only up to Monte Carlo error."""
import numpy as np
from scipy.stats import beta

PI = np.array([[.42, .22, .16, .12, .08],
               [.20, .20, .20, .20, .20],
               [.08, .12, .16, .22, .42]])
RHO = np.array([[.12, .10, .08, .06, .04],
                [.04, .06, .08, .10, .12],
                [.04, .06, .08, .10, .12]])
SCORES = np.arange(1, 6)
MU = PI @ SCORES


def cp(x, n, a):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return lo, hi


def cdf_map(Q, gamma, eta, upper):
    if upper:
        S = min(1.0, Q / (1 - eta))
        return gamma * S / (gamma * S + 1 - S)
    S = max(0.0, (Q - eta) / (1 - eta))
    return S / (S + gamma * (1 - S))


def fit(counts, gamma, cap, alpha=.05):
    C, K = counts.shape
    delta = alpha / (C * K)
    box = np.zeros((C, K, 2))
    cdf_lo = np.zeros((C, K - 1)); cdf_hi = np.zeros((C, K - 1))
    means = np.zeros((C, 2))
    for c in range(C):
        n = counts[c].sum()
        iv = np.array([cp(x, n, delta) for x in counts[c]])
        lo, hi = iv[:, 0], iv[:, 1]
        lo2 = np.maximum.reduce([lo, 1 - (hi.sum() - hi), np.zeros(K)])
        hi2 = np.minimum.reduce([hi, 1 - (lo.sum() - lo), np.ones(K)])
        box[c, :, 0], box[c, :, 1] = lo2, hi2
        for k in range(1, K):
            Qlo = max(lo2[:k].sum(), 1 - hi2[k:].sum(), 0)
            Qhi = min(hi2[:k].sum(), 1 - lo2[k:].sum(), 1)
            cdf_lo[c, k - 1] = max(0, cdf_map(Qlo, gamma, cap, False) - 1e-12)
            cdf_hi[c, k - 1] = min(1, cdf_map(Qhi, gamma, cap, True) + 1e-12)
        d = np.diff(SCORES)
        means[c, 0] = max(SCORES[0], SCORES[-1] - (d * cdf_hi[c]).sum())
        means[c, 1] = min(SCORES[-1], SCORES[-1] - (d * cdf_lo[c]).sum())
    return box, cdf_lo, cdf_hi, means


def run(expected, regime, B=1000, seed=1):
    rng = np.random.default_rng(seed)
    eta = np.zeros(3) if regime == "clean" else np.array([0, .12, 0])
    eta_cap = 0 if regime == "clean" else .15
    eps = 0 if regime == "clean" else .04
    v = (PI * RHO).sum(1)
    s = PI * RHO / v[:, None]
    f = np.array([[.2] * 5, [1, 0, 0, 0, 0], [.2] * 5])
    q = (1 - eta)[:, None] * s + eta[:, None] * f
    cap = min(eta_cap, eps * 3)
    env_lo = np.array([[cdf_map(Q, 3, cap, False) for Q in np.cumsum(q[c])[:4]] for c in range(3)])
    env_hi = np.array([[cdf_map(Q, 3, cap, True) for Q in np.cumsum(q[c])[:4]] for c in range(3)])
    truth = np.cumsum(PI, 1)[:, :4]
    out = []
    pair_res = np.zeros(3)
    for b in range(B):
        counts = rng.poisson(q * expected)
        box, lo, hi, m = fit(counts, 3, cap)
        rf = np.all((box[:, :, 0] <= q + 1e-12) & (box[:, :, 1] >= q - 1e-12))
        env = np.all((lo <= env_lo + 1e-12) & (hi >= env_hi - 1e-12))
        cov = np.all((lo <= truth + 1e-12) & (hi >= truth - 1e-12))
        better = 0; false = 0
        for ip, (c, d) in enumerate([(0, 1), (0, 2), (1, 2)]):
            for (a, bb) in [(c, d), (d, c)]:
                if m[a, 0] - m[bb, 1] > 1e-10:
                    better += 1
                    pair_res[ip] += 1
                    if MU[a] <= MU[bb]:
                        false = 1
        # naive
        _, _, _, mn = fit(counts, 1, 0)
        nfalse = 0
        for c in range(3):
            for d in range(3):
                if c != d and mn[c, 0] - mn[d, 1] > 1e-10 and MU[c] <= MU[d]:
                    nfalse = 1
        out.append([rf, env, cov, false, 1 - better / 3, (m[:, 1] - m[:, 0]).mean(), nfalse])
    out = np.array(out, float)
    return out.mean(0), pair_res / B


if __name__ == "__main__":
    print("design N | reduced env truth false abst width naive_false | pair resolution (1-2, 1-3, 2-3)")
    for regime in ["clean", "concentrated"]:
        for N in [25, 100, 500]:
            r, pr = run(N, regime, B=1000, seed=(1 if regime == 'clean' else 2) * 100000 + N)
            print(regime, N, np.round(r, 3), np.round(pr, 3))
