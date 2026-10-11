"""Population-level (N = infinity) identification width in the manuscript's prototype design.
Compares (a) the manuscript's summation-by-parts OUTER mean bounds with
(b) the SHARP within-cell mean bounds under the same bounded-ratio model
    (threshold-weight solution; linear-fractional program solved by enumeration + LP check)."""
import numpy as np
from scipy.optimize import linprog

PI = np.array([[.42, .22, .16, .12, .08],
               [.20, .20, .20, .20, .20],
               [.08, .12, .16, .22, .42]])
RHO = np.array([[.12, .10, .08, .06, .04],
                [.04, .06, .08, .10, .12],
                [.04, .06, .08, .10, .12]])
y = np.arange(1, 6.0)


def cdf_map(Q, gamma, eta, upper):
    if upper:
        S = min(1.0, Q / (1 - eta)); return gamma * S / (gamma * S + 1 - S)
    S = max(0.0, (Q - eta) / (1 - eta)); return S / (S + gamma * (1 - S))


def outer_mean(q, gamma, e):
    Q = np.cumsum(q)[:-1]
    lo = np.array([cdf_map(x, gamma, e, False) for x in Q])
    hi = np.array([cdf_map(x, gamma, e, True) for x in Q])
    d = np.diff(y)
    return y[-1] - (d * hi).sum(), y[-1] - (d * lo).sum()


def sharp_mean_clean(s, gamma):
    """min/max of sum y_h s_h w_h / sum s_h w_h over w in [1,gamma]^K (ratio-bounded weights).
    Optimum is a threshold rule; enumerate thresholds."""
    K = len(s); best_lo, best_hi = np.inf, -np.inf
    for k in range(K + 1):
        w = np.where(np.arange(K) < k, gamma, 1.0)      # heavy weight on low categories -> min mean
        best_lo = min(best_lo, (y * s * w).sum() / (s * w).sum())
        w = np.where(np.arange(K) < k, 1.0, gamma)      # heavy weight on high categories -> max mean
        best_hi = max(best_hi, (y * s * w).sum() / (s * w).sum())
    return best_lo, best_hi


def sharp_mean_contaminated(q, gamma, e, grid=None):
    """Sharp mean bounds with mixture contamination share <= e and ratio-bounded weights.
    For fixed weights w the problem min_g sum(y g w)/sum(g w), 0<=g<=q, sum g >= 1-e is a
    linear-fractional program -> Charnes-Cooper LP. Enumerate threshold weights."""
    K = len(q); res = []
    for sign in (+1, -1):
        best = np.inf
        for k in range(K + 1):
            w = np.where(np.arange(K) < k, gamma, 1.0) if sign > 0 else np.where(np.arange(K) < k, 1.0, gamma)
            # variables z (K) and tau; minimize sign * sum(y w z) s.t. sum(w z)=1, 0<=z<=q tau, sum z >= (1-e) tau, sum z <= tau
            c = np.append(sign * y * w, 0.0)
            A_eq = [np.append(w, 0.0)]; b_eq = [1.0]
            A_ub = []; b_ub = []
            for h in range(K):
                row = np.zeros(K + 1); row[h] = 1; row[-1] = -q[h]; A_ub.append(row); b_ub.append(0)
            row = np.append(-np.ones(K), (1 - e)); A_ub.append(row); b_ub.append(0)
            row = np.append(np.ones(K), -1.0); A_ub.append(row); b_ub.append(0)
            r = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * (K + 1))
            if r.status == 0:
                best = min(best, r.fun)
        res.append(sign * best)
    return res[0], res[1]


v = (PI * RHO).sum(1)
s = PI * RHO / v[:, None]

if __name__ == "__main__":
    print("latent means :", PI @ y)
    print("visible means:", s @ y)
    for gamma in (1.5, 2, 3):
        print(f"\n=== Gamma = {gamma}, clean, population level (no sampling error) ===")
        for c in range(3):
            o = outer_mean(s[c], gamma, 0); sh = sharp_mean_clean(s[c], gamma)
            print(f"cell {c+1}: outer [{o[0]:.3f},{o[1]:.3f}] width {o[1]-o[0]:.3f} | sharp [{sh[0]:.3f},{sh[1]:.3f}] width {sh[1]-sh[0]:.3f}")

    print("\n=== Gamma = 3, concentrated (e_c = .12), population level ===")
    eta = np.array([0, .12, 0]); f = np.array([[.2] * 5, [1, 0, 0, 0, 0], [.2] * 5])
    q = (1 - eta)[:, None] * s + eta[:, None] * f
    for c in range(3):
        o = outer_mean(q[c], 3, .12); sh = sharp_mean_contaminated(q[c], 3, .12)
        print(f"cell {c+1}: outer [{o[0]:.3f},{o[1]:.3f}] width {o[1]-o[0]:.3f} | sharp [{sh[0]:.3f},{sh[1]:.3f}] width {sh[1]-sh[0]:.3f}")
