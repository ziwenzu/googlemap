"""Independent numerical checks of the manuscript's formal claims and of the reviewer's new claims."""
import numpy as np
from scipy.optimize import linprog, minimize

rng = np.random.default_rng(20260921)
y5 = np.arange(1, 6.0)

# ---------------------------------------------------------------------------
# 1. Theorem 1 (known-rate bounds) – brute force
# ---------------------------------------------------------------------------
def check_thm1():
    s = np.array([.25, .25, .5]); v = .1; y = np.array([1., 2., 3.])
    lo, hi = v * (y @ s) + (1 - v) * y[0], v * (y @ s) + (1 - v) * y[-1]
    # random feasible pi: pi = v s + (1-v) G ; verify rho=v s/pi in (0,1]
    ok = True; mn, mx = 9, -9
    for _ in range(20000):
        G = rng.dirichlet(np.ones(3) * .3); pi = v * s + (1 - v) * G
        rho = v * s / pi; ok &= np.all((rho > 0) & (rho <= 1 + 1e-12))
        mn = min(mn, y @ pi); mx = max(mx, y @ pi)
    print(f"[Thm1] formula bounds [{lo:.4f},{hi:.4f}]  brute-force range [{mn:.4f},{mx:.4f}]  rho valid: {ok}")

# ---------------------------------------------------------------------------
# 2. Eq. (14) clean unanchored CDF bounds + contaminated version – brute force
# ---------------------------------------------------------------------------
def cdf_map(Q, gamma, eta, upper):
    if upper:
        S = min(1.0, Q / (1 - eta)); return gamma * S / (gamma * S + 1 - S)
    S = max(0.0, (Q - eta) / (1 - eta)); return S / (S + gamma * (1 - S))

def check_cdf_bounds():
    q = np.array([.1, .15, .2, .25, .3]); gamma = 3.0; e = .12
    worst = 0
    for k in range(1, 5):
        Q = q[:k].sum(); lo, hi = cdf_map(Q, gamma, e, False), cdf_map(Q, gamma, e, True)
        mn, mx = 1, 0
        for _ in range(40000):
            eta = rng.uniform(0, e)
            # random genuine mass g<=q with sum 1-eta: remove contamination mass eta from q
            rem = rng.dirichlet(np.ones(5) * .2) * eta
            if np.any(rem > q):
                continue
            g = q - rem
            w = np.exp(rng.uniform(0, np.log(gamma), 5))
            if rng.random() < .5:   # also extreme weights
                w = np.where(rng.random(5) < .5, 1.0, gamma)
            p = g * w / (g * w).sum(); F = p[:k].sum()
            mn, mx = min(mn, F), max(mx, F)
        # attaining allocations
        worst = max(worst, lo - mn, mx - hi)
        print(f"[Eq14+contam] k={k}: formula [{lo:.4f},{hi:.4f}] brute [{mn:.4f},{mx:.4f}]")
    print(f"   max violation of formula by brute force: {worst:.2e} (<=0 means formula is a valid outer bound)")

# ---------------------------------------------------------------------------
# 3. Binary shared-budget proposition – brute force of inf(p1-p2)
# ---------------------------------------------------------------------------
def check_binary(Q1=.65, Q2=.5, b=.2):
    best = 9
    for e1 in np.linspace(0, b, 2001):
        e2 = b - e1
        p1 = max(0, (Q1 - e1) / (1 - e1)); p2 = min(1, Q2 / (1 - e2))
        best = min(best, p1 - p2)
    formula = min(Q1 - Q2 / (1 - b), (Q1 - b) / (1 - b) - Q2)
    print(f"[Binary] formula inf(p1-p2)={formula:.5f}; brute-force over budget splits={best:.5f}; "
          f"product-of-marginals={(Q1-b)/(1-b) - Q2/(1-b):.5f}")

# ---------------------------------------------------------------------------
# 4. Reviewer claim A: the shared budget stops binding as the number of cells grows
# ---------------------------------------------------------------------------
def budget_scaling(Q1=.65, Q2=.5, eps=1/15, cap=.2):
    print("[Budget scaling] eps_G fixed at 1/15, local caps .2, equal weights; J = # cells in the domain (incl. clean anchor)")
    for J in (3, 4, 5, 6, 8, 12, 50, 1000):
        B = eps * J                      # total contamination mass available (in cell units)
        best = 9
        for e1 in np.linspace(0, min(cap, B), 801):
            e2 = min(cap, B - e1)
            p1 = (Q1 - e1) / (1 - e1); p2 = Q2 / (1 - e2)
            best = min(best, p1 - p2)
        print(f"   J={J:5d}: total budget={B:6.3f}  sharp inf(p1-p2)={best:+.4f}")

# ---------------------------------------------------------------------------
# 5. Reviewer claim C: drift + contamination single-cell bounds are an LP (known clean anchor)
# ---------------------------------------------------------------------------
def lp_anchored(q, a, Lam, e, y, sense=+1, q_box=None):
    """min (sense=+1) or max (sense=-1) of mean over {pi ∝ a*yv : 0<=g<=q, sum g>=1-e, g<=yv<=Lam g}.
    Charnes–Cooper: z_g = tau g, z_y = tau yv, sum(a z_y)=1."""
    K = len(q)
    # variables: z_g (K), z_y (K), tau
    n = 2 * K + 1
    c = np.zeros(n); c[K:2 * K] = sense * y * a
    A_ub, b_ub = [], []
    for h in range(K):
        r = np.zeros(n); r[h] = 1; r[-1] = -q[h]; A_ub.append(r); b_ub.append(0)       # g<=q
        r = np.zeros(n); r[h] = 1; r[K + h] = -1; A_ub.append(r); b_ub.append(0)        # g<=yv
        r = np.zeros(n); r[K + h] = 1; r[h] = -Lam; A_ub.append(r); b_ub.append(0)      # yv<=Lam g
    r = np.zeros(n); r[:K] = -1; r[-1] = (1 - e); A_ub.append(r); b_ub.append(0)        # sum g >= 1-e
    r = np.zeros(n); r[:K] = 1; r[-1] = -1; A_ub.append(r); b_ub.append(0)              # sum g <= 1
    A_eq = [np.concatenate([np.zeros(K), a, [0]])]; b_eq = [1]
    res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=np.array(A_eq), b_eq=b_eq, bounds=[(0, None)] * n)
    return sense * res.fun

def brute_anchored(q, a, Lam, e, y, sense=+1, n=200000):
    K = len(q); best = np.inf
    for _ in range(n):
        eta = e * rng.random() ** .3
        rem = rng.dirichlet(np.ones(K) * rng.choice([.05, .2, 1.])) * eta
        if np.any(rem > q):
            continue
        g = q - rem
        d = np.where(rng.random(K) < .5, 1.0, Lam) if rng.random() < .7 else np.exp(rng.uniform(0, np.log(Lam), K))
        p = a * g * d; p /= p.sum()
        best = min(best, sense * (y @ p))
    return sense * best

def check_lp():
    q = np.array([.40, .08, .07, .15, .30])          # J-shaped visible distribution
    pi0 = np.array([.10, .10, .20, .30, .30]); s0 = np.array([.35, .08, .07, .15, .35])
    a = pi0 / s0                                     # inverse selection weights from a clean anchor
    for Lam, e in [(1.0, 0.0), (1.5, 0.0), (1.5, .05), (2.0, .10)]:
        lo, hi = lp_anchored(q, a, Lam, e, y5, +1), lp_anchored(q, a, Lam, e, y5, -1)
        blo, bhi = brute_anchored(q, a, Lam, e, y5, +1), brute_anchored(q, a, Lam, e, y5, -1)
        print(f"[LP anchored] Lam={Lam}, e={e}: LP mean bounds [{lo:.4f},{hi:.4f}]  random-search [{blo:.4f},{bhi:.4f}]"
              f"  (random search must lie inside LP interval: {blo >= lo - 1e-9 and bhi <= hi + 1e-9})")

# (Claim B on kappa_s is checked by full optimisation in check_06_kappa_overlap.py)

# ---------------------------------------------------------------------------
# 7. Reviewer claim: exact/bounded separability identifies orderings that marginal bounds never can
# ---------------------------------------------------------------------------
def common_shape():
    PI = np.array([[.42, .22, .16, .12, .08], [.20] * 5, [.08, .12, .16, .22, .42]])
    RHO = np.array([[.12, .10, .08, .06, .04], [.04, .06, .08, .10, .12], [.04, .06, .08, .10, .12]])
    s = PI * RHO; s /= s.sum(1, keepdims=True)
    def contrast(logw, c, d, Lam):
        # common inverse-selection weights w (log-range<=log Gamma enforced by caller through bounds),
        # pair-specific drift handled by worst-case threshold search below
        w = np.exp(logw); pc = s[c] * w; pc /= pc.sum(); pd_ = s[d] * w; pd_ /= pd_.sum()
        return y5 @ pc - y5 @ pd_
    for (c, d) in [(2, 1), (1, 0)]:
        for Gamma in (3.0, 10.0, 100.0):
            for Lam in (1.0, 1.25, 1.5, 2.0):
                best = np.inf
                # enumerate extreme points: common weights in {1,Gamma}^K, drift weights in {1,Lam}^K on each cell
                from itertools import product
                for wb in product([1.0, Gamma], repeat=5):
                    wb = np.array(wb)
                    # worst case drift: lower mu_c (threshold: heavy low cats), raise mu_d (heavy high cats)
                    lo_c = min((y5 @ (s[c] * wb * np.where(np.arange(5) < k, Lam, 1.0))) / (s[c] * wb * np.where(np.arange(5) < k, Lam, 1.0)).sum() for k in range(6))
                    hi_d = max((y5 @ (s[d] * wb * np.where(np.arange(5) < k, 1.0, Lam))) / (s[d] * wb * np.where(np.arange(5) < k, 1.0, Lam)).sum() for k in range(6))
                    best = min(best, lo_c - hi_d)
                print(f"[Common shape] cells {c+1} vs {d+1}: Gamma={Gamma:5.0f}, pair drift Lam={Lam:4.2f}: "
                      f"lower bound on mu_{c+1}-mu_{d+1} (vertex enumeration) = {best:+.3f}")
            print()

if __name__ == "__main__":
    check_thm1(); print()
    check_cdf_bounds(); print()
    check_binary(); check_binary(.5, .5, .2); print()
    budget_scaling(); print()
    check_lp(); print()
    common_shape()
