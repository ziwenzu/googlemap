"""Numerical verification of the formal claims in the manuscript (population level, no sampling noise)."""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logsumexp
from vrindex import (family_probs, mean_score, _invert_mean, drift_interval, breakdown, contamination_interval, _extreme_mean,
                     profile_drift_set, min_drift, hilbert_distance, display_fixed_point, _influence_rating, simulate_sequential,
                     exchangeability_test, source_consistency, simulate_class, RatingScaleEB)

rng = np.random.default_rng(11)
nu = np.arange(1, 6.0)
c = np.log(np.array([.05, .08, .17, .35, .35]))
print("=== Claim 1: levels are not identified; every tilt b reproduces the same visible table ===")
theta = np.array([-.8, -.2, .1, .6])
pi = family_probs(theta, c, nu)
for b in (np.ones(5), np.array([6, 3, 1.5, 1, 1.]), np.array([15, 2, 1, 1.5, 5.])):
    vis = pi * b; vis /= vis.sum(1, keepdims=True)
    # an analyst who (wrongly) believes b' recovers pi' = normalise(vis / b'); all are compatible with the data
    for bprime in (np.ones(5), np.array([3, 2, 1, 1, 2.])):
        pi2 = vis / bprime; pi2 /= pi2.sum(1, keepdims=True)
        vis2 = pi2 * bprime; vis2 /= vis2.sum(1, keepdims=True)
        assert np.allclose(vis, vis2)
    print(" b=", b, " latent means", np.round(pi @ nu, 3), " visible means", np.round(vis @ nu, 3))

print("\n=== Claim 2: odds-ratio structure and the rating-scale index are invariant to common voice ===")
def local_log_or(P):  # log odds ratios of adjacent rows/columns
    L = np.log(P); return (L[1:, 1:] - L[1:, :-1]) - (L[:-1, 1:] - L[:-1, :-1])
for b in (np.array([6, 3, 1.5, 1, 1.]), np.array([15, 2, 1, 1.5, 5.]), rng.uniform(.2, 9, 5)):
    a = rng.uniform(.001, .01, len(theta))                      # unit-specific loudness
    U = rng.uniform(1e4, 1e6, len(theta))                       # unknown encounter volumes
    lam = U[:, None] * pi * a[:, None] * b[None, :]              # expected rating-event counts
    assert np.allclose(local_log_or(lam), local_log_or(pi))
    vis = lam / lam.sum(1, keepdims=True)
    psi = np.log(vis[0]) - theta[0] * nu; psi -= psi[0]          # category effects implied by unit 1
    th_vis = np.array([_invert_mean(v @ nu, psi, nu) for v in vis])
    print(" b=", np.round(b, 2), " index from visible data - latent theta =", np.round(th_vis - theta, 10))

print("\n=== Claim 2b: common aggregators are NOT invariant (same latent units, three voice functions) ===")
theta2 = np.array([-.5, 0., .5])
pi2 = family_probs(theta2, c, nu)
def cumlogit_effect(P):
    """location differences from a cumulative-logit fit to each row (population ML, common thresholds)."""
    def nll(par):
        kap = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:4])])); loc = np.concatenate([[0.], par[4:]])
        cdf = expit(kap[None, :] - loc[:, None]); cdf = np.concatenate([np.zeros((3, 1)), cdf, np.ones((3, 1))], 1)
        return -(P * np.log(np.clip(np.diff(cdf, axis=1), 1e-300, None))).sum()
    r = minimize(nll, np.array([-2, 0, 0, 0, 0, 0.]), method="BFGS"); return r.x[4:]
for name, b in [("no selection", np.ones(5)), ("negativity", np.array([6, 3, 1.5, 1, 1.])), ("J-shaped", np.array([15, 2, 1, 1.5, 5.]))]:
    vis = pi2 * b; vis /= vis.sum(1, keepdims=True)
    psi = np.log(vis[1]); psi -= psi[0]
    th = np.array([_invert_mean(v @ nu, psi, nu) for v in vis])
    print(f" {name:13s}: index diffs {np.round(np.diff(th), 3)} | mean-star diffs {np.round(np.diff(vis @ nu), 3)} | "
          f"top-2-box diffs {np.round(np.diff(vis[:, 3:].sum(1)), 3)} | cumulative-logit diffs {np.round(np.diff(np.concatenate([[0], cumlogit_effect(vis)])), 3)}")

print("\n=== Claim 2c: the same holds when the latent law is exactly cumulative-logit ===")
kap0 = np.array([-2.9, -1.9, -0.9, 0.6]); loc0 = np.array([0., .5, 1.])
cdf0 = expit(kap0[None, :] - loc0[:, None]); cdf0 = np.concatenate([np.zeros((3, 1)), cdf0, np.ones((3, 1))], 1)
pi_cl = np.diff(cdf0, axis=1)
for name, b in [("no selection", np.ones(5)), ("negativity", np.array([6, 3, 1.5, 1, 1.])), ("J-shaped", np.array([15, 2, 1, 1.5, 5.]))]:
    vis = pi_cl * b; vis /= vis.sum(1, keepdims=True)
    print(f" {name:13s}: fitted cumulative-logit location diffs {np.round(np.diff(np.concatenate([[0], cumlogit_effect(vis)])), 3)} (latent truth: 0.5, 0.5)")

print("\n=== Claim 3: mean-based drift interval is valid and sharp (given the mean score) ===")
psi = c + np.log(np.array([15, 2, 1, 1.5, 5.])); psi -= psi[0]
worst = 0; attained_lo = attained_hi = np.inf
for rep in range(4000):
    th0 = rng.normal(0, .8); Lam = rng.choice([1.25, 1.5, 2., 3.])
    kind = rng.integers(3)
    if kind == 0: d = rng.uniform(0, np.log(Lam), 5)
    elif kind == 1: d = np.where(np.arange(5) >= rng.integers(0, 6), np.log(Lam), 0.)
    else: d = np.where(np.arange(5) < rng.integers(0, 6), np.log(Lam), 0.)
    vis = family_probs(th0, psi, nu) * np.exp(d); vis /= vis.sum()
    th_tilde = _invert_mean(vis @ nu, psi, nu)                   # what the (population) estimator returns
    lo, hi = drift_interval(th_tilde, psi, nu, Lam)
    worst = max(worst, lo - th0, th0 - hi)
    if kind == 1: attained_lo = min(attained_lo, abs(lo - th0))
    if kind == 2: attained_hi = min(attained_hi, abs(hi - th0))
print(f" max violation over 4000 random drifts: {worst:.2e} (<= 0 means the interval always covers the truth)")
print(f" closest approach of truth to the lower / upper endpoint under threshold drifts: {attained_lo:.2e} / {attained_hi:.2e} (sharpness)")
for Lam in (1.5, 2., 3.):
    lo, hi = drift_interval(0., psi, nu, Lam)
    print(f" Lambda={Lam}: interval around theta_hat=0 is [{lo:.3f},{hi:.3f}]; rule of thumb ±log(Lam)/(K-1) = ±{np.log(Lam)/4:.3f}")
psi_flat = np.zeros(5)
lo, hi = drift_interval(0., psi_flat, nu, 2.)
print(f" (uniform visible distribution, Lambda=2: [{lo:.3f},{hi:.3f}] vs rule of thumb ±{np.log(2)/4:.3f})")

print("\n=== Claim 3b: profile-based identified set (uses the whole visible profile) ===")
def profile_set(vis, psi, Lam, grid=np.linspace(-6, 6, 120001)):
    e = np.log(vis) - psi
    r = (e[None, :] - grid[:, None] * nu[None, :]); osc = r.max(1) - r.min(1)
    ok = grid[osc <= np.log(Lam) + 1e-12]
    return (ok.min(), ok.max()) if len(ok) else (np.nan, np.nan)
vis = family_probs(.3, psi, nu)
for Lam in (1.5, 2.):
    print(f" exact family, Lambda={Lam}: profile set {np.round(profile_set(vis, psi, Lam), 4)}  closed form 0.3 ± {np.log(Lam)/4:.4f}")
d = np.where(np.arange(5) >= 3, np.log(2.), 0.); vis_d = family_probs(.3, psi, nu) * np.exp(d); vis_d /= vis_d.sum()
print(f" step drift (osc=log 2): profile set at Lambda=2 {np.round(profile_set(vis_d, psi, 2.), 4)} contains truth 0.3; "
      f"mean-based interval {np.round(drift_interval(_invert_mean(vis_d @ nu, psi, nu), psi, nu, 2.), 4)}")

for Lam in (1.2, 1.5, 2.):
    print(f"   library profile_drift_set at Lambda={Lam}: {profile_drift_set(vis_d, psi, nu, Lam)} (None = empty: the unit's own drift must exceed Lambda)")
print(f"   identified lower bound on this unit's drift: {min_drift(vis_d, psi, nu)[0]:.4f} (true oscillation: 2; a linear component is absorbed by theta)")
g_ = np.round(profile_set(vis_d, psi, 2.), 4); l_ = np.round(profile_drift_set(vis_d, psi, nu, 2.), 4)
assert np.allclose(g_, l_, atol=2e-4), (g_, l_)

print("\n=== Claim 3c: half-width of the mean-based interval relative to the rule of thumb log(Lambda)/(K-1) ===")
def kappa(th):
    p = family_probs(th, psi, nu); m = p @ nu; var = p @ (nu - m) ** 2
    return max(((nu - m) * p)[k:].sum() for k in range(1, 5)) / var
for th in (-1., -.5, 0., .5, 1.):
    lo, hi = drift_interval(th, psi, nu, 2.)
    print(f" theta_tilde={th:+.1f}: half-widths {th - lo:.3f}/{hi - th:.3f}; first-order kappa*log2 = {kappa(th) * np.log(2):.3f}; rule of thumb {np.log(2) / 4:.3f}")
ks = []
for rep in range(20000):
    p = rng.dirichlet(np.ones(5) * rng.choice([.3, 1., 3.])); m = p @ nu; var = p @ (nu - m) ** 2
    ks.append(max(((nu - m) * p)[k:].sum() for k in range(1, 5)) / var)
print(f" kappa over 20000 random five-point distributions: min {min(ks):.4f} (>= 1/4), max {max(ks):.4f} (<= 1)")
assert min(ks) >= .25 - 1e-9 and max(ks) <= 1 + 1e-9

print("\n=== Claim 4: breakdown value (exp{(K-1)|diff|/2} is an upper bound, close for central pairs) ===")
for tj, tk in [(.125, -.125), (.25, 0.), (.25, -.25), (.5, 0.), (.75, .25), (1., 0.), (.3, -.3)]:
    kk = kappa(tj) + kappa(tk)
    print(f" theta_j={tj}, theta_k={tk}: breakdown Lambda* = {breakdown(tj, tk, psi, nu):.3f};  first-order exp(|diff|/(kappa_j+kappa_k)) = {np.exp(abs(tj - tk) / kk):.3f};"
          f"  upper bound exp((K-1)|diff|/2) = {np.exp(4 * abs(tj - tk) / 2):.3f}")
worst = -np.inf
for rep in range(300):
    ps = rng.normal(0, 1.2, 5); ps -= ps[0]; tj, tk = rng.normal(0, .7, 2)
    if abs(tj - tk) < 1e-3: continue
    worst = max(worst, np.log(breakdown(tj, tk, ps, nu)) - 2 * abs(tj - tk))
print(f" max over 300 random (psi, theta_j, theta_k) of log Lambda* - 2|diff| = {worst:.4f} (<= 0: upper bound holds)")
assert worst <= 1e-6

print("\n=== Claim 5: contamination interval is valid ===")
worst = 0
for rep in range(3000):
    th0 = rng.normal(0, .8); eta = rng.uniform(0, .3); f = rng.dirichlet(np.ones(5) * .3)
    vis = (1 - eta) * family_probs(th0, psi, nu) + eta * f
    lo, hi = contamination_interval(_invert_mean(vis @ nu, psi, nu), psi, nu, eta)
    worst = max(worst, lo - th0, th0 - hi)
print(f" max violation over 3000 random contaminations: {worst:.2e}")
print(f" truncation: theta_tilde=0.5, eps=0.2 -> {contamination_interval(.5, psi, nu, .2)} (upper bound on the genuine mean leaves the scale)")
# with the whole histogram the identified set is {theta: (1-eps) p_h(theta) <= q_h for all h}, a subset of the mean-based interval
grid = np.linspace(-3, 3, 60001)
for th0, eps in ((0., .2), (0., .1), (-.5, .2)):
    qv = family_probs(th0, psi, nu); P = family_probs(grid, psi, nu)
    ok = grid[((1 - eps) * P <= qv[None, :] + 1e-15).all(1)]
    mlo, mhi = contamination_interval(th0, psi, nu, eps)
    print(f" theta_tilde={th0:+.1f}, eps={eps}: mean-based [{mlo:.3f},{mhi:.3f}]; using the histogram [{ok.min():.3f},{ok.max():.3f}]")
    assert ok.min() >= mlo - 1e-3 and ok.max() <= mhi + 1e-3

print("\n=== Claim 6: the index can be computed from a unit's rating count and (rounded) average alone ===")
from vrindex import RatingScaleEB, simulate_class
rng2 = np.random.default_rng(77)
d = simulate_class(1500, rng2, b=np.array([15, 2, 1, 1.5, 5.]))
X = d["X"]; X = X[X.sum(1) > 0].astype(float)
fit = RatingScaleEB().fit(X)
N = X.sum(1); avg = np.round((X @ nu) / N, 1)                       # what a platform displays
psi2 = np.atleast_2d(fit.psi_)
post = fit._posterior(psi2, np.zeros(len(N)), fit.tau_, X[:, None, :], N[:, None], (N * avg)[:, None])
print(f" max |theta(full histogram) - theta(count, average rounded to 0.1)| = {np.max(np.abs(post['mean'] - fit.theta_)):.4f};"
      f" correlation = {np.corrcoef(post['mean'], fit.theta_)[0, 1]:.6f}")

print("\n=== Claim 7: with unit-specific polarisation, the ordering of latent means is not identified for every pair ===")
kap = (nu - 3.0) ** 2
def two_par(c_, th, ph):
    e = c_ + th * nu + ph * kap; p = np.exp(e - logsumexp(e)); return p
cA = np.log(np.array([.05, .08, .17, .35, .35]))
units = {"moderate unit": (0.15, -0.35), "polarised unit": (0.0, 0.35)}
vis = {}
bJ = np.array([15, 2, 1, 1.5, 5.])
for k_, (th, ph) in units.items():
    p = two_par(cA, th, ph); v = p * bJ; vis[k_] = v / v.sum()
print(" visible means:", {k_: round(float(v @ nu), 3) for k_, v in vis.items()})
for name, bb in (("true voice (J-shaped)", bJ), ("analyst believes: no selection", np.ones(5)), ("analyst believes: negativity", np.array([6, 3, 1.5, 1, 1.]))):
    lat = {k_: (v / bb) / (v / bb).sum() for k_, v in vis.items()}
    print(f" {name:32s} implied latent means:", {k_: round(float(p @ nu), 3) for k_, p in lat.items()})
print(" (theta, phi) contrasts are the same under every b; the sign of the mean difference is not.")

print("\n=== Claim 8: content-based curation common to all units is harmless; unit-specific deletion is a departure of 1/(1-kappa) ===")
theta8 = np.array([-.6, 0., .4]); pi8 = family_probs(theta8, c, nu)
bJ = np.array([15, 2, 1, 1.5, 5.]); moder = np.array([.5, .7, 1., 1., 1.])
for label, ret in (("no curation", np.ones(5)), ("common moderation", moder)):
    vis8 = pi8 * bJ * ret; vis8 /= vis8.sum(1, keepdims=True)
    ps = np.log(vis8[1]); ps -= ps[0]
    print(f" {label:18s}: index contrasts from visible data {np.round(np.diff([_invert_mean(v @ nu, ps, nu) for v in vis8]), 10)} (latent: {np.diff(theta8)})")
for kap in (.25, .5, .75):
    w = np.array([1 - kap, 1 - kap, 1, 1, 1.])
    print(f" kappa={kap}: oscillation of the unit-specific retention profile = log {np.exp(hilbert_distance(w, np.ones(5))):.3f}; 1/(1-kappa) = {1 / (1 - kap):.3f}")

print("\n=== Claim 9: one-sided (favourable) sensitivity interval is valid and sharp at both ends ===")
worst = 0; near_lo = near_hi = np.inf
for rep in range(4000):
    th0 = rng.normal(0, .8); Lam = rng.choice([1.25, 1.5, 2., 4.])
    kind = rng.integers(3)
    if kind == 0: d = np.sort(rng.uniform(0, np.log(Lam), 5))                      # any nondecreasing departure
    elif kind == 1: d = np.where(np.arange(5) >= rng.integers(0, 6), np.log(Lam), 0.)   # threshold (deletion of low categories)
    else: d = np.zeros(5)                                                           # no departure at all
    vis = family_probs(th0, psi, nu) * np.exp(d); vis /= vis.sum()
    tt = _invert_mean(vis @ nu, psi, nu)
    lo, hi = drift_interval(tt, psi, nu, Lam, "favorable")
    worst = max(worst, lo - th0, th0 - hi)
    if kind == 1: near_lo = min(near_lo, abs(lo - th0))
    if kind == 2: near_hi = min(near_hi, abs(hi - th0))
print(f" max violation over 4000 monotone departures: {worst:.2e}; closest approach to lower / upper end: {near_lo:.1e} / {near_hi:.1e}")
assert worst < 1e-9

print("\n=== Claim 10: Lemma 1 (rater types) -- mixtures of visibility functions stay within the Hilbert diameter of the types ===")
worst = -np.inf
for rep in range(20000):
    M = rng.integers(2, 5); B = np.exp(rng.normal(0, 1.2, (M, 5)))
    D = max(hilbert_distance(B[m], B[n]) for m in range(M) for n in range(M))
    w1, w2, w3 = rng.dirichlet(np.ones(M) * rng.choice([.3, 1., 5.]), 3) * rng.uniform(.1, 10, (3, 1))
    worst = max(worst, hilbert_distance(w1 @ B, w2 @ B) - D, hilbert_distance(w1 @ B, np.exp(.5 * np.log(w2 @ B) + .5 * np.log(w3 @ B))) - D)
print(f" max over 20000 random draws of d_H(mixture, mixture or geometric mean of mixtures) - D = {worst:.2e} (<= 0)")
assert worst < 1e-9
bO, bM = np.array([6, 3, 1.5, 1, 1.]), bJ
print(f" two types of Section 6: e^D = {np.exp(hilbert_distance(bO, bM)):.2f}; a 30/70 unit against a 70/30 unit: {np.exp(hilbert_distance(.7 * bO + .3 * bM, .3 * bO + .7 * bM)):.2f}")

print("\n=== Claim 11: fabrication bounds with known direction ===")
worst = 0
for rep in range(3000):
    th0 = rng.normal(0, .8); eps = rng.uniform(0, .3); e_true = rng.uniform(0, eps)
    vis = (1 - e_true) * family_probs(th0, psi, nu) + e_true * np.eye(5)[4]
    lo, hi = contamination_interval(_invert_mean(vis @ nu, psi, nu), psi, nu, eps, "promotional")
    worst = max(worst, lo - th0, th0 - hi)
print(f" max violation over 3000 promotional contaminations: {worst:.2e}")
assert worst < 1e-9

print("\n=== Claim 12: breakdown values when only the higher-ranked unit may depart (favourably) ===")
for tj, tk in [(.125, -.125), (.25, -.25)]:
    print(f" theta = ({tj}, {tk}): both units {breakdown(tj, tk, psi, nu):.2f}; higher unit only, favourable {breakdown(tj, tk, psi, nu, suspect='high', direction='favorable'):.2f}")

print("\n=== Claim 13: the ladder of Figure 3 (units B and C) ===")
piBC = family_probs(np.array([0., .5]), c, nu); qBC = piBC * bJ; qBC /= qBC.sum(1, keepdims=True)
lor = lambda a, b_: np.log(a[4] / a[0]) - np.log(b_[4] / b_[0])
print(f" log odds ratio (5 vs 1 star, C vs B): latent {lor(piBC[1], piBC[0]):.4f}, visible {lor(qBC[1], qBC[0]):.4f}, (nu_K - nu_1) * 0.5 = 2")
grid_ok = []
for rep in range(20000):                                           # random departures up to Lambda = 1.5 in both units
    dB, dC = rng.uniform(0, np.log(1.5), (2, 5))
    grid_ok.append(lor(qBC[1] / np.exp(dC), qBC[0] / np.exp(dB)))
print(f" departures up to 1.5 in both units: implied latent log odds ratios span [{min(grid_ok):.3f}, {max(grid_ok):.3f}] inside [2 - 2 log 1.5, 2 + 2 log 1.5] = [{2 - 2 * np.log(1.5):.3f}, {2 + 2 * np.log(1.5):.3f}]")


print("\n=== Claim 14: Proposition 6(a) -- a stochastically monotone rating kernel preserves the ordering of units (FSD) but not the likelihood-ratio order ===")
def fsd_leq(a, b_):                       # a <=_st b : CDF of a pointwise >= CDF of b
    return np.all(np.cumsum(a) >= np.cumsum(b_) - 1e-12)
def lr_leq(a, b_):
    r = np.diff(np.log(b_) - np.log(a)); return np.all(r >= -1e-12)
viol_fsd = 0; lr_broken = 0; trials = 20000
for rep in range(trials):
    ps = rng.normal(0, 1.2, 5); t1, t2 = np.sort(rng.normal(0, .7, 2))
    q1, q2 = family_probs(t1, ps, nu), family_probs(t2, ps, nu)          # LR-ordered inputs (voiced evaluations)
    assert lr_leq(q1, q2)
    # a random stochastically monotone kernel: CDF rows nonincreasing in the rater's own evaluation
    raw = np.sort(rng.random((5, 4)), axis=1)                                  # random CDFs (first four values)
    C = np.minimum.accumulate(raw, axis=0)                                     # nonincreasing in y*
    T = np.diff(np.concatenate([np.zeros((5, 1)), C, np.ones((5, 1))], 1), axis=1)
    o1, o2 = q1 @ T, q2 @ T
    if not fsd_leq(o1, o2): viol_fsd += 1
    o1c, o2c = np.clip(o1, 1e-300, None), np.clip(o2, 1e-300, None)
    if not lr_leq(o1c, o2c): lr_broken += 1
print(f" FSD ordering violated in {viol_fsd} of {trials} random (kernel, pair) draws; LR ordering lost in {lr_broken} of them")
assert viol_fsd == 0

print("\n=== Claim 15: Proposition 6(b,c) -- with the display at its long-run value, the visible mean is increasing in theta and contrasts are not those of the latent index ===")
psi_J = c + np.log(np.array([15, 2, 1, 1.5, 5.])); psi_J -= psi_J[0]
grid = np.linspace(-1.5, 1.5, 61)
for alpha, gamma in ((0.3, 1.0), (0.15, 0.5), (0.3, 0.5), (0.5, 0.5)):
    fps = np.array([display_fixed_point(t, psi_J, nu, alpha, gamma)[0] for t in grid])
    m0 = np.array([mean_score(t, psi_J, nu) for t in grid])
    stable = all(display_fixed_point(t, psi_J, nu, alpha, gamma)[2] for t in grid[::10])
    # slope of F in the display is at most alpha*max(1,gamma) < 1 (contraction), so the fixed point is unique
    slope = max((float((family_probs(t, psi_J, nu) * _influence_rating(nu, m + 1e-4, alpha, gamma, 5)).sum()) -
                 float((family_probs(t, psi_J, nu) * _influence_rating(nu, m - 1e-4, alpha, gamma, 5)).sum())) / 2e-4
                for t in grid[::6] for m in np.linspace(1.2, 4.8, 7))
    print(f" alpha={alpha}, gamma={gamma}: fixed point increasing in theta: {np.all(np.diff(fps) > 0)}; max slope of F in m = {slope:.3f} (< 1); "
          f"stable at sampled theta: {stable}; long-run mean at theta=-0.45 / +0.45: {np.interp(-0.45, grid, fps):.3f} / {np.interp(0.45, grid, fps):.3f} "
          f"(no influence: {np.interp(-0.45, grid, m0):.3f} / {np.interp(0.45, grid, m0):.3f})")
    assert np.all(np.diff(fps) > 0) and slope < 1
# symmetric conformity leaves the long-run mean where it was; asymmetric conformity moves it, by a unit-specific amount
sym = [display_fixed_point(t, psi_J, nu, 0.3, 1.0)[0] - mean_score(t, psi_J, nu) for t in (-1, 0, 1)]
print(f" symmetric conformity (gamma=1): long-run mean minus no-influence mean at theta=-1,0,1: {np.round(sym, 6)}")
assert max(abs(x) for x in sym) < 1e-6
# pseudo-true index contrasts at the population level (no path dependence): fit psi to the long-run visible table
def longrun_table(thetas, alpha, gamma):
    rows = []
    for t in thetas:
        p = family_probs(t, psi_J, nu); m = display_fixed_point(t, psi_J, nu, alpha, gamma)[0]
        x = _influence_rating(nu, m, alpha, gamma, 5); lo = np.floor(x); fr = x - lo
        T = np.zeros((5, 5))
        for i in range(5):
            T[i, int(lo[i]) - 1] += 1 - fr[i]
            if fr[i] > 0: T[i, min(int(lo[i]), 4)] += fr[i]
        rows.append(p @ T)
    return np.array(rows)
thetas = np.linspace(-1.2, 1.2, 25)
for alpha, gamma in ((0.0, 0.5), (0.3, 0.5), (0.5, 0.5)):
    Q = longrun_table(thetas, alpha, gamma)
    fitQ = RatingScaleEB().fit(np.round(1e5 * Q))            # population-level fit of the class model to the long-run table
    idx = fitQ.theta_
    print(f" alpha={alpha}: pseudo-true index contrast between theta=+0.5 and -0.5 (truth 1.0): {np.interp(0.5, thetas, idx) - np.interp(-0.5, thetas, idx):.3f}; "
          f"between +1 and -1 (truth 2.0): {np.interp(1.0, thetas, idx) - np.interp(-1.0, thetas, idx):.3f}; ordering preserved: {np.all(np.diff(idx) > 0)}")

print("\n=== Claim 15b: Proposition 6(c) -- two structures with different latent contrasts and the same visible table ===")
from scipy.optimize import linprog
def monotone_kernel_for(cA, cB, strict=1e-3):
    """Row-stochastic T (K x K), strictly stochastically increasing in y* and constant in the display, with
    p(+-cA) T = p(+-cB) for the two-unit class at theta = +-cA.  Returns T or None (linear-programming feasibility)."""
    pA = family_probs(np.array([-cA, cA]), psi_J, nu); pB = family_probs(np.array([-cB, cB]), psi_J, nu)
    K = 5; n = K * K; A_eq, b_eq, A_ub, b_ub = [], [], [], []
    for y in range(K):
        row = np.zeros(n); row[y * K:(y + 1) * K] = 1; A_eq.append(row); b_eq.append(1.0)
    for u in range(2):
        for h in range(K):
            row = np.zeros(n)
            for y in range(K): row[y * K + h] = pA[u, y]
            A_eq.append(row); b_eq.append(pB[u, h])
    for y in range(K - 1):
        for h in range(K - 1):
            row = np.zeros(n)
            for hh in range(h + 1): row[(y + 1) * K + hh] += 1; row[y * K + hh] -= 1
            A_ub.append(row); b_ub.append(-strict)
    r = linprog(np.zeros(n), A_ub=np.array(A_ub), b_ub=np.array(b_ub), A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, 1)] * n, method="highs")
    return r.x.reshape(K, K) if r.status == 0 else None
T = monotone_kernel_for(0.5, 0.35)
assert T is not None
pA = family_probs(np.array([-0.5, 0.5]), psi_J, nu); pB = family_probs(np.array([-0.35, 0.35]), psi_J, nu)
print(f" a kernel constant in the display, strictly increasing in y*, maps the units theta=+-0.5 onto the visible table of a no-influence class with theta=+-0.35: max |pA T - pB| = {np.abs(pA @ T - pB).max():.1e}")
print(f" the same for a target of +-0.7 (a larger visible contrast): {'feasible' if monotone_kernel_for(0.5, 0.7) is not None else 'infeasible without display feedback'}")

print("\n=== Claim 16: the exchangeability diagnostic has the right size without influence and power with it ===")
rej0 = []; rej1 = []
for rep in range(60):
    d0 = simulate_sequential(300, np.random.default_rng([5, rep]), b=np.array([15, 2, 1, 1.5, 5.]), alpha=0.0)
    rej0.append(exchangeability_test([s for s in d0["seqs"] if len(s)], 199, np.random.default_rng([6, rep]))["p"] <= 0.05)
    d1 = simulate_sequential(300, np.random.default_rng([5, rep]), b=np.array([15, 2, 1, 1.5, 5.]), alpha=0.15, gamma=0.5)
    rej1.append(exchangeability_test([s for s in d1["seqs"] if len(s)], 199, np.random.default_rng([6, rep]))["p"] <= 0.05)
print(f" rejection rate at the 5 percent level over 60 classes of 300 units: {np.mean(rej0):.3f} without influence, {np.mean(rej1):.3f} with alpha=0.15")

print("\n=== Claim 17: the two-source consistency statistic is centred and calibrated for honest units and unconfounded with theta ===")
rng3 = np.random.default_rng(3); J3 = 3000
th3 = 0.45 * rng3.standard_normal(J3)
XA = simulate_class(J3, rng3, b=np.array([15, 2, 1, 1.5, 5.]), theta=th3, loud=np.exp(np.log(20) + 1.6 * rng3.standard_normal(J3)))["X"]
XB = simulate_class(J3, rng3, b=np.ones(5), theta=th3, loud=np.exp(np.log(20) + 1.6 * rng3.standard_normal(J3)))["X"]
X3 = np.stack([XA, XB], 1).astype(float); both = (XA.sum(1) > 0) & (XB.sum(1) > 0)
fit3 = RatingScaleEB().fit(X3[both]); s3 = source_consistency(X3[both], fit3.psi_, nu)
big = (XA.sum(1)[both] >= 30) & (XB.sum(1)[both] >= 30)
print(f" all units: mean {np.nanmean(s3):+.3f}, sd {np.nanstd(s3):.3f}, share |s|>1.96: {np.mean(np.abs(s3) > 1.96):.3f}, corr with theta {np.corrcoef(s3, th3[both])[0, 1]:+.3f}")
print(f" units with 30+ ratings in both sources ({big.sum()}): mean {s3[big].mean():+.3f}, sd {s3[big].std():.3f}, share |s|>1.96: {np.mean(np.abs(s3[big]) > 1.96):.3f}")
