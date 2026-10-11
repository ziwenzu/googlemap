"""Numerical verification of the formal claims in the manuscript (population level, no sampling noise)."""
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.special import expit, logsumexp
from vrindex import (family_probs, mean_score, _invert_mean, drift_interval, breakdown, contamination_interval, _extreme_mean,
                     profile_drift_set, min_drift)

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
