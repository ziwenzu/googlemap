"""Monte Carlo experiments for the manuscript.  Artificial data only.

Usage:  python3 run_simulations.py E1 E2 E3 E4 E5 E6 E7   [--reps 200] [--J 2000]
Every experiment writes output/<name>.json; make_exhibits.py turns those files into tables, macros and figures.
"""
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import spearmanr, norm

from vrindex import (RatingScaleEB, CumulativeEB, simulate_class, family_probs, bayes_average, wilson_lower,
                     drift_interval, contamination_interval)

NU = np.arange(1, 6.0)
C_LATENT = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35]))
VOICE = {"none": np.ones(5), "negativity": np.array([6, 3, 1.5, 1, 1.0]), "J-shaped": np.array([15, 2, 1, 1.5, 5.0])}
TAU = 0.45
SEED = 20260921


def _args():
    a = sys.argv[1:]
    reps = int(a[a.index("--reps") + 1]) if "--reps" in a else 200
    J = int(a[a.index("--J") + 1]) if "--J" in a else 2000
    exps = [x for x in a if x.startswith("E")]
    return exps, reps, J


# ---------------------------------------------------------------------------------------------------------
# E1  Invariance: the same covariate effect under three voice functions
# ---------------------------------------------------------------------------------------------------------
def e1_one(args):
    rep, J = args
    rng = np.random.default_rng([SEED, 1, rep])
    D = (rng.random(J) < 0.5).astype(float)
    theta = 0.30 * D + TAU * rng.standard_normal(J)
    loud = np.exp(np.log(6) + 1.6 * rng.standard_normal(J))
    out = {}
    for name, b in VOICE.items():
        d = simulate_class(J, rng, c=C_LATENT, b=b, theta=theta, loud=loud)
        X = d["X"]
        ok = X.sum(1) > 0
        Xo, Do = X[ok], D[ok]
        Zc = (Do - Do.mean())[:, None]
        raw = (Xo @ NU) / Xo.sum(1)
        ba = bayes_average(Xo)
        rs = RatingScaleEB().fit(Xo, Zc)
        cu = CumulativeEB().fit(Xo, Zc)
        out[name] = dict(raw=raw[Do == 1].mean() - raw[Do == 0].mean(),
                         bayes=ba[Do == 1].mean() - ba[Do == 0].mean(),
                         cumulative=float(cu.beta_[0]), index=float(rs.beta_[0]),
                         latent_stars=float(d["latent_mean"][D == 1].mean() - d["latent_mean"][D == 0].mean()),
                         tau=float(rs.tau_), converged=bool(rs.opt_.success and cu.opt_.success))
    return out


# ---------------------------------------------------------------------------------------------------------
# E2  Sparse and uneven data: accuracy and calibration by review volume
# ---------------------------------------------------------------------------------------------------------
BINS = [(1, 2), (3, 5), (6, 20), (21, 100), (101, 10 ** 9)]


def e2_one(args):
    rep, J = args
    rng = np.random.default_rng([SEED, 2, rep])
    d = simulate_class(J, rng, c=C_LATENT, b=VOICE["J-shaped"], tau=TAU)
    X = d["X"]
    ok = X.sum(1) > 0
    X, theta = X[ok], d["theta"][ok]
    N = X.sum(1)
    rs = RatingScaleEB().fit(X)
    est = dict(raw=(X @ NU) / N, wilson=wilson_lower(X), bayes=bayes_average(X), index=rs.theta_)
    lo, hi = rs.interval()
    shift = np.mean(rs.theta_ - theta)            # location is not identified: evaluate the relative index
    cover = (lo - shift <= theta) & (theta <= hi - shift)
    res = dict(converged=bool(rs.opt_.success), tau=float(rs.tau_),
               overall={k: float(spearmanr(v, theta)[0]) for k, v in est.items()},
               share_lt5=float(np.mean(N < 5)), median_n=float(np.median(N)), n_units=int(ok.sum()))
    truly_bottom = theta <= np.quantile(theta, 0.10)
    res["bottom_decile_precision"] = {k: float(truly_bottom[np.argsort(v)[:int(0.1 * len(v))]].mean()) for k, v in est.items()}
    res["bins"] = {}
    for lo_n, hi_n in BINS:
        m = (N >= lo_n) & (N <= hi_n)
        if m.sum() < 20:
            continue
        key = f"{lo_n}-{hi_n if hi_n < 10**9 else 'inf'}"
        res["bins"][key] = dict(share=float(m.mean()), coverage=float(cover[m].mean()),
                                rmse_index=float(np.sqrt(np.mean((rs.theta_[m] - shift - theta[m]) ** 2))),
                                **{f"rho_{k}": float(spearmanr(v[m], theta[m])[0]) for k, v in est.items()})
    res["coverage"] = float(cover.mean())
    return res


# ---------------------------------------------------------------------------------------------------------
# E3  Composition: two strata (periods / platforms / reviewer types) with different voice functions
# ---------------------------------------------------------------------------------------------------------
def e3_one(args):
    rep, J, corr = args
    rng = np.random.default_rng([SEED, 3, rep, int(corr * 10)])
    theta = TAU * rng.standard_normal(J)
    loud = np.exp(np.log(8) + 1.6 * rng.standard_normal(J))
    # share of a unit's loudness that falls in stratum A; possibly related to theta
    z = corr * theta / TAU + np.sqrt(1 - corr ** 2) * rng.standard_normal(J)
    wA = norm.cdf(1.2 * z)
    bA, bB = np.array([6, 3, 1.5, 1, 1.0]), np.array([4, 1.5, 1, 2, 8.0])
    XA = simulate_class(J, rng, c=C_LATENT, b=bA, theta=theta, loud=loud * wA)["X"]
    XB = simulate_class(J, rng, c=C_LATENT, b=bB, theta=theta, loud=loud * (1 - wA))["X"]
    X3 = np.stack([XA, XB], axis=1)
    ok = X3.sum((1, 2)) > 0
    X3, th = X3[ok], theta[ok]
    Xp = X3.sum(1)
    Nst = X3.sum(2)
    ybar_s = (X3 @ NU).sum(0) / Nst.sum(0)                      # stratum means
    demeaned = ((X3 @ NU) - Nst * ybar_s[None, :]).sum(1) / Nst.sum(1)
    kappa = 5.0
    est = dict(raw=(Xp @ NU) / Xp.sum(1), bayes=bayes_average(Xp),
               demeaned=demeaned * Nst.sum(1) / (Nst.sum(1) + kappa),
               index_pooled=RatingScaleEB().fit(Xp).theta_, index_stratified=RatingScaleEB().fit(X3).theta_)
    return {k: float(spearmanr(v, th)[0]) for k, v in est.items()}


# ---------------------------------------------------------------------------------------------------------
# E4  Unit-specific voice drift: coverage of plain and drift-widened intervals
# ---------------------------------------------------------------------------------------------------------
def _random_drift(J, Lam0, rng):
    d = np.zeros((J, 5))
    if Lam0 <= 1:
        return d
    L = np.log(Lam0)
    kind = rng.integers(0, 3, J)
    sign = rng.choice([-1.0, 1.0], J)
    cut = rng.integers(1, 5, J)
    for j in range(J):
        if kind[j] == 0:
            d[j] = sign[j] * L * (NU - 1) / 4                       # linear in the score
        elif kind[j] == 1:
            d[j] = sign[j] * L * (np.arange(5) >= cut[j])             # step
        else:
            u = rng.uniform(0, 1, 5); u = (u - u.min()) / (u.max() - u.min())
            d[j] = L * u                                               # arbitrary shape, full oscillation
    return d - d.mean(1, keepdims=True)


def e4_one(args):
    rep, J, Lam0 = args
    rng = np.random.default_rng([SEED, 4, rep, int(Lam0 * 100)])
    drift = _random_drift(J, Lam0, rng)
    d = simulate_class(J, rng, c=C_LATENT, b=VOICE["J-shaped"], tau=TAU, drift=drift, med_n=20.0)
    X = d["X"]
    ok = X.sum(1) > 0
    X, theta = X[ok], d["theta"][ok]
    rs = RatingScaleEB().fit(X)
    lo, hi = rs.interval()
    shift = np.mean(rs.theta_ - theta)
    out = {}
    sub = rng.choice(len(theta), size=min(400, len(theta)), replace=False)   # root finding is per unit
    for Lam in (1.0, 1.5, 2.0, 3.0):
        L = np.array([drift_interval(lo[j], rs.psi_, NU, Lam)[0] for j in sub])
        H = np.array([drift_interval(hi[j], rs.psi_, NU, Lam)[1] for j in sub])
        cov = (L - shift <= theta[sub]) & (theta[sub] <= H - shift)
        # robust pairwise statements: j declared above k when the widened intervals do not overlap
        idx = rng.permutation(len(sub)); a, bb = idx[: len(idx) // 2], idx[len(idx) // 2: 2 * (len(idx) // 2)]
        decl = L[a] > H[bb]
        decl2 = L[bb] > H[a]
        wrong = (decl & (theta[sub][a] <= theta[sub][bb])) | (decl2 & (theta[sub][bb] <= theta[sub][a]))
        out[str(Lam)] = dict(coverage=float(cov.mean()), width=float(np.mean(H - L)),
                             declared=float((decl | decl2).mean()),
                             wrong_among_declared=float(wrong.sum() / max((decl | decl2).sum(), 1)))
    return out


# ---------------------------------------------------------------------------------------------------------
# E5  Misspecified latent family (supplement): cumulative-probit populations, with and without polarisation
# ---------------------------------------------------------------------------------------------------------
def _probit_pop(theta, scale, cuts=(-1.6, -1.1, -0.45, 0.45)):
    cuts = np.asarray(cuts)
    cdf = norm.cdf((cuts[None, :] - theta[:, None]) / scale[:, None])
    cdf = np.concatenate([np.zeros((len(theta), 1)), cdf, np.ones((len(theta), 1))], 1)
    return np.diff(cdf, axis=1)


def e5_one(args):
    rep, J, hetero = args
    rng = np.random.default_rng([SEED, 5, rep, int(hetero)])
    theta = 0.5 * rng.standard_normal(J)
    scale = np.exp(0.35 * rng.standard_normal(J)) if hetero else np.ones(J)
    pi = _probit_pop(theta, scale)
    mu = pi @ NU
    loud = np.exp(np.log(6) + 1.6 * rng.standard_normal(J))
    res, keep = {}, None
    ests = {}
    for name, b in VOICE.items():
        lam = loud[:, None] * pi * b[None, :] / (pi.mean(0) * b).sum()
        X = rng.poisson(lam)
        ests[name] = (X, X.sum(1) > 0)
    keep = np.all([v[1] for v in ests.values()], axis=0)
    for name, (X, _) in ests.items():
        Xo = X[keep]
        rs = RatingScaleEB().fit(Xo)
        res[name] = dict(index=float(spearmanr(rs.theta_, mu[keep])[0]),
                         bayes=float(spearmanr(bayes_average(Xo), mu[keep])[0]),
                         raw=float(spearmanr((Xo @ NU) / Xo.sum(1), mu[keep])[0]))
    # population-level (no sampling noise) stability of contrasts across voice functions
    def pop_index(b):
        vis = pi * b; vis /= vis.sum(1, keepdims=True)
        psi = np.log((vis * loud[:, None]).sum(0)); psi -= psi[0]
        from vrindex import _invert_mean
        return np.array([_invert_mean(v @ NU, psi, NU) for v in vis[:300]]), (vis @ NU)[:300]
    i0, m0 = pop_index(VOICE["none"]); i2, m2 = pop_index(VOICE["J-shaped"])
    res["pop_slope_index"] = float(np.polyfit(i0, i2, 1)[0]); res["pop_slope_mean"] = float(np.polyfit(m0, m2, 1)[0])
    res["pop_rank_agree_index"] = float(spearmanr(i0, i2)[0]); res["pop_rank_agree_mean"] = float(spearmanr(m0, m2)[0])
    res["pop_rank_index_vs_latent"] = float(spearmanr(i2, mu[:300])[0]); res["pop_rank_mean_vs_latent"] = float(spearmanr(m2, mu[:300])[0])
    return res


# ---------------------------------------------------------------------------------------------------------
# E6  Informative rating volume (supplement): the voice model makes volume depend on satisfaction
# ---------------------------------------------------------------------------------------------------------
def e6_one(args):
    rep, J = args
    rng = np.random.default_rng([SEED, 6, rep])
    D = (rng.random(J) < 0.5).astype(float)
    theta = 0.30 * D + TAU * rng.standard_normal(J)
    loud = np.exp(np.log(6) + 1.6 * rng.standard_normal(J))
    b = VOICE["J-shaped"]
    pi = family_probs(theta, C_LATENT, NU)
    vis = pi * b; vis /= vis.sum(1, keepdims=True)
    g = (pi * b).sum(1) / (family_probs(np.zeros(1), C_LATENT, NU)[0] * b).sum()
    out = {}
    for tag, mult in (("exogenous", np.ones(J)), ("endogenous", g)):
        N = rng.poisson(loud * mult)
        X = np.array([rng.multinomial(n, v) for n, v in zip(N, vis)])
        ok = N > 0
        Xo, Do, th = X[ok], D[ok], theta[ok]
        Z0 = (Do - Do.mean())[:, None]
        m0 = RatingScaleEB().fit(Xo, Z0)
        res = dict(gamma=float(m0.beta_[0]), tau=float(m0.tau_), gamma_se=float(m0.beta_se_[0]), tau_se=float(m0.tau_se_),
                   cover_gamma=bool(abs(m0.beta_[0] - 0.30) <= 1.96 * m0.beta_se_[0]),
                   cover_tau=bool(abs(m0.tau_ - TAU) <= 1.96 * m0.tau_se_),
                   truth=float(th[Do == 1].mean() - th[Do == 0].mean()),
                   corr_theta_logN=float(np.corrcoef(th, np.log(Xo.sum(1)))[0, 1]),
                   rho=float(spearmanr(m0.theta_, th)[0]))
        if tag == "endogenous":
            lN = np.log(Xo.sum(1)); l1 = lN - lN.mean()
            Z1 = np.column_stack([Z0[:, 0], l1, l1 ** 2 - np.mean(l1 ** 2)])
            res["gamma_volume_terms"] = float(RatingScaleEB().fit(Xo, Z1, se=False).beta_[0])
            # covariate-by-volume interactions; report the average contrast in fitted prior means
            q2 = l1 ** 2 - np.mean(l1 ** 2)
            Z2 = np.column_stack([Z0[:, 0], l1, q2, Z0[:, 0] * l1, Z0[:, 0] * q2])
            mu2 = Z2 @ RatingScaleEB().fit(Xo, Z2, se=False).beta_
            res["gamma_interacted"] = float(mu2[Do == 1].mean() - mu2[Do == 0].mean())
        out[tag] = res
    return out


# ---------------------------------------------------------------------------------------------------------
# E7  Contamination (supplement): fake five-star ratings in a subset of units
# ---------------------------------------------------------------------------------------------------------
def e7_one(args):
    rep, J = args
    rng = np.random.default_rng([SEED, 7, rep])
    d = simulate_class(J, rng, c=C_LATENT, b=VOICE["J-shaped"], tau=TAU, med_n=30.0)
    X = d["X"].astype(float); ok = X.sum(1) > 0
    X, theta = X[ok], d["theta"][ok]
    fake = rng.random(len(theta)) < 0.10
    eta = 0.20
    add = np.round(eta / (1 - eta) * X.sum(1) * fake)
    Xc = X.copy(); Xc[:, 4] += add
    rs = RatingScaleEB().fit(Xc)
    shift = np.mean(rs.theta_[~fake] - theta[~fake])
    lo, hi = rs.interval()
    idx = np.where(fake & (add > 0))[0][:150]
    cov_plain = np.mean((lo[idx] - shift <= theta[idx]) & (theta[idx] <= hi[idx] - shift))
    Lb = np.array([contamination_interval(lo[j], rs.psi_, NU, eta)[0] for j in idx])
    Hb = np.array([contamination_interval(hi[j], rs.psi_, NU, eta)[1] for j in idx])
    cov_rob = np.mean((Lb - shift <= theta[idx]) & (theta[idx] <= Hb - shift))
    return dict(bias_fake=float(np.mean(rs.theta_[idx] - shift - theta[idx])), cov_plain=float(cov_plain), cov_robust=float(cov_rob))


# ---------------------------------------------------------------------------------------------------------
# E8  Units nested in jurisdictions (supplement): recovery of the group-level index
# ---------------------------------------------------------------------------------------------------------
def e8_one(args):
    rep, _ = args
    from vrindex import TwoLevelRatingScaleEB
    rng = np.random.default_rng([SEED, 8, rep])
    G = 150
    sizes = rng.integers(3, 40, G)
    g = np.repeat(np.arange(G), sizes)
    J = len(g)
    u = 0.30 * rng.standard_normal(G)
    theta = u[g] + 0.40 * rng.standard_normal(J)
    d = simulate_class(J, rng, c=C_LATENT, b=VOICE["J-shaped"], theta=theta)
    X = d["X"]; ok = X.sum(1) > 0
    X, g, theta = X[ok], g[ok], theta[ok]
    m = TwoLevelRatingScaleEB().fit(X, g)
    lab = m.group_labels_
    shift = np.mean(m.group_ - u[lab])
    N = X.sum(1); Tt = X @ NU
    pooled = np.bincount(g, Tt, G)[lab] / np.bincount(g, N, G)[lab]
    unitavg = np.bincount(g, Tt / N, G)[lab] / np.bincount(g, np.ones(len(g)), G)[lab]
    ba = bayes_average(X); baavg = np.bincount(g, ba, G)[lab] / np.bincount(g, np.ones(len(g)), G)[lab]
    return dict(tau=float(m.tau_), omega=float(m.omega_), iters=int(m.n_iter_),
                rho_group=float(spearmanr(m.group_, u[lab])[0]), rho_pooled=float(spearmanr(pooled, u[lab])[0]),
                rho_unitavg=float(spearmanr(unitavg, u[lab])[0]), rho_bayesavg=float(spearmanr(baavg, u[lab])[0]),
                cover_group=float(np.mean(np.abs(m.group_ - shift - u[lab]) <= 1.96 * m.group_sd_)),
                cover_unit=float(np.mean(np.abs(m.theta_ - np.mean(m.theta_ - theta) - theta) <= 1.96 * m.theta_sd_)))


RUNNERS = dict(E1=(e1_one, lambda r, J: [(i, J) for i in range(r)]),
               E2=(e2_one, lambda r, J: [(i, J) for i in range(r)]),
               E3=(e3_one, lambda r, J: [(i, J, c) for c in (0.0, 0.5) for i in range(r)]),
               E4=(e4_one, lambda r, J: [(i, J, L) for L in (1.0, 1.5, 2.0) for i in range(r)]),
               E5=(e5_one, lambda r, J: [(i, J, h) for h in (False, True) for i in range(r)]),
               E6=(e6_one, lambda r, J: [(i, J) for i in range(r)]),
               E7=(e7_one, lambda r, J: [(i, J) for i in range(r)]),
               E8=(e8_one, lambda r, J: [(i, J) for i in range(r)]))

if __name__ == "__main__":
    exps, reps, J = _args()
    for e in exps:
        fn, mk = RUNNERS[e]
        tasks = mk(reps, J)
        t0 = time.time()
        with Pool(2) as pool:
            res = pool.map(fn, tasks, chunksize=4)
        json.dump(dict(experiment=e, reps=reps, J=J, seed=SEED, tasks=[list(map(float, t)) for t in tasks], results=res),
                  open(f"../output/{e}.json", "w"))
        print(f"{e}: {len(tasks)} runs in {time.time() - t0:.0f}s", flush=True)
