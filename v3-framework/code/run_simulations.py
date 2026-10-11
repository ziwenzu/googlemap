"""Monte Carlo experiments for the manuscript.  Artificial data only.

Usage:  python3 run_simulations.py E1 E2 E3 E4 E5 E6 E8 E9 E10 E11 E12   [--reps 200] [--J 2000]
Every experiment writes ../results/simulations/<name>.json; make_exhibits.py turns those files into tables, macros and figures.
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, norm

from vrindex import (RatingScaleEB, CumulativeEB, simulate_class, family_probs, bayes_average, wilson_lower,
                     share_favorable, drift_interval, contamination_interval, hilbert_distance)

OUT = Path("../results/simulations"); OUT.mkdir(parents=True, exist_ok=True)
NU = np.arange(1, 6.0)
C_LATENT = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35]))
VOICE = {"none": np.ones(5), "negativity": np.array([6, 3, 1.5, 1, 1.0]), "J-shaped": np.array([15, 2, 1, 1.5, 5.0])}
# content-based moderation applied to every unit alike: half of one-star and 30 percent of two-star ratings removed
MODERATION = np.array([0.5, 0.7, 1.0, 1.0, 1.0])
REGIMES = dict(VOICE, **{"J-shaped, moderated": VOICE["J-shaped"] * MODERATION})
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
    for name, b in REGIMES.items():
        d = simulate_class(J, rng, c=C_LATENT, b=b, theta=theta, loud=loud)
        X = d["X"]
        ok = X.sum(1) > 0
        Xo, Do = X[ok], D[ok]
        Zc = (Do - Do.mean())[:, None]
        diff = lambda v: float(v[Do == 1].mean() - v[Do == 0].mean())
        rs = RatingScaleEB().fit(Xo, Zc, se=False)
        cu = CumulativeEB().fit(Xo, Zc)
        out[name] = dict(raw=diff((Xo @ NU) / Xo.sum(1)), fav=diff(share_favorable(Xo)), wilson=diff(wilson_lower(Xo)),
                         bayes=diff(bayes_average(Xo)), cumulative=float(cu.beta_[0]), index=float(rs.beta_[0]),
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
    est = dict(raw=(X @ NU) / N, fav=share_favorable(X), wilson=wilson_lower(X), bayes=bayes_average(X),
               cumulative=CumulativeEB().fit(X).theta_, index=rs.theta_)
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
    est = dict(raw=(Xp @ NU) / Xp.sum(1), fav=share_favorable(Xp), wilson=wilson_lower(Xp), bayes=bayes_average(Xp),
               cumulative=CumulativeEB().fit(Xp).theta_,
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
# helpers for the manipulation experiments
# ---------------------------------------------------------------------------------------------------------
def _pct_rank(v):
    """percentile rank (0-100) of every unit on a summary v (higher = better)."""
    from scipy.stats import rankdata
    return 100.0 * (rankdata(v) - 0.5) / len(v)


def _summaries(X, rs):
    N = X.sum(1)
    return dict(raw=(X @ NU) / N, fav=share_favorable(X), wilson=wilson_lower(X), bayes=bayes_average(X), index=rs.theta_)


# ---------------------------------------------------------------------------------------------------------
# E9  Unit-specific curation.  Content-based moderation common to the class keeps 50 and 70 percent of one- and
#     two-star ratings in EVERY unit; one unit in ten removes a further share kappa of what would remain visible
#     in those two categories.  Coverage is evaluated on (at most) the first 150 manipulating units, and the
#     unidentified location of the index is fixed with the units that do not manipulate.
# ---------------------------------------------------------------------------------------------------------
def e9_one(args):
    rep, J, kappa = args
    rng = np.random.default_rng([SEED, 9, rep])                     # same units for every kappa
    theta = TAU * rng.standard_normal(J)
    loud = np.exp(np.log(20) + 1.6 * rng.standard_normal(J))
    manip = rng.random(J) < 0.10
    retain = np.tile(MODERATION, (J, 1))
    retain[manip, :2] *= (1.0 - kappa)
    d = simulate_class(J, np.random.default_rng([SEED, 9, rep, int(round(100 * kappa))]), c=C_LATENT, b=VOICE["J-shaped"],
                       theta=theta, loud=loud, retain=retain)
    X = d["X"].astype(float); ok = X.sum(1) > 0
    X, th, mp = X[ok], theta[ok], manip[ok]
    rs = RatingScaleEB().fit(X)
    est = _summaries(X, rs)
    true_rank = _pct_rank(th)
    res = dict(kappa=kappa, n_manip=int(mp.sum()),
               rank_gain={k: float(np.mean(_pct_rank(v)[mp] - true_rank[mp])) for k, v in est.items()},
               rho={k: float(spearmanr(v, th)[0]) for k, v in est.items()})
    shift = np.mean(rs.theta_[~mp] - th[~mp])
    lo, hi = rs.interval()
    idx = np.where(mp)[0][:150]
    Lam = 1.0 / (1.0 - kappa) if kappa > 0 else 1.0
    one = np.array([drift_interval(lo[j], rs.psi_, NU, Lam, "favorable")[0] for j in idx])
    two = np.array([drift_interval(lo[j], rs.psi_, NU, Lam)[0] for j in idx]), np.array([drift_interval(hi[j], rs.psi_, NU, Lam)[1] for j in idx])
    res.update(bias=float(np.mean(rs.theta_[idx] - shift - th[idx])),
               cov_plain=float(np.mean((lo[idx] - shift <= th[idx]) & (th[idx] <= hi[idx] - shift))),
               cov_directional=float(np.mean((one - shift <= th[idx]) & (th[idx] <= hi[idx] - shift))),
               cov_twosided=float(np.mean((two[0] - shift <= th[idx]) & (th[idx] <= two[1] - shift))),
               width_plain=float(np.median(hi[idx] - lo[idx])), width_directional=float(np.median(hi[idx] - one)),
               width_twosided=float(np.median(two[1] - two[0])),
               cov_clean=float(np.mean((lo[~mp] - shift <= th[~mp]) & (th[~mp] <= hi[~mp] - shift))))
    return res


# ---------------------------------------------------------------------------------------------------------
# E10  Fabrication: some units add five-star ratings equal to a share eps of what is then visible
# ---------------------------------------------------------------------------------------------------------
def e10_one(args):
    rep, J, eps = args
    rng = np.random.default_rng([SEED, 10, rep])
    theta = TAU * rng.standard_normal(J)
    loud = np.exp(np.log(20) + 1.6 * rng.standard_normal(J))
    manip = rng.random(J) < 0.10
    pi = family_probs(theta, C_LATENT, NU)
    lam = loud * (pi * VOICE["J-shaped"]).sum(1) / (family_probs(np.zeros(1), C_LATENT, NU)[0] * VOICE["J-shaped"]).sum()
    fake = np.zeros((J, 5)); fake[manip, 4] = eps / (1.0 - eps) * lam[manip]
    d = simulate_class(J, np.random.default_rng([SEED, 10, rep, int(round(100 * eps))]), c=C_LATENT, b=VOICE["J-shaped"],
                       theta=theta, loud=loud, fake=fake)
    X = d["X"].astype(float); ok = X.sum(1) > 0
    X, th, mp = X[ok], theta[ok], manip[ok]
    rs = RatingScaleEB().fit(X)
    est = _summaries(X, rs)
    true_rank = _pct_rank(th)
    res = dict(eps=eps, n_manip=int(mp.sum()),
               rank_gain={k: float(np.mean(_pct_rank(v)[mp] - true_rank[mp])) for k, v in est.items()},
               rho={k: float(spearmanr(v, th)[0]) for k, v in est.items()})
    shift = np.mean(rs.theta_[~mp] - th[~mp])
    lo, hi = rs.interval()
    idx = np.where(mp)[0][:150]
    one = np.array([contamination_interval(lo[j], rs.psi_, NU, eps, "promotional")[0] for j in idx]) if eps > 0 else lo[idx]
    res.update(bias=float(np.mean(rs.theta_[idx] - shift - th[idx])),
               cov_plain=float(np.mean((lo[idx] - shift <= th[idx]) & (th[idx] <= hi[idx] - shift))),
               cov_directional=float(np.mean((one - shift <= th[idx]) & (th[idx] <= hi[idx] - shift))),
               width_plain=float(np.median(hi[idx] - lo[idx])), width_directional=float(np.median(hi[idx] - one)),
               share_unbounded=float(np.mean(~np.isfinite(one))),
               cov_clean=float(np.mean((lo[~mp] - shift <= th[~mp]) & (th[~mp] <= hi[~mp] - shift))))
    return res


# ---------------------------------------------------------------------------------------------------------
# E11  Rater types: ordinary and motivated raters voice differently, and units differ in their mix of types
# ---------------------------------------------------------------------------------------------------------
TYPE_B = np.array([[6, 3, 1.5, 1, 1.0], [15, 2, 1, 1.5, 5.0]])     # ordinary raters, motivated raters


def e11_one(args):
    rep, J, conc = args
    rng = np.random.default_rng([SEED, 11, rep, int(conc)])
    theta = TAU * rng.standard_normal(J)
    loud = np.exp(np.log(20) + 1.6 * rng.standard_normal(J))
    share = np.full(J, 0.5) if conc >= 10 ** 6 else rng.beta(0.5 * conc, 0.5 * conc, J)   # share of motivated raters
    pi = family_probs(theta, C_LATENT, NU)
    ref = (family_probs(np.zeros(1), C_LATENT, NU)[0] * TYPE_B.mean(0)).sum()
    lam_t = np.stack([loud[:, None] * (1 - share)[:, None] * pi * TYPE_B[0], loud[:, None] * share[:, None] * pi * TYPE_B[1]], 1) / ref
    Xt = rng.poisson(lam_t).astype(float)                           # J x 2 x K, by rater type
    X = Xt.sum(1)
    ok = X.sum(1) > 0
    X, Xt, th, sh = X[ok], Xt[ok], theta[ok], share[ok]
    # realised departure of each unit's effective visibility function from the class norm (the volume-weighted mix)
    beff = (1 - sh)[:, None] * TYPE_B[0] + sh[:, None] * TYPE_B[1]
    norm = (X.sum(1)[:, None] * beff).sum(0)
    drift = np.exp(np.array([hilbert_distance(bj, norm) for bj in beff]))
    rs = RatingScaleEB().fit(X)
    shift = np.mean(rs.theta_ - th)
    lo, hi = rs.interval()
    sub = rng.choice(len(th), size=min(300, len(th)), replace=False)
    out = dict(conc=conc, sd_share=float(np.std(sh)), drift_median=float(np.median(drift)), drift_p90=float(np.quantile(drift, 0.9)),
               drift_max=float(drift.max()), bound=float(np.exp(hilbert_distance(TYPE_B[0], TYPE_B[1]))),
               rho_pooled=float(spearmanr(rs.theta_, th)[0]),
               cov_plain=float(np.mean((lo - shift <= th) & (th <= hi - shift))))
    Lq = float(np.quantile(drift, 0.9))
    L = np.array([drift_interval(lo[j], rs.psi_, NU, Lq)[0] for j in sub]); H = np.array([drift_interval(hi[j], rs.psi_, NU, Lq)[1] for j in sub])
    out["cov_widened_p90"] = float(np.mean((L - shift <= th[sub]) & (th[sub] <= H - shift)))
    st = RatingScaleEB().fit(Xt)                                     # rater type observed: stratify
    s2 = np.mean(st.theta_ - th); lo2, hi2 = st.interval()
    out.update(rho_stratified=float(spearmanr(st.theta_, th)[0]), cov_stratified=float(np.mean((lo2 - s2 <= th) & (th <= hi2 - s2))))
    return out


# ---------------------------------------------------------------------------------------------------------
# E12  How many units does a comparison class need?
# ---------------------------------------------------------------------------------------------------------
def e12_one(args):
    rep, _, Jc = args
    Jc = int(Jc)
    rng = np.random.default_rng([SEED, 12, rep, Jc])
    d = simulate_class(Jc, rng, c=C_LATENT, b=VOICE["J-shaped"], tau=TAU, med_n=8.0)
    X = d["X"].astype(float); ok = X.sum(1) > 0
    X, th = X[ok], d["theta"][ok]
    if len(th) < 5 or (X.sum(0) == 0).any():
        return dict(J=Jc, failed=True)
    rs = RatingScaleEB().fit(X)
    shift = np.mean(rs.theta_ - th); lo, hi = rs.interval()
    return dict(J=Jc, failed=False, tau=float(rs.tau_), converged=bool(rs.opt_.success),
                rho=float(spearmanr(rs.theta_, th)[0]), rho_bayes=float(spearmanr(bayes_average(X), th)[0]),
                rmse=float(np.sqrt(np.mean((rs.theta_ - shift - th) ** 2))),
                coverage=float(np.mean((lo - shift <= th) & (th <= hi - shift))))


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
               E8=(e8_one, lambda r, J: [(i, J) for i in range(r)]),
               E9=(e9_one, lambda r, J: [(i, J, k) for k in (0.0, 0.25, 0.5, 0.75) for i in range(r)]),
               E10=(e10_one, lambda r, J: [(i, J, e) for e in (0.0, 0.1, 0.2, 0.3) for i in range(r)]),
               E11=(e11_one, lambda r, J: [(i, J, c) for c in (10 ** 6, 50, 10, 4, 2) for i in range(r)]),
               E12=(e12_one, lambda r, J: [(i, J, n) for n in (15, 30, 60, 125, 250, 500, 1000) for i in range(r)]))

if __name__ == "__main__":
    exps, reps, J = _args()
    for e in exps:
        fn, mk = RUNNERS[e]
        tasks = mk(reps, J)
        t0 = time.time()
        with Pool(2) as pool:
            res = pool.map(fn, tasks, chunksize=4)
        json.dump(dict(experiment=e, reps=reps, J=J, seed=SEED, tasks=[list(map(float, t)) for t in tasks], results=res),
                  open(OUT / f"{e}.json", "w"))
        print(f"{e}: {len(tasks)} runs in {time.time() - t0:.0f}s", flush=True)
