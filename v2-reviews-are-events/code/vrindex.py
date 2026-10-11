"""vrindex: a selection-invariant index from voluntary ratings.

Core objects
------------
RatingScaleEB   hierarchical adjacent-category (rating-scale) model for a units-by-categories count
                table, estimated by marginal maximum likelihood with adaptive Gauss-Hermite quadrature.
                The category effects psi absorb the (unknown) common voice function and the row totals
                absorb unknown encounter volumes; the unit effects theta are the index.
CumulativeEB    same hierarchy with a cumulative-logit (ordered logit) link; used only as a comparator.
drift_interval  sensitivity interval for theta under unit-specific voice drift bounded by Lambda.
breakdown       largest Lambda under which the sign of a pairwise contrast is preserved.
contamination_interval  bounds on theta when up to a share eta of visible ratings are not genuine.
profile_drift_set, min_drift  sharper drift set for a well-measured unit; lower bound on its own drift.
simulate_class  data-generating process for one comparison class (Poisson rating events).

Only numpy and scipy are required.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, expit

__all__ = ["RatingScaleEB", "TwoLevelRatingScaleEB", "CumulativeEB", "drift_interval", "breakdown", "contamination_interval",
           "profile_drift_set", "min_drift",
           "simulate_class", "family_probs", "mean_score", "bayes_average", "wilson_lower"]


# ----------------------------------------------------------------------------------------------
# The one-parameter exponential family  p_h(theta) ∝ exp(psi_h + theta * nu_h)
# ----------------------------------------------------------------------------------------------
def family_probs(theta, psi, nu):
    """Category probabilities for (an array of) theta.  Returns array (..., K)."""
    theta = np.asarray(theta, float)
    eta = psi + theta[..., None] * nu
    return np.exp(eta - logsumexp(eta, axis=-1, keepdims=True))


def mean_score(theta, psi, nu):
    return (family_probs(theta, psi, nu) * nu).sum(-1)


def _invert_mean(target, psi, nu, lo=-40.0, hi=40.0):
    """theta such that mean_score(theta) = target (target strictly inside (min nu, max nu))."""
    eps = 1e-10
    target = min(max(target, nu.min() + eps), nu.max() - eps)
    return brentq(lambda t: mean_score(t, psi, nu) - target, lo, hi, xtol=1e-12)


class RatingScaleEB:
    """Hierarchical rating-scale (adjacent-category logit) model, optionally stratified.

    X_js | N_js, theta_j ~ Multinomial(N_js, p(theta_j; psi_s)),   p_h(theta; psi_s) ∝ exp(psi_sh + theta nu_h)
    theta_j ~ Normal(z_j' beta, tau^2)

    Strata s (platforms, periods, reviewer types, service types) may have arbitrarily different category
    effects psi_s: each absorbs its own baseline and its own voice function.  psi_s1 = 0.  There is no
    intercept in z_j' beta: the location of theta is not identified separately from the linear component of
    psi, so theta is a *relative* index centred at its prior mean.
    """

    def __init__(self, scores=None, n_quad=25):
        self.scores = None if scores is None else np.asarray(scores, float)
        self.n_quad = n_quad
        self._x, self._w = hermegauss(n_quad)
        self._logw = np.log(self._w) + 0.5 * self._x ** 2

    # -- quadrature machinery ---------------------------------------------------------------
    def _moments(self, th, psi):
        """mean and variance of the score in every stratum at theta.  th: (J,), psi: (S,K) -> (J,S)."""
        nu = self.scores
        eta = psi[None, :, :] + th[:, None, None] * nu[None, None, :]
        p = np.exp(eta - logsumexp(eta, axis=2, keepdims=True))
        m = (p * nu).sum(2)
        v = (p * nu ** 2).sum(2) - m ** 2
        return m, v

    def _posterior(self, psi, mu, tau, X, N, T):
        nu = self.scores
        th = mu.copy()
        Ttot = T.sum(1)
        for _ in range(200):  # vectorised Newton for the posterior mode (strictly concave)
            m, v = self._moments(th, psi)
            g = Ttot - (N * m).sum(1) - (th - mu) / tau ** 2
            h = -(N * v).sum(1) - 1.0 / tau ** 2
            step = np.clip(g / h, -1.0, 1.0)          # damped Newton: the target is strictly concave
            th = th - step
            if np.max(np.abs(step)) < 1e-10:
                break
        m, v = self._moments(th, psi)
        sd = 1.0 / np.sqrt((N * v).sum(1) + 1.0 / tau ** 2)
        nodes = th[:, None] + sd[:, None] * self._x[None, :]                           # J x Q
        # log-normaliser A_s(theta) at every node: J x Q x S
        A = logsumexp(psi[None, None, :, :] + nodes[:, :, None, None] * nu[None, None, None, :], axis=3)
        ll = Ttot[:, None] * nodes + (X * psi[None, :, :]).sum((1, 2))[:, None] - (N[:, None, :] * A).sum(2)
        lp = -0.5 * ((nodes - mu[:, None]) / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi)
        lw = self._logw[None, :] + ll + lp + np.log(sd)[:, None]
        logmarg = logsumexp(lw, axis=1)
        W = np.exp(lw - logmarg[:, None])
        pm = (W * nodes).sum(1)
        pv = (W * nodes ** 2).sum(1) - pm ** 2
        return dict(mode=th, nodes=nodes, W=W, logmarg=logmarg, mean=pm, var=pv)

    def _unpack(self, par, S, K, p):
        psi = np.concatenate([np.zeros((S, 1)), par[:S * (K - 1)].reshape(S, K - 1)], axis=1)
        beta = par[S * (K - 1):S * (K - 1) + p]
        tau = np.exp(par[S * (K - 1) + p])
        return psi, beta, tau

    def _negloglik(self, par, X, Z, N, T):
        J, S, K = X.shape
        p = Z.shape[1]
        psi, beta, tau = self._unpack(par, S, K, p)
        mu = Z @ beta if p else np.zeros(J)
        post = self._posterior(psi, mu, tau, X, N, T)
        nu = self.scores
        # gradients by Fisher's identity (posterior expectations of complete-data scores)
        eta = psi[None, None, :, :] + post["nodes"][:, :, None, None] * nu[None, None, None, :]
        P = np.exp(eta - logsumexp(eta, axis=3, keepdims=True))                         # J x Q x S x K
        Ep = np.einsum("jq,jqsk->jsk", post["W"], P)
        g_psi = (X - N[:, :, None] * Ep).sum(0)[:, 1:].ravel()
        d = post["mean"] - mu
        g_beta = Z.T @ (d / tau ** 2) if p else np.zeros(0)
        g_ltau = ((post["var"] + d ** 2) / tau ** 2 - 1.0).sum()
        return -post["logmarg"].sum(), -np.concatenate([g_psi, g_beta, [g_ltau]])

    # -- public API ---------------------------------------------------------------------------
    def fit(self, X, Z=None, start=None, se=None):
        """X: (J, K) counts, or (J, S, K) counts for S strata.  Z: optional (J, p) covariates (centred).

        se: compute standard errors of (psi, beta, log tau) from the observed information (finite differences
        of the analytic score).  Default: only when covariates are supplied.  Results: vcov_, beta_se_, tau_se_."""
        X = np.asarray(X, float)
        if X.ndim == 2:
            X = X[:, None, :]
        J, S, K = X.shape
        if self.scores is None:
            self.scores = np.arange(1, K + 1, dtype=float)
        nu = self.scores
        Z = np.zeros((J, 0)) if Z is None else np.asarray(Z, float).reshape(J, -1)
        N = X.sum(2)                                   # J x S
        T = X @ nu                                     # J x S
        keep = N.sum(1) > 0
        if start is None:
            tot = X.sum(0) + 0.5
            psi0 = np.log(tot / tot[:, :1])[:, 1:].ravel()
            start = np.concatenate([psi0, np.zeros(Z.shape[1]), [np.log(0.5)]])
        bounds = [(-15.0, 15.0)] * (S * (K - 1)) + [(-10.0, 10.0)] * Z.shape[1] + [(np.log(1e-2), np.log(5.0))]
        res = minimize(self._negloglik, start, args=(X[keep], Z[keep], N[keep], T[keep]), jac=True,
                       method="L-BFGS-B", bounds=bounds, options=dict(maxiter=1000, ftol=1e-12, gtol=1e-6))
        self.opt_ = res
        self.psi_, self.beta_, self.tau_ = self._unpack(res.x, S, K, Z.shape[1])
        self.vcov_ = self.beta_se_ = self.tau_se_ = None
        if se or (se is None and Z.shape[1] > 0):
            args = (X[keep], Z[keep], N[keep], T[keep])
            P = len(res.x)
            H = np.zeros((P, P))
            for i in range(P):
                h = 1e-4 * max(1.0, abs(res.x[i]))
                e = np.zeros(P); e[i] = h
                H[:, i] = (self._negloglik(res.x + e, *args)[1] - self._negloglik(res.x - e, *args)[1]) / (2 * h)
            H = 0.5 * (H + H.T)
            try:
                V = np.linalg.inv(H)
                if np.any(np.diag(V) <= 0):
                    raise np.linalg.LinAlgError
            except np.linalg.LinAlgError:
                V = np.linalg.pinv(H)
            self.vcov_ = V
            nb = S * (K - 1)
            self.beta_se_ = np.sqrt(np.clip(np.diag(V)[nb:nb + Z.shape[1]], 0, None))
            self.tau_se_ = float(self.tau_ * np.sqrt(max(V[-1, -1], 0.0)))          # delta method from log tau
        mu = Z @ self.beta_ if Z.shape[1] else np.zeros(J)
        post = self._posterior(self.psi_, mu, self.tau_, X, N, T)   # units with no ratings return the prior
        self.theta_, self.theta_sd_ = post["mean"], np.sqrt(post["var"])
        self.prior_mean_ = mu
        self.loglik_ = -res.fun
        self.N_ = N.sum(1)
        self.mean_visible_ = np.where(self.N_ > 0, T.sum(1) / np.maximum(self.N_, 1), np.nan)
        if S == 1:
            self.psi_ = self.psi_[0]
        return self

    def theta_mle(self, X):
        """Unshrunk single-stratum index: the monotone transform of the visible mean (NaN on the boundary)."""
        X = np.asarray(X, float)
        N, T = X.sum(1), X @ self.scores
        out = np.full(len(N), np.nan)
        for j in np.where(N > 0)[0]:
            ybar = T[j] / N[j]
            if self.scores.min() < ybar < self.scores.max():
                out[j] = _invert_mean(ybar, self.psi_, self.scores)
        return out

    def interval(self, level=0.95):
        from scipy.stats import norm
        z = norm.ppf(0.5 + level / 2)
        return self.theta_ - z * self.theta_sd_, self.theta_ + z * self.theta_sd_


class CumulativeEB:
    """Comparator: hierarchical cumulative-logit (ordered logit) model with unit random effects.
    P(Y <= h | theta) = expit(kappa_h - theta), theta_j ~ N(0, tau^2)."""

    def __init__(self, n_quad=25):
        self.n_quad = n_quad
        self._x, self._w = hermegauss(n_quad)
        self._logw = np.log(self._w) + 0.5 * self._x ** 2

    @staticmethod
    def _probs(theta, kappa):
        c = expit(kappa[None, :] - np.asarray(theta, float).reshape(-1, 1))
        c = np.concatenate([np.zeros((c.shape[0], 1)), c, np.ones((c.shape[0], 1))], axis=1)
        return np.clip(np.diff(c, axis=1), 1e-300, None)

    def _loglik_theta(self, theta, kappa, X):
        return (X * np.log(self._probs(theta, kappa))).sum(1)

    def _posterior(self, kappa, tau, X, mu=None):
        J = X.shape[0]
        mu = np.zeros(J) if mu is None else mu
        th = mu.copy()
        f = lambda t: self._loglik_theta(t, kappa, X) - 0.5 * ((t - mu) / tau) ** 2
        h = 1e-4
        for _ in range(80):  # Newton with numerical derivatives (log-concave target)
            f0, fp, fm = f(th), f(th + h), f(th - h)
            g = (fp - fm) / (2 * h)
            H = np.minimum((fp - 2 * f0 + fm) / h ** 2, -1e-8)
            step = np.clip(g / H, -2, 2)
            th = th - step
            if np.max(np.abs(step)) < 1e-8:
                break
        f0, fp, fm = f(th), f(th + h), f(th - h)
        H = np.minimum((fp - 2 * f0 + fm) / h ** 2, -1e-8)
        sd = 1 / np.sqrt(-H)
        nodes = th[:, None] + sd[:, None] * self._x[None, :]
        ll = np.stack([self._loglik_theta(nodes[:, q], kappa, X) for q in range(nodes.shape[1])], axis=1)
        lp = -0.5 * ((nodes - mu[:, None]) / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi)
        lw = self._logw[None, :] + ll + lp + np.log(sd)[:, None]
        logmarg = logsumexp(lw, axis=1)
        W = np.exp(lw - logmarg[:, None])
        pm = (W * nodes).sum(1)
        pv = (W * nodes ** 2).sum(1) - pm ** 2
        return logmarg, pm, pv

    def fit(self, X, Z=None):
        X = np.asarray(X, float)
        J, K = X.shape
        Z = np.zeros((J, 0)) if Z is None else np.asarray(Z, float).reshape(J, -1)
        p = Z.shape[1]
        cum = np.cumsum(X.sum(0) + 0.5)[:-1] / (X.sum() + 0.5 * K)
        k0 = np.log(cum / (1 - cum))
        start = np.concatenate([[k0[0]], np.log(np.diff(k0)), [np.log(0.7)], np.zeros(p)])

        def unpack(par):
            kappa = np.cumsum(np.concatenate([[par[0]], np.exp(par[1:K - 1])]))
            return kappa, np.exp(par[K - 1]), par[K:]

        def nll(par):
            kappa, tau, beta = unpack(par)
            return -self._posterior(kappa, tau, X, Z @ beta if p else None)[0].sum()

        bounds = [(-15.0, 15.0)] + [(-8.0, 4.0)] * (K - 2) + [(np.log(1e-2), np.log(8.0))] + [(-10.0, 10.0)] * p
        res = minimize(nll, start, method="L-BFGS-B", bounds=bounds, options=dict(maxiter=300))
        self.opt_ = res
        self.kappa_, self.tau_, self.beta_ = unpack(res.x)
        _, pm, pv = self._posterior(self.kappa_, self.tau_, X, Z @ self.beta_ if p else None)
        self.theta_, self.theta_sd_ = pm, np.sqrt(pv)
        return self


# ----------------------------------------------------------------------------------------------
# Sensitivity analysis
# ----------------------------------------------------------------------------------------------
def _extreme_mean(theta, psi, nu, Lam, upper=True):
    """Largest (smallest) visible score mean that unit-specific drift with oscillation <= log Lam can
    produce at index theta.  The optimum is a threshold reweighting (weights Lam above/below a cut)."""
    p = family_probs(theta, psi, nu)
    order = np.argsort(nu)
    p, s = p[order], nu[order]
    best = -np.inf if upper else np.inf
    for k in range(len(s) + 1):
        w = np.ones(len(s))
        if upper:
            w[k:] = Lam
        else:
            w[:k] = Lam
        val = (w * p * s).sum() / (w * p).sum()
        best = max(best, val) if upper else min(best, val)
    return best


def drift_interval(theta_hat, psi, nu, Lam):
    """Set of true theta compatible with the index value theta_hat when the unit's voice function may
    depart from the class-wide shape by a factor of at most Lam (Hilbert oscillation <= log Lam).
    Works on the sufficient statistic (the visible score mean), so it is valid for sparse units."""
    if Lam <= 1:
        return theta_hat, theta_hat
    target = mean_score(theta_hat, psi, nu)
    lo = brentq(lambda t: _extreme_mean(t, psi, nu, Lam, True) - target, theta_hat - 50, theta_hat + 1e-9)
    hi = brentq(lambda t: _extreme_mean(t, psi, nu, Lam, False) - target, theta_hat - 1e-9, theta_hat + 50)
    return lo, hi


def breakdown(theta_j, theta_k, psi, nu, Lam_max=1e6):
    """Largest Lam (drift allowed in *each* of the two units) for which the sign of theta_j - theta_k
    cannot be overturned."""
    if theta_j == theta_k:
        return 1.0
    hi_unit, lo_unit = (theta_j, theta_k) if theta_j > theta_k else (theta_k, theta_j)

    def gap(logL):
        L = np.exp(logL)
        return drift_interval(hi_unit, psi, nu, L)[0] - drift_interval(lo_unit, psi, nu, L)[1]

    if gap(np.log(Lam_max)) > 0:
        return Lam_max
    return float(np.exp(brentq(gap, 1e-12, np.log(Lam_max))))


def contamination_interval(theta_hat, psi, nu, eta):
    """Bounds on theta when at most a share eta of the unit's visible ratings are not genuine.

    Uses the visible score mean only (valid for sparse units).  The bounds on the genuine mean are
    truncated to the score range; when a bound reaches the end of the scale the corresponding bound on
    theta is infinite."""
    m = mean_score(theta_hat, psi, nu)
    lo = (m - eta * nu.max()) / (1 - eta)
    hi = (m - eta * nu.min()) / (1 - eta)
    th_lo = -np.inf if lo <= nu.min() else _invert_mean(lo, psi, nu)
    th_hi = np.inf if hi >= nu.max() else _invert_mean(hi, psi, nu)
    return th_lo, th_hi


def min_drift(q, psi, nu):
    """Identified lower bound on a unit's own drift: exp{min_theta osc_h(log q_h - psi_h - theta nu_h)}.

    For a unit whose visible profile q is well estimated.  Equals 1 when the profile fits the class
    model exactly.  Returns (Lambda_min, argmin theta)."""
    r = np.log(np.asarray(q, float)) - psi
    K = len(nu)
    cand = [(r[h] - r[l]) / (nu[h] - nu[l]) for h in range(K) for l in range(h)]
    osc = lambda t: np.ptp(r - t * nu)
    # osc is convex and piecewise linear, so its minimum is at a kink, i.e. at a crossing of two lines;
    # crossings of (max,max) or (min,min) pairs are also kinks, so checking all pairwise crossings suffices
    vals = [osc(t) for t in cand]
    i = int(np.argmin(vals))
    return float(np.exp(vals[i])), float(cand[i])


def profile_drift_set(q, psi, nu, Lam):
    """Sharp set of theta for a well-measured unit with visible profile q under drift bounded by Lam:
    {theta : osc_h(log q_h - psi_h - theta nu_h) <= log Lam}.  Returns (lo, hi), or None if the set is
    empty (the unit's own drift must then exceed Lam)."""
    r = np.log(np.asarray(q, float)) - psi
    lam_min, t0 = min_drift(q, psi, nu)
    if lam_min > Lam * (1 + 1e-12):
        return None
    f = lambda t: np.ptp(r - t * nu) - np.log(Lam)
    span = (np.log(Lam) + np.ptp(r)) / np.min(np.diff(np.sort(nu))) + 1.0
    lo = t0 if f(t0) >= 0 else brentq(f, t0 - span, t0)
    hi = t0 if f(t0) >= 0 else brentq(f, t0, t0 + span)
    return float(lo), float(hi)


# ----------------------------------------------------------------------------------------------
# Simple aggregators used as comparators
# ----------------------------------------------------------------------------------------------
def bayes_average(X, nu=None):
    """Empirical-Bayes shrinkage of the raw mean rating toward the grand mean ('Bayesian average')."""
    X = np.asarray(X, float)
    nu = np.arange(1, X.shape[1] + 1.0) if nu is None else nu
    N = X.sum(1)
    ybar = (X @ nu) / np.maximum(N, 1)
    grand = (X @ nu).sum() / N.sum()
    within = ((X * (nu[None, :] - ybar[:, None]) ** 2).sum() / max(N.sum() - (N > 0).sum(), 1))
    ok = N > 0
    between = max(np.average((ybar[ok] - grand) ** 2, weights=N[ok]) - within * ok.sum() / N.sum(), 1e-6)
    kappa = within / between
    return (N * ybar + kappa * grand) / (N + kappa)


def wilson_lower(X, top=2, z=1.96):
    """Wilson lower confidence bound for the share of ratings in the top `top` categories."""
    X = np.asarray(X, float)
    N = X.sum(1)
    ph = X[:, -top:].sum(1) / np.maximum(N, 1)
    den = 1 + z ** 2 / np.maximum(N, 1)
    centre = ph + z ** 2 / (2 * np.maximum(N, 1))
    rad = z * np.sqrt((ph * (1 - ph) + z ** 2 / (4 * np.maximum(N, 1))) / np.maximum(N, 1))
    return np.where(N > 0, (centre - rad) / den, 0.0)


# ----------------------------------------------------------------------------------------------
# Data-generating process
# ----------------------------------------------------------------------------------------------
def simulate_class(J, rng, c=None, b=None, tau=0.45, nu=None, med_n=6.0, sigma_n=1.6, beta=None, Z=None,
                   drift=None, theta=None, loud=None):
    """One comparison class of J units.

    Latent law:   pi_jh ∝ exp(c_h + theta_j nu_h)                       (what a census of encounters would show)
    Voice:        rho_jh = a_j * b_h * exp(drift_jh)                   (separable up to bounded drift)
    Events:       X_jh ~ Poisson(U_j * pi_jh * rho_jh)                 (rating events)
    `loud` is U_j * a_j (unknown to the analyst).  Returns a dict with the latent truth and the visible table.
    """
    K = 5 if nu is None else len(nu)
    nu = np.arange(1, K + 1.0) if nu is None else np.asarray(nu, float)
    c = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35])) if c is None else np.asarray(c, float)
    b = np.ones(K) if b is None else np.asarray(b, float)
    if theta is None:
        mu = np.zeros(J) if Z is None else Z @ np.asarray(beta, float)
        theta = mu + tau * rng.standard_normal(J)
    pi = family_probs(theta, c, nu)
    if loud is None:
        loud = np.exp(np.log(med_n) + sigma_n * rng.standard_normal(J))
    w = b[None, :] * (np.exp(drift) if drift is not None else 1.0)
    # scale so that the *average* unit has expected count `loud`
    ref = (family_probs(np.zeros(1), c, nu)[0] * b).sum()
    lam = loud[:, None] * pi * w / ref
    X = rng.poisson(lam)
    vis = pi * w
    vis = vis / vis.sum(1, keepdims=True)
    return dict(X=X, theta=theta, pi=pi, latent_mean=pi @ nu, visible=vis, visible_mean=vis @ nu,
                loud=loud, nu=nu, c=c, b=b)


# ----------------------------------------------------------------------------------------------
# Two-level model: units nested in groups (e.g. offices within jurisdictions)
# ----------------------------------------------------------------------------------------------
class TwoLevelRatingScaleEB(RatingScaleEB):
    """theta_j = z_j' beta + u_g(j) + v_j,   u_g ~ N(0, omega^2),  v_j ~ N(0, tau^2).

    Estimation: EM in which unit effects are integrated by adaptive Gauss-Hermite quadrature and each
    group effect by a Laplace approximation around its posterior mode (one-dimensional, log-concave).
    `group_` holds the group index (posterior mode) and `group_sd_` its approximate posterior s.d.
    """

    def fit(self, X, groups, Z=None, max_iter=300, tol=1e-4, verbose=False):
        X = np.asarray(X, float)
        if X.ndim == 2:
            X = X[:, None, :]
        J, S, K = X.shape
        if self.scores is None:
            self.scores = np.arange(1, K + 1, dtype=float)
        nu = self.scores
        Z = np.zeros((J, 0)) if Z is None else np.asarray(Z, float).reshape(J, -1)
        labels, gidx = np.unique(np.asarray(groups), return_inverse=True)
        G = len(labels)
        N, T = X.sum(2), X @ nu
        if np.any(N.sum(1) == 0):
            raise ValueError("drop units without ratings before fitting the two-level model")
        base = RatingScaleEB(scores=nu, n_quad=self.n_quad).fit(X, Z if Z.shape[1] else None)
        psi = np.atleast_2d(base.psi_).copy()
        beta, tau = base.beta_.copy(), base.tau_ * 0.8
        omega = base.tau_ * 0.6
        u = np.zeros(G)
        tot = X.sum(0)                                                        # S x K
        for it in range(max_iter):
            mu = Z @ beta if Z.shape[1] else np.zeros(J)
            for _ in range(8):                                               # Newton for the group modes
                post = self._posterior(psi, mu + u[gidx], tau, X, N, T)
                gj = (post["mean"] - mu - u[gidx]) / tau ** 2
                hj = (post["var"] / tau ** 2 - 1.0) / tau ** 2
                grad = -u / omega ** 2 + np.bincount(gidx, gj, G)
                hess = -1.0 / omega ** 2 + np.bincount(gidx, hj, G)
                step = grad / hess
                u = u - step
                if np.max(np.abs(step)) < 1e-8:
                    break
            post = self._posterior(psi, mu + u[gidx], tau, X, N, T)
            hj = (post["var"] / tau ** 2 - 1.0) / tau ** 2
            Sg = -1.0 / (-1.0 / omega ** 2 + np.bincount(gidx, hj, G))
            d = post["mean"] - mu - u[gidx]
            shrink = post["var"] / tau ** 2
            old = np.concatenate([psi.ravel(), beta, [tau, omega]])
            tau = np.sqrt(np.mean(d ** 2 + post["var"] + (shrink - 1.0) ** 2 * Sg[gidx]))
            omega = np.sqrt(np.mean(u ** 2 + Sg))
            if Z.shape[1]:
                beta = np.linalg.lstsq(Z, post["mean"] - u[gidx], rcond=None)[0]
            for _ in range(2):                                                # iterative scaling for psi
                eta = psi[None, None, :, :] + post["nodes"][:, :, None, None] * nu[None, None, None, :]
                P = np.exp(eta - logsumexp(eta, axis=3, keepdims=True))
                Ep = np.einsum("jq,jqsk->jsk", post["W"], P)
                fitted = (N[:, :, None] * Ep).sum(0)
                psi = psi + np.log(np.maximum(tot, 1e-12) / np.maximum(fitted, 1e-12))
                psi = psi - psi[:, :1]
            # keep the location convention: group and unit effects are centred at zero
            new = np.concatenate([psi.ravel(), beta, [tau, omega]])
            if verbose and it % 20 == 0:
                print(it, round(tau, 4), round(omega, 4), np.max(np.abs(new - old)))
            if np.max(np.abs(new - old)) < tol:
                break
        self.n_iter_ = it + 1
        self.psi_, self.beta_, self.tau_, self.omega_ = (psi[0] if S == 1 else psi), beta, tau, omega
        mu = Z @ beta if Z.shape[1] else np.zeros(J)
        post = self._posterior(psi, mu + u[gidx], tau, X, N, T)
        shrink = post["var"] / tau ** 2
        self.theta_ = post["mean"]
        self.theta_sd_ = np.sqrt(post["var"] + shrink ** 2 * Sg[gidx])
        self.group_labels_, self.group_, self.group_sd_ = labels, u, np.sqrt(Sg)
        self.N_ = N.sum(1)
        return self
