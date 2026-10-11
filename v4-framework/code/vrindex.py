"""vrindex: a selection-invariant index from voluntary ratings.

Core objects
------------
RatingScaleEB   hierarchical adjacent-category (rating-scale) model for a units-by-categories count
                table, estimated by marginal maximum likelihood with adaptive Gauss-Hermite quadrature.
                The category effects psi absorb the (unknown) common voice function and the row totals
                absorb unknown encounter volumes; the unit effects theta are the index.
CumulativeEB    same hierarchy with a cumulative-logit (ordered logit) link; used only as a comparator.
drift_interval  sensitivity interval for theta when a unit's visibility function (voice x curation) may
                depart from the class norm by a factor of at most Lambda; optionally one-sided.
breakdown       largest Lambda under which the sign of a pairwise contrast is preserved.
contamination_interval  bounds on theta when up to a share eta of visible ratings are fabricated;
                optionally one-sided (promotional or hostile fabrication).
hilbert_distance  projective distance between two visibility functions (the scale on which Lambda lives).
profile_drift_set, min_drift  sharper drift set for a well-measured unit; lower bound on its own drift.
simulate_class  data-generating process for one comparison class (Poisson rating events).
simulate_sequential  the same class when raters see the running mean of earlier ratings before they rate
                (social influence); returns the ordered rating sequences as well as the table.
SequentialRatingScaleEB  the hierarchical model with an influence term: the rater's effective index is
                theta_j + alpha * (displayed mean - class mean).  Needs the order of ratings within units.
exchangeability_test  permutation diagnostic for sequential dependence of ratings on earlier ratings.
display_fixed_point  long-run displayed mean of a unit under a given influence kernel (population level).
source_consistency  signed-root likelihood-ratio statistic for "the unit has the same index in two sources";
                flags unit-specific curation or fabrication in a source the unit controls.

Only numpy and scipy are required.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy.optimize import minimize, brentq
from scipy.special import logsumexp, expit

__all__ = ["RatingScaleEB", "TwoLevelRatingScaleEB", "SequentialRatingScaleEB", "CumulativeEB", "drift_interval", "breakdown",
           "contamination_interval", "profile_drift_set", "min_drift", "hilbert_distance", "share_favorable", "lr_ordered",
           "simulate_class", "simulate_sequential", "running_means", "exchangeability_test", "display_fixed_point",
           "source_consistency", "family_probs", "mean_score", "bayes_average", "wilson_lower"]


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

    def __init__(self, scores=None, n_quad=25, slopes=False):
        """slopes=True: each stratum after the first gets its own slope kappa_s on theta (kappa_1 = 1), so that a
        displayed source whose contrasts are stretched (Proposition 6c) can be combined with a blind source that
        fixes the scale.  The first stratum must then be the reference (blind) source."""
        self.scores = None if scores is None else np.asarray(scores, float)
        self.n_quad = n_quad
        self.slopes = slopes
        self._x, self._w = hermegauss(n_quad)
        self._logw = np.log(self._w) + 0.5 * self._x ** 2
        self.kappa_ = None

    # -- quadrature machinery ---------------------------------------------------------------
    def _kap(self, S):
        return np.ones(S) if self.kappa_ is None else self.kappa_

    def _moments(self, th, psi):
        """mean and variance of the score in every stratum at theta.  th: (J,), psi: (S,K) -> (J,S)."""
        nu = self.scores
        kap = self._kap(psi.shape[0])
        eta = psi[None, :, :] + (th[:, None] * kap[None, :])[:, :, None] * nu[None, None, :]
        p = np.exp(eta - logsumexp(eta, axis=2, keepdims=True))
        m = (p * nu).sum(2)
        v = (p * nu ** 2).sum(2) - m ** 2
        return m, v

    def _posterior(self, psi, mu, tau, X, N, T):
        nu = self.scores
        kap = self._kap(psi.shape[0])
        th = mu.copy()
        Tk = (T * kap[None, :]).sum(1)                   # sum over strata of kappa_s * T_js
        for _ in range(200):  # vectorised Newton for the posterior mode (strictly concave)
            m, v = self._moments(th, psi)
            g = Tk - (N * m * kap[None, :]).sum(1) - (th - mu) / tau ** 2
            h = -(N * v * kap[None, :] ** 2).sum(1) - 1.0 / tau ** 2
            step = np.clip(g / h, -1.0, 1.0)          # damped Newton: the target is strictly concave
            th = th - step
            if np.max(np.abs(step)) < 1e-10:
                break
        m, v = self._moments(th, psi)
        sd = 1.0 / np.sqrt((N * v * kap[None, :] ** 2).sum(1) + 1.0 / tau ** 2)
        nodes = th[:, None] + sd[:, None] * self._x[None, :]                           # J x Q
        # log-normaliser A_s(kappa_s theta) at every node: J x Q x S
        A = logsumexp(psi[None, None, :, :] + (nodes[:, :, None] * kap[None, None, :])[:, :, :, None] * nu[None, None, None, :], axis=3)
        ll = Tk[:, None] * nodes + (X * psi[None, :, :]).sum((1, 2))[:, None] - (N[:, None, :] * A).sum(2)
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
        if self.slopes and S > 1:
            self.kappa_ = np.concatenate([[1.0], np.exp(par[S * (K - 1) + p + 1:S * (K - 1) + p + S])])
        return psi, beta, tau

    def _negloglik(self, par, X, Z, N, T):
        J, S, K = X.shape
        p = Z.shape[1]
        psi, beta, tau = self._unpack(par, S, K, p)
        mu = Z @ beta if p else np.zeros(J)
        post = self._posterior(psi, mu, tau, X, N, T)
        nu = self.scores
        kap = self._kap(S)
        # gradients by Fisher's identity (posterior expectations of complete-data scores)
        eta = psi[None, None, :, :] + (post["nodes"][:, :, None] * kap[None, None, :])[:, :, :, None] * nu[None, None, None, :]
        P = np.exp(eta - logsumexp(eta, axis=3, keepdims=True))                         # J x Q x S x K
        Ep = np.einsum("jq,jqsk->jsk", post["W"], P)
        g_psi = (X - N[:, :, None] * Ep).sum(0)[:, 1:].ravel()
        d = post["mean"] - mu
        g_beta = Z.T @ (d / tau ** 2) if p else np.zeros(0)
        g_ltau = ((post["var"] + d ** 2) / tau ** 2 - 1.0).sum()
        grad = [g_psi, g_beta, [g_ltau]]
        if self.slopes and S > 1:
            # score for log kappa_s: sum_j E[ theta_j (T_js - N_js m_s(kappa_s theta_j)) ] * kappa_s
            Em = np.einsum("jq,jqs->js", post["W"] * post["nodes"], np.einsum("jqsk,k->jqs", P, nu))   # E[theta * m_s]
            Eth = post["mean"]
            g_lk = ((T * Eth[:, None] - N * Em) * kap[None, :]).sum(0)[1:]
            grad.append(g_lk)
        return -post["logmarg"].sum(), -np.concatenate(grad)

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
        if self.slopes and S > 1:
            start = np.concatenate([start, np.zeros(S - 1)])
            bounds = bounds + [(np.log(0.1), np.log(10.0))] * (S - 1)
        self.kappa_ = None
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
            it = nb + Z.shape[1]
            self.tau_se_ = float(self.tau_ * np.sqrt(max(V[it, it], 0.0)))          # delta method from log tau
            self.psi_, self.beta_, self.tau_ = self._unpack(res.x, S, K, Z.shape[1])   # restore kappa_ after the perturbations
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


def drift_interval(theta_hat, psi, nu, Lam, direction=None):
    """Set of true theta compatible with the index value theta_hat when the unit's visibility function may
    depart from the class norm by a factor of at most Lam (Hilbert oscillation <= log Lam).
    Works on the sufficient statistic (the visible score mean), so it is valid for sparse units.

    direction=None          the departure may favour any categories (two-sided interval)
    direction="favorable"   the departure is nondecreasing in the score (unfavourable ratings suppressed or
                            favourable ones solicited): the truth can only lie below theta_hat
    direction="unfavorable" the mirror image"""
    if direction not in (None, "favorable", "unfavorable"):
        raise ValueError("direction must be None, 'favorable' or 'unfavorable'")
    if Lam <= 1:
        return theta_hat, theta_hat
    target = mean_score(theta_hat, psi, nu)
    lo, hi = theta_hat, theta_hat
    if direction in (None, "favorable"):
        lo = brentq(lambda t: _extreme_mean(t, psi, nu, Lam, True) - target, theta_hat - 50, theta_hat + 1e-9)
    if direction in (None, "unfavorable"):
        hi = brentq(lambda t: _extreme_mean(t, psi, nu, Lam, False) - target, theta_hat - 1e-9, theta_hat + 50)
    return lo, hi


def breakdown(theta_j, theta_k, psi, nu, Lam_max=1e6, suspect="both", direction=None):
    """Largest Lam for which the sign of theta_j - theta_k cannot be overturned.

    suspect="both": drift up to Lam is allowed in each of the two units (default).
    suspect="high" / "low": only the unit with the higher / lower index value is allowed to drift.
    direction is passed to drift_interval for the suspect unit(s)."""
    if suspect not in ("both", "high", "low"):
        raise ValueError("suspect must be 'both', 'high' or 'low'")
    if theta_j == theta_k:
        return 1.0
    hi_unit, lo_unit = (theta_j, theta_k) if theta_j > theta_k else (theta_k, theta_j)

    def gap(logL):
        L = np.exp(logL)
        a = drift_interval(hi_unit, psi, nu, L, direction)[0] if suspect in ("both", "high") else hi_unit
        b = drift_interval(lo_unit, psi, nu, L, direction)[1] if suspect in ("both", "low") else lo_unit
        return a - b

    if gap(np.log(Lam_max)) > 0:
        return Lam_max
    return float(np.exp(brentq(gap, 1e-12, np.log(Lam_max))))


def contamination_interval(theta_hat, psi, nu, eta, direction=None):
    """Bounds on theta when at most a share eta of the unit's visible ratings are fabricated.

    Uses the visible score mean only (valid for sparse units).  Bounds on the genuine mean are truncated to
    the score range; when one reaches the end of the scale the corresponding bound on theta is infinite.
    direction="promotional": fabricated ratings sit in the top category, so the truth lies below theta_hat;
    direction="hostile": they sit in the bottom category, so the truth lies above theta_hat."""
    if direction not in (None, "promotional", "hostile"):
        raise ValueError("direction must be None, 'promotional' or 'hostile'")
    m = mean_score(theta_hat, psi, nu)
    lo = (m - eta * nu.max()) / (1 - eta)
    hi = (m - eta * nu.min()) / (1 - eta)
    th_lo = -np.inf if lo <= nu.min() else _invert_mean(lo, psi, nu)
    th_hi = np.inf if hi >= nu.max() else _invert_mean(hi, psi, nu)
    if direction == "promotional":
        th_hi = theta_hat
    elif direction == "hostile":
        th_lo = theta_hat
    return th_lo, th_hi


def hilbert_distance(x, y):
    """Hilbert projective distance osc_h log(x_h / y_h) between two positive vectors.  It is unchanged when
    either vector is rescaled, which is why it is the natural scale for differences between visibility
    functions: Lambda = exp(distance)."""
    r = np.log(np.asarray(x, float)) - np.log(np.asarray(y, float))
    return float(r.max() - r.min())


def lr_ordered(qj, qk, tol=0.0):
    """+1 if q_j/q_k is nondecreasing in the category (j dominates k under every common visibility function),
    -1 if nonincreasing, 0 otherwise."""
    r = np.diff(np.log(np.asarray(qj, float)) - np.log(np.asarray(qk, float)))
    return 1 if np.all(r >= -tol) else (-1 if np.all(r <= tol) else 0)


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


def share_favorable(X, top=2):
    """Share of ratings in the `top` highest categories (NaN for units without ratings)."""
    X = np.asarray(X, float)
    n = X.sum(1)
    return np.where(n > 0, X[:, -top:].sum(1) / np.maximum(n, 1), np.nan)


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
                   drift=None, theta=None, loud=None, retain=None, fake=None):
    """One comparison class of J units.

    Latent law:   pi_jh ∝ exp(c_h + theta_j nu_h)                       (what a census of encounters would show)
    Voice:        rho_jh = a_j * b_h * exp(drift_jh)                   (separable up to bounded drift)
    Curation:     a voiced rating stays visible with probability retain_jh   (K-vector or J x K array)
    Events:       X_jh ~ Poisson(U_j * pi_jh * rho_jh * retain_jh) + Poisson(fake_jh)
    `loud` is U_j * a_j (unknown to the analyst); `fake` holds the intensities of fabricated ratings (J x K).
    Returns a dict with the latent truth, the visible table and the genuine part of it.
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
    if retain is not None:
        w = w * np.asarray(retain, float)
    # scale so that the *average* unit has expected count `loud`
    ref = (family_probs(np.zeros(1), c, nu)[0] * b).sum()
    lam = loud[:, None] * pi * w / ref
    X = rng.poisson(lam)
    G = X.copy()
    if fake is not None:
        X = X + rng.poisson(np.asarray(fake, float))
    vis = pi * w
    vis = vis / vis.sum(1, keepdims=True)
    return dict(X=X, genuine=G, theta=theta, pi=pi, latent_mean=pi @ nu, visible=vis, visible_mean=vis @ nu,
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


# ----------------------------------------------------------------------------------------------
# Social influence: raters see the running mean of earlier visible ratings before they rate
# ----------------------------------------------------------------------------------------------
def _influence_rating(ystar, m, alpha, gamma, K, rng=None):
    """Expressed rating of a rater whose own evaluation is ystar and who sees a displayed mean m.

    The rater moves a share alpha of the gap toward the display when the display is above the evaluation and a
    share alpha*gamma when it is below (gamma < 1: praise is more contagious than criticism), which gives a
    real-valued rating x; the recorded rating is x rounded at random to one of the two neighbouring categories
    with probabilities equal to the fractional parts, so that E[Y | ystar, m] = x exactly.  With rng=None the
    expected rating x is returned.  Nondecreasing in ystar and in m whenever alpha*max(1, gamma) < 1, which is the
    monotonicity Proposition 6 needs; the influence acts on the star scale, not on the estimator's logit scale."""
    ystar = np.asarray(ystar, float)
    gap = m - ystar
    w = np.where(gap > 0, alpha, alpha * gamma)
    x = np.clip(ystar + w * gap, 1, K)
    if rng is None:
        return x
    lo = np.floor(x)
    return lo + (rng.random(np.shape(x)) < (x - lo))


def simulate_sequential(J, rng, c=None, b=None, tau=0.45, nu=None, med_n=20.0, sigma_n=1.6, theta=None, loud=None,
                        alpha=0.0, gamma=0.5):
    """One comparison class in which ratings arrive in order and each rater sees the running mean so far.

    Voice is the shared visibility function b (self-selection on the evaluation); the expressed rating is then
    pulled toward the display by `_influence_rating`.  The first rater of a unit sees no display.  Returns the
    rating sequences (`seqs`, list of int arrays, categories 1..K), the raters' own evaluations (`seqs0`), the
    visible table `X`, the table that would have been visible without influence (`X0`), and the truth."""
    K = 5 if nu is None else len(nu)
    nu = np.arange(1, K + 1.0) if nu is None else np.asarray(nu, float)
    c = np.log(np.array([0.05, 0.08, 0.17, 0.35, 0.35])) if c is None else np.asarray(c, float)
    b = np.ones(K) if b is None else np.asarray(b, float)
    if theta is None:
        theta = tau * rng.standard_normal(J)
    pi = family_probs(theta, c, nu)
    if loud is None:
        loud = np.exp(np.log(med_n) + sigma_n * rng.standard_normal(J))
    ref = (family_probs(np.zeros(1), c, nu)[0] * b).sum()
    lam = loud * (pi * b).sum(1) / ref                       # expected number of visible ratings
    vis = pi * b; vis /= vis.sum(1, keepdims=True)           # distribution of voiced evaluations
    N = rng.poisson(lam)
    seqs, seqs0, X, X0 = [], [], np.zeros((J, K), int), np.zeros((J, K), int)
    for j in range(J):
        n = int(N[j])
        if n == 0:
            seqs.append(np.zeros(0, int)); seqs0.append(np.zeros(0, int)); continue
        ystar = rng.choice(np.arange(1, K + 1), size=n, p=vis[j]).astype(float)
        y = ystar.copy()
        if alpha > 0:
            s = ystar[0]
            u = rng.random(n)
            for t in range(1, n):
                x = _influence_rating(ystar[t], s / t, alpha, gamma, K)
                lo = np.floor(x)
                y[t] = lo + (u[t] < x - lo)
                s += y[t]
        y = y.astype(int)
        seqs.append(y); seqs0.append(ystar.astype(int))
        X[j] = np.bincount(y, minlength=K + 1)[1:]
        X0[j] = np.bincount(ystar.astype(int), minlength=K + 1)[1:]
    return dict(seqs=seqs, seqs0=seqs0, X=X, X0=X0, theta=theta, pi=pi, latent_mean=pi @ nu, visible_mean=vis @ nu, nu=nu, c=c,
                b=b, loud=loud)


def display_fixed_point(theta, psi, nu, alpha, gamma=0.5, kernel=None, tol=1e-10, max_iter=10000):
    """Long-run displayed mean of a unit whose voiced evaluations follow p(theta; psi) when every rater moves
    toward the display as in `_influence_rating` (or as `kernel(ystar, m)` returns, an expected rating).
    Iterates m <- F(m) = E[Y | m] from the no-influence mean; F is nondecreasing and bounded, so the iteration is
    monotone and stops at a fixed point.  Returns (m*, number of iterations, stable flag)."""
    p = family_probs(theta, psi, nu)
    K = len(nu)
    if kernel is None:
        kernel = lambda ys, m: _influence_rating(ys, m, alpha, gamma, K)      # expected rating given the display
    F = lambda m: float((p * kernel(nu, m)).sum())
    m = float(p @ nu)
    for it in range(max_iter):
        m2 = F(m)
        if abs(m2 - m) < tol:
            break
        m = m2
    stable = F(m + 1e-3) - F(m - 1e-3) < 2e-3            # slope of F below one around the fixed point
    return m, it + 1, bool(stable)


def running_means(seqs):
    """Flatten rating sequences.  Returns (y, d, unit, first) with y the ratings, d the mean of the unit's earlier
    ratings (0 for the first rating, which sees no display), unit the unit index of each rating, and first a
    boolean marking first ratings."""
    lens = np.array([len(s) for s in seqs], int)
    R = int(lens.sum())
    if R == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0, int), np.zeros(0, bool)
    y = np.concatenate([np.asarray(s, float) for s in seqs if len(s)])
    unit = np.repeat(np.arange(len(seqs)), lens)
    pos = np.concatenate([np.arange(n) for n in lens if n])
    cs = np.cumsum(y)
    start_of = np.arange(R) - pos                                   # flat index of the unit's first rating
    cs_before_unit = np.where(start_of > 0, cs[np.maximum(start_of - 1, 0)], 0.0)
    prev = cs - y - cs_before_unit                                  # sum of the unit's earlier ratings
    d = np.where(pos > 0, prev / np.maximum(pos, 1), 0.0)
    return y, d, unit, pos == 0


def exchangeability_test(seqs, n_perm=200, rng=None, min_len=3):
    """Permutation test of sequential dependence: does a rating covary with the mean of the unit's earlier ratings?

    Statistic: pooled within-unit covariance between the rating and the running mean of earlier ratings, both
    demeaned within the unit.  Under the model without influence (and without change in the unit over time) the
    order of a unit's ratings is exchangeable, so permuting each unit's sequence gives the exact null
    distribution.  One-sided (influence and quality drift both make the covariance positive).
    Returns dict(stat, p, null_mean, null_sd, z)."""
    rng = np.random.default_rng(0) if rng is None else rng
    seqs = [np.asarray(s, float) for s in seqs if len(s) >= min_len]
    if not seqs:
        return dict(stat=np.nan, p=np.nan, null_mean=np.nan, null_sd=np.nan, z=np.nan)
    lens = np.array([len(s) for s in seqs])
    unit = np.repeat(np.arange(len(seqs)), lens)
    ybar = np.repeat(np.array([s.mean() for s in seqs]), lens)
    pos = np.concatenate([np.arange(n) for n in lens])
    start_of = np.arange(lens.sum()) - pos

    def stat(yflat):
        cs = np.cumsum(yflat)
        prev = cs - yflat - np.where(start_of > 0, cs[np.maximum(start_of - 1, 0)], 0.0)
        d = prev / np.maximum(pos, 1)
        ok = pos > 0
        return float(np.sum((yflat[ok] - ybar[ok]) * (d[ok] - ybar[ok])) / ok.sum())

    yflat = np.concatenate(seqs)
    s0 = stat(yflat)
    null = np.empty(n_perm)
    for b in range(n_perm):
        key = unit + rng.random(len(yflat))                 # random order within each unit
        null[b] = stat(yflat[np.argsort(key, kind="stable")])
    p = float((np.sum(null >= s0) + 1) / (n_perm + 1))
    sd = float(null.std(ddof=1)) if n_perm > 1 else np.nan
    return dict(stat=s0, p=p, null_mean=float(null.mean()), null_sd=sd, z=float((s0 - null.mean()) / sd) if sd > 0 else np.nan)


class SequentialRatingScaleEB(RatingScaleEB):
    """Rating-scale model with a social-influence term, fitted to ordered ratings.

    p(Y_jt = h) ∝ exp(psi_h + [theta_j + alpha * (m_{j,t-1} - mbar)] nu_h),   theta_j ~ N(z_j'beta, tau^2),
    where m_{j,t-1} is the mean of the unit's earlier visible ratings (the display) and mbar a centring constant
    (the class-wide mean rating).  The first rating of a unit sees no display and gets no shift.  alpha is
    identified from within-unit variation in the display; theta_j is the index net of influence.  Estimated by
    marginal maximum likelihood with adaptive Gauss-Hermite quadrature, as RatingScaleEB.  Attributes after fit:
    psi_, alpha_, beta_, tau_, theta_, theta_sd_, mbar_, and (with se) alpha_se_.
    """

    def _prep(self, seqs):
        y, d, unit, first = running_means(seqs)
        return y.astype(int), d, unit, first

    def _unit_post(self, psi, alpha, mu, tau, y, d, unit, J):
        nu = self.scores
        K = len(nu)
        shift = alpha * d                                   # per-rating shift of the index
        T = np.bincount(unit, nu[y - 1], J)                 # sum of scores per unit
        S = np.bincount(unit, shift * nu[y - 1], J)          # sum of shift * score
        th = mu.copy()
        for _ in range(200):                                # Newton for the posterior mode
            x = th[unit] + shift
            eta = psi[None, :] + x[:, None] * nu[None, :]
            p = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
            m = p @ nu; v = p @ nu ** 2 - m ** 2
            g = T - np.bincount(unit, m, J) - (th - mu) / tau ** 2
            h = -np.bincount(unit, v, J) - 1.0 / tau ** 2
            step = np.clip(g / h, -1.0, 1.0)
            th = th - step
            if np.max(np.abs(step)) < 1e-10:
                break
        x = th[unit] + shift
        eta = psi[None, :] + x[:, None] * nu[None, :]
        p = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
        v = p @ nu ** 2 - (p @ nu) ** 2
        sd = 1.0 / np.sqrt(np.bincount(unit, v, J) + 1.0 / tau ** 2)
        nodes = th[:, None] + sd[:, None] * self._x[None, :]                      # J x Q
        # log-normaliser at every rating and node: R x Q
        A = logsumexp(psi[None, None, :] + (nodes[unit][:, :, None] + shift[:, None, None]) * nu[None, None, :], axis=2)
        Asum = np.zeros((J, self.n_quad))
        np.add.at(Asum, unit, A)
        ll = T[:, None] * nodes + (np.bincount(unit, psi[y - 1], J) + S)[:, None] - Asum
        lp = -0.5 * ((nodes - mu[:, None]) / tau) ** 2 - np.log(tau) - 0.5 * np.log(2 * np.pi)
        lw = self._logw[None, :] + ll + lp + np.log(sd)[:, None]
        logmarg = logsumexp(lw, axis=1)
        W = np.exp(lw - logmarg[:, None])
        pm = (W * nodes).sum(1); pv = (W * nodes ** 2).sum(1) - pm ** 2
        return dict(nodes=nodes, W=W, logmarg=logmarg, mean=pm, var=pv)

    def _negloglik_seq(self, par, y, d, unit, Z, J):
        nu = self.scores; K = len(nu); p_ = Z.shape[1]
        psi = np.concatenate([[0.0], par[:K - 1]])
        alpha = par[K - 1]
        beta = par[K:K + p_]
        tau = np.exp(par[K + p_])
        mu = Z @ beta if p_ else np.zeros(J)
        post = self._unit_post(psi, alpha, mu, tau, y, d, unit, J)
        # posterior expectations of the complete-data score (Fisher's identity)
        shift = alpha * d
        x = post["nodes"][unit] + shift[:, None]                                   # R x Q
        eta = psi[None, None, :] + x[:, :, None] * nu[None, None, :]
        P = np.exp(eta - logsumexp(eta, axis=2, keepdims=True))                    # R x Q x K
        Wr = post["W"][unit]                                                       # R x Q
        Ep = np.einsum("rq,rqk->rk", Wr, P)                                        # R x K
        Em = Ep @ nu
        g_psi = (np.bincount(y - 1, minlength=K) - Ep.sum(0))[1:]
        g_alpha = float(np.sum(d * (nu[y - 1] - Em)))
        dd = post["mean"] - mu
        g_beta = Z.T @ (dd / tau ** 2) if p_ else np.zeros(0)
        g_ltau = ((post["var"] + dd ** 2) / tau ** 2 - 1.0).sum()
        return -post["logmarg"].sum(), -np.concatenate([g_psi, [g_alpha], g_beta, [g_ltau]])

    def fit(self, seqs, Z=None, se=False, alpha_fixed=None):
        y, d, unit, first = self._prep(seqs)
        J = len(seqs)
        K = int(max(int(y.max()) if len(y) else 5, 5)) if self.scores is None else len(self.scores)
        if self.scores is None:
            self.scores = np.arange(1, K + 1, dtype=float)
        nu = self.scores; K = len(nu)
        self.mbar_ = float(nu[y - 1].mean())
        d = np.where(first, 0.0, d - self.mbar_)
        Z = np.zeros((J, 0)) if Z is None else np.asarray(Z, float).reshape(J, -1)
        tot = np.bincount(y - 1, minlength=K) + 0.5
        psi0 = np.log(tot / tot[0])[1:]
        start = np.concatenate([psi0, [0.0 if alpha_fixed is None else alpha_fixed], np.zeros(Z.shape[1]), [np.log(0.5)]])
        abound = (-3.0, 3.0) if alpha_fixed is None else (alpha_fixed, alpha_fixed)
        bounds = [(-15.0, 15.0)] * (K - 1) + [abound] + [(-10.0, 10.0)] * Z.shape[1] + [(np.log(1e-2), np.log(5.0))]
        res = minimize(self._negloglik_seq, start, args=(y, d, unit, Z, J), jac=True, method="L-BFGS-B", bounds=bounds,
                       options=dict(maxiter=1000, ftol=1e-12, gtol=1e-6))
        self.opt_ = res
        self.psi_ = np.concatenate([[0.0], res.x[:K - 1]]); self.alpha_ = float(res.x[K - 1])
        self.beta_ = res.x[K:K + Z.shape[1]]; self.tau_ = float(np.exp(res.x[K + Z.shape[1]]))
        self.alpha_se_ = None
        if se:
            P_ = len(res.x); H = np.zeros((P_, P_)); args = (y, d, unit, Z, J)
            for i in range(P_):
                h = 1e-4 * max(1.0, abs(res.x[i])); e = np.zeros(P_); e[i] = h
                H[:, i] = (self._negloglik_seq(res.x + e, *args)[1] - self._negloglik_seq(res.x - e, *args)[1]) / (2 * h)
            H = 0.5 * (H + H.T)
            try:
                V = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                V = np.linalg.pinv(H)
            self.vcov_ = V; self.alpha_se_ = float(np.sqrt(max(V[K - 1, K - 1], 0.0)))
        mu = Z @ self.beta_ if Z.shape[1] else np.zeros(J)
        post = self._unit_post(self.psi_, self.alpha_, mu, self.tau_, y, d, unit, J)
        self.theta_, self.theta_sd_ = post["mean"], np.sqrt(post["var"])
        self.N_ = np.bincount(unit, minlength=J)
        self.loglik_ = -res.fun
        return self


# ----------------------------------------------------------------------------------------------
# Two sources: a consistency statistic for unit-specific manipulation in one of them
# ----------------------------------------------------------------------------------------------
def source_consistency(X3, psi, nu, a=0, b=1):
    """Signed root of the likelihood-ratio statistic for H0: unit j has the same index in sources a and b.

    X3: (J, S, K) counts; psi: (S, K) category effects (from the stratified fit).  Given psi_r, the conditional
    multinomial log-likelihood of a source depends on theta only through the source's mean score; its supremum
    is finite even when every rating sits in one extreme category, so the statistic is defined for every unit
    with ratings in both sources.  Positive values mean that source a rates the unit higher than source b does,
    relative to what the class model implies.  Approximately N(0,1) under H0 for well-measured units; no shrinkage
    is involved, so the statistic is not confounded with the unit's location.  Returns an array with NaN for
    units that lack ratings in either source."""
    from scipy.optimize import minimize_scalar
    X3 = np.asarray(X3, float); psi = np.atleast_2d(psi); nu = np.asarray(nu, float)
    J = X3.shape[0]

    def loglik(Xr, ps, th):
        return (Xr @ nu) * th + Xr @ ps - Xr.sum() * logsumexp(ps + th * nu)

    def sup(Xr, ps):
        N = Xr.sum(); yb = (Xr @ nu) / N
        if yb <= nu.min() + 1e-12:
            return 0.0, -np.inf
        if yb >= nu.max() - 1e-12:
            return 0.0, np.inf
        th = _invert_mean(yb, ps, nu)
        return loglik(Xr, ps, th), th

    out = np.full(J, np.nan)
    for j in range(J):
        Xa, Xb = X3[j, a], X3[j, b]
        if Xa.sum() == 0 or Xb.sum() == 0:
            continue
        la, ta = sup(Xa, psi[a]); lb, tb = sup(Xb, psi[b])
        if ta == tb:
            out[j] = 0.0; continue
        f = lambda t: -(loglik(Xa, psi[a], t) + loglik(Xb, psi[b], t))
        r = minimize_scalar(f, bounds=(-30.0, 30.0), method="bounded", options=dict(xatol=1e-8))
        lr = max(2.0 * (la + lb + r.fun), 0.0)
        out[j] = np.sign(ta - tb) * np.sqrt(lr)
    return out
