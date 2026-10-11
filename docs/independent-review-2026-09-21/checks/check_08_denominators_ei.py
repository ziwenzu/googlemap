"""Reviewer's conjecture check: with encounter denominators U_c for MANY offices and a selection shape
common across offices (office-specific overall scale a_c allowed, independent of composition), the
identity  U_c = sum_h X_ch / rho_ch  is a Goodman-type ecological regression whose coefficients are
inverse posting rates. The SHAPE theta_h/theta_1 is what is needed to map visible s_c to latent pi_c."""
import numpy as np
from scipy.optimize import nnls
rng = np.random.default_rng(5)
K = 5; y = np.arange(1, 6.0)
rho = np.array([.0040, .0015, .0008, .0012, .0025])        # J-shaped posting rates (unknown to analyst)
for C, scale_sd in [(60, 0.0), (60, 0.5), (400, 0.5)]:
    est = []
    for rep in range(200):
        pi = rng.dirichlet(np.array([1.2, 1, 1.5, 2.5, 3.0]), size=C)
        U = rng.integers(40_000, 600_000, size=C).astype(float)
        a = np.exp(rng.normal(0, scale_sd, C)); a /= np.mean(1 / a) ** -1 * 0 + 1   # office-specific overall scale
        X = rng.poisson(U[:, None] * pi * rho * a[:, None])
        # bias-corrected (Poisson errors-in-variables) normal equations: (X'X - diag(sum X)) theta = X'U
        A = X.T @ X - np.diag(X.sum(0)); b = X.T @ U
        theta = np.linalg.solve(A, b)
        est.append(theta / theta[0])
    est = np.array(est)
    true = (1 / rho) / (1 / rho[0])
    print(f"C={C:4d}, sd(log a_c)={scale_sd}: true shape {np.round(true,2)} | mean est {np.round(est.mean(0),2)} | sd {np.round(est.std(0),2)}")
