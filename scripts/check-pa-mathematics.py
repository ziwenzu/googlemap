#!/usr/bin/env python3
"""Reproducible algebra diagnostics, NOT a proof audit or coverage simulation."""
from fractions import Fraction
from math import comb, exp, isclose, log
from random import Random
import unittest


def normalize(values):
    total = sum(values)
    return [value / total for value in values]


def inverse(g, r):
    return normalize([x / y for x, y in zip(g, r)])


def distance(x, y):
    differences = [log(a / b) for a, b in zip(x, y)]
    return max(differences) - min(differences)


def binomial(n, x, p):
    return comb(n, x) * p**x * (1 - p)**(n - x)


def cp_interval(n, x, error):
    if n == 0:
        return 0.0, 1.0
    # Invert the two binomial tail tests directly, without special libraries.
    def solve_tail(start, end, increasing):
        low, high = 0.0, 1.0
        for _ in range(65):
            mid = (low + high) / 2
            tail = sum(binomial(n, j, mid) for j in range(start, end + 1))
            if (tail < error / 2) == increasing:
                low = mid
            else:
                high = mid
        return (low + high) / 2
    lower = 0.0 if x == 0 else solve_tail(x, n, True)
    upper = 1.0 if x == n else solve_tail(0, x, False)
    return lower, upper


class MathematicsChecks(unittest.TestCase):
    def setUp(self):
        self.random = Random(90621)

    def vector(self, k=5):
        return normalize([self.random.random() + .01 for _ in range(k)])

    def assertVectorClose(self, a, b):
        for x, y in zip(a, b):
            self.assertTrue(isclose(x, y, abs_tol=1e-11, rel_tol=1e-11))

    def test_constructive_intensities_and_known_rate(self):
        for _ in range(500):
            s, f = self.vector(), self.vector()
            eta = self.random.uniform(0, .4)
            g = [(1 - eta) * x for x in s]
            r = [exp(self.random.uniform(-2, 2)) for _ in s]
            r = [x / r[0] for x in r]
            pi = inverse(g, r)
            z = sum(x / y for x, y in zip(g, r))
            a = .8 / max(r)
            omega = 12.3
            encounter = omega * z / a
            rho = [a * x for x in r]
            self.assertTrue(all(0 < x <= 1 for x in rho))
            generated = [encounter * p * v + omega * eta * f_h
                         for p, v, f_h in zip(pi, rho, f)]
            expected = [omega * (x + eta * y) for x, y in zip(g, f)]
            self.assertVectorClose(generated, expected)
            t = omega / encounter
            self.assertAlmostEqual(t * z * max(r), .8)
            self.assertVectorClose([t * z * x for x in r], rho)

    def test_reference_and_scale_invariance(self):
        for _ in range(500):
            x, y = self.vector(), self.vector()
            original = distance(x, y)
            for reference in range(5):
                self.assertAlmostEqual(original, distance(
                    [a / x[reference] for a in x],
                    [b / y[reference] for b in y]))
            self.assertAlmostEqual(original, distance([4 * a for a in x], y))

    def test_clean_unanchored_cdf_endpoints(self):
        for _ in range(200):
            q = self.vector()
            gamma = exp(self.random.uniform(0, 3))
            for k in range(1, 5):
                Q = sum(q[:k])
                lower = Q / (Q + gamma * (1 - Q))
                upper = gamma * Q / (gamma * Q + 1 - Q)
                low_r = [1 if h < k else 1 / gamma for h in range(5)]
                high_r = [1 / gamma if h < k else 1 for h in range(5)]
                self.assertAlmostEqual(sum(inverse(q, low_r)[:k]), lower)
                self.assertAlmostEqual(sum(inverse(q, high_r)[:k]), upper)
                random_r = [exp(self.random.uniform(0, log(gamma))) for _ in q]
                actual = sum(inverse(q, random_r)[:k])
                self.assertLessEqual(lower - 1e-12, actual)
                self.assertLessEqual(actual, upper + 1e-12)
        for q in ([0, 1], [1, 0]):
            self.assertVectorClose(inverse(q, [1, 10]), q)

    def test_ordinal_summation_by_parts(self):
        for _ in range(500):
            p, q = self.vector(), self.vector()
            scores = sorted(self.random.uniform(-3, 8) for _ in p)
            direct = sum(y * (a - b) for y, a, b in zip(scores, p, q))
            transformed = -sum((scores[k + 1] - scores[k]) *
                               (sum(p[:k + 1]) - sum(q[:k + 1]))
                               for k in range(4))
            self.assertAlmostEqual(direct, transformed)

    def test_contaminated_cdf_attainment_and_boundary_cases(self):
        # Check the constructive endpoint proof, including exhausted groups.
        cases = [([0, 1], .2, 3), ([1, 0], .2, 3),
                 ([.1, .2, .7], .5, 1), ([.1, .2, .7], .8, 4)]
        cases += [(self.vector(), self.random.uniform(0, .9),
                   exp(self.random.uniform(0, 3))) for _ in range(250)]
        for q, cap, gamma in cases:
            mass = 1 - cap
            for k in range(1, len(q)):
                Q = sum(q[:k])
                targets = (max(0, Q - cap), min(Q, mass))
                for upper, group_mass in enumerate(targets):
                    g = [x * group_mass / Q if Q else 0 for x in q[:k]]
                    g += [x * (mass - group_mass) / (1 - Q)
                          if Q < 1 else 0 for x in q[k:]]
                    self.assertAlmostEqual(sum(g), mass)
                    self.assertTrue(all(-1e-12 <= x <= y + 1e-12
                                        for x, y in zip(g, q)))
                    w = ([gamma if upper else 1] * k +
                         [1 if upper else gamma] * (len(q) - k))
                    pi = normalize([x * y for x, y in zip(g, w)])
                    S = group_mass / mass
                    bound = (gamma * S / (gamma * S + 1 - S) if upper
                             else S / (S + gamma * (1 - S)))
                    self.assertAlmostEqual(sum(pi[:k]), bound)
                    if cap:
                        residual = [(y - x) / cap for x, y in zip(g, q)]
                        self.assertAlmostEqual(sum(residual), 1)
                    self.assertLessEqual(distance([1 / x for x in w],
                                                  [1] * len(q)), log(gamma) + 1e-12)

    def test_measurement_kernel_construction(self):
        for _ in range(500):
            columns = [self.vector() for _ in range(5)]
            latent_g = [.7 * x for x in self.vector()]
            recorded_g = [sum(columns[h][j] * latent_g[h] for h in range(5))
                          for j in range(5)]
            f = self.vector()
            q = [x + .3 * y for x, y in zip(recorded_g, f)]
            self.assertAlmostEqual(sum(recorded_g), .7)
            self.assertAlmostEqual(sum(q), 1)
            self.assertTrue(all(a <= b for a, b in zip(recorded_g, q)))
        # A singular kernel is still a valid forward measurement law.
        columns = [[.2] * 5] * 5
        latent_g = [.1, .1, .1, .1, .3]
        self.assertVectorClose(
            [sum(columns[h][j] * latent_g[h] for h in range(5)) for j in range(5)],
            [.14] * 5)

    def test_mean_reversal_and_shared_budget_exact_arithmetic(self):
        self.assertLess(Fraction(126, 109), Fraction(358, 199))
        self.assertGreater(Fraction(270, 100), Fraction(259, 100))
        required_eta = 1 - Fraction(1, 2) / Fraction(5, 8)
        self.assertEqual(required_eta, Fraction(1, 5))
        self.assertGreater(2 * required_eta, Fraction(1, 5))

    def test_binomial_interval_grid_coverage(self):
        # Deterministic enumeration checks the component interval only.
        # It is not a Monte Carlo study of the proposed projected region.
        error = .05
        self.assertEqual(cp_interval(0, 0, error), (0, 1))
        for n in (1, 2, 5, 10, 20):
            intervals = [cp_interval(n, x, error) for x in range(n + 1)]
            for j in range(501):
                p = j / 500
                coverage = sum(binomial(n, x, p) for x, (lo, hi) in
                               enumerate(intervals) if lo - 1e-13 <= p <= hi + 1e-13)
                self.assertGreaterEqual(coverage, 1 - error - 1e-11)

    def test_binary_joint_rank_with_overlapping_marginals(self):
        # Exact analytical benchmark; global validity is proved by concavity
        # in manuscript/theory-geometry.tex, not by this finite check.
        a, b, budget = Fraction(13, 20), Fraction(1, 2), Fraction(1, 5)
        contrast_ends = (a - b / (1 - budget),
                         (a - budget) / (1 - budget) - b)
        self.assertEqual(min(contrast_ends), Fraction(1, 40))
        marginal_gap = (a - budget) / (1 - budget) - b / (1 - budget)
        self.assertEqual(marginal_gap, -Fraction(1, 16))
        # Rational points check the display curve and mixture witnesses.
        def cost(p, q):
            return max(0, 1 - q / p, 1 - (1 - q) / (1 - p))
        for j in range(101):
            e = budget * Fraction(j, 100)
            p = (a - e) / (1 - e)
            q = b / (1 - budget + e)
            self.assertGreaterEqual(p - q, Fraction(1, 40))
            self.assertEqual(cost(p, a) + cost(q, b), budget)

    def test_projective_hexagon_excludes_reference_box_corner(self):
        vertices = [(-1, -1), (0, -1), (1, 0), (1, 1), (0, 1), (-1, 0)]
        for x, y in vertices:
            self.assertEqual(max(0, x, y) - min(0, x, y), 1)
        self.assertEqual(max(0, 1, -1) - min(0, 1, -1), 2)
        # For a finite rational display grid, compare all-pair restrictions.
        for x in range(-12, 13):
            for y in range(-12, 13):
                self.assertEqual(max(0, x, y) - min(0, x, y) <= 10,
                                 abs(x) <= 10 and abs(y) <= 10 and abs(x-y) <= 10)

    def test_same_poisson_law_distinct_latent_means(self):
        q = [Fraction(1, 4), Fraction(1, 4), Fraction(1, 2)]
        latent = [[Fraction(6, 10), Fraction(3, 10), Fraction(1, 10)],
                  [Fraction(1, 10), Fraction(3, 10), Fraction(6, 10)]]
        for pi in latent:
            rho = [Fraction(1, 10) * x / p for x, p in zip(q, pi)]
            self.assertTrue(all(0 < value <= 1 for value in rho))
            self.assertEqual([1000 * p * r for p, r in zip(pi, rho)],
                             [25, 25, 50])
        self.assertEqual(sum((h+1)*p for h, p in enumerate(latent[0])), Fraction(3, 2))
        self.assertEqual(sum((h+1)*p for h, p in enumerate(latent[1])), Fraction(5, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
