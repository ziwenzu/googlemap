#!/usr/bin/env python3
"""Reference implementation of exact stable-ratio anchor inversion.

This module implements only the algebraic point-transport result. It does not
estimate an audit distribution, handle sampling uncertainty, or solve the bounded
drift/contamination problem.
"""

import argparse
import json
import math


def validate_probability_vector(values, name):
    values = [float(value) for value in values]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} must contain finite, strictly positive probabilities")
    total = sum(values)
    if not math.isclose(total, 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError(f"{name} must sum to one; observed {total}")
    return values


def relative_selection(anchor_observed, anchor_latent, reference=0):
    """Recover b_h / b_reference from an observed and latent anchor histogram."""
    q = validate_probability_vector(anchor_observed, "anchor_observed")
    pi = validate_probability_vector(anchor_latent, "anchor_latent")
    if len(q) != len(pi):
        raise ValueError("Anchor vectors must have the same number of categories")
    if not 0 <= reference < len(q):
        raise ValueError("Reference category is out of range")
    return [(q_h / q[reference]) / (pi_h / pi[reference]) for q_h, pi_h in zip(q, pi)]


def invert_observed(observed, relative_selection_factors):
    """Recover a latent category distribution by normalization of q_h / b_h."""
    q = validate_probability_vector(observed, "observed")
    factors = [float(value) for value in relative_selection_factors]
    if len(q) != len(factors):
        raise ValueError("Observed vector and selection factors must have equal length")
    if any(not math.isfinite(value) or value <= 0 for value in factors):
        raise ValueError("Selection factors must be finite and strictly positive")
    unnormalized = [q_h / b_h for q_h, b_h in zip(q, factors)]
    total = sum(unnormalized)
    return [value / total for value in unnormalized]


def parse_vector(value):
    return [float(item.strip()) for item in value.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-observed", required=True, type=parse_vector)
    parser.add_argument("--anchor-latent", required=True, type=parse_vector)
    parser.add_argument("--target-observed", required=True, type=parse_vector)
    parser.add_argument("--reference", type=int, default=1, help="One-based reference category")
    args = parser.parse_args()

    factors = relative_selection(
        args.anchor_observed, args.anchor_latent, reference=args.reference - 1
    )
    recovered = invert_observed(args.target_observed, factors)
    print(
        json.dumps(
            {
                "relative_selection_factors": factors,
                "recovered_target_distribution": recovered,
                "warning": "Exact-stability algebra only; sampling uncertainty is not included.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
