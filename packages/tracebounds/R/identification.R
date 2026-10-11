#' Hilbert projective distance between positive selection shapes
#' @param x,z Positive numeric vectors of equal length.
#' @return The range of log(x/z), invariant to either vector's scale.
#' @export
projective_distance <- function(x, z) {
  if (!is.numeric(x) || !is.numeric(z) || length(x) != length(z) || length(x) < 2 ||
      any(!is.finite(x)) || any(!is.finite(z)) || any(x <= 0) || any(z <= 0))
    stop("Selection shapes must be equal-length, finite positive vectors.")
  u <- log(x) - log(z)
  diff(range(u))
}

#' Exact clean-anchor inversion
#' @param q0 Positive genuine anchor mark probabilities, not raw contaminated marks.
#' @param pi0 Positive anchor population probabilities.
#' @param q Genuine target mark probabilities.
#' @param reference Normalization category.
#' @return Relative selection and the target distribution under exact transport.
#'   No sampling uncertainty is included.
#' @export
#' @examples
#' anchor_invert(c(.2, .8), c(.5, .5), c(.4, .6))
anchor_invert <- function(q0, pi0, q, reference = 1L) {
  .prob(q0, "q0", TRUE); .prob(pi0, "pi0", TRUE); .prob(q, "q")
  if (length(q0) != length(pi0) || length(q) != length(q0))
    stop("Category dimensions differ.")
  .scalar(reference, "reference", 1, length(q0))
  if (reference != floor(reference)) stop("reference must be an integer.")
  r <- (q0 / pi0) / (q0[reference] / pi0[reference])
  p <- q / r
  list(r = r, pi = p / sum(p), claim = "exact_clean_anchor_no_sampling")
}

.cdf_map <- function(Q, gamma, eta, upper) {
  S <- if (upper) pmin(1, Q / (1 - eta)) else pmax(0, (Q - eta) / (1 - eta))
  if (upper) gamma * S / (gamma * S + 1 - S)
  else S / (S + gamma * (1 - S))
}

#' Sharp marginal cumulative bounds under within-cell selection and contamination
#' @param q Population visible probabilities.
#' @param gamma Finite upper bound on the largest within-cell visibility ratio.
#' @param eta Upper bound on the mixture contamination share, below one.
#' @return A data frame of sharp marginal CDF endpoints. Their Cartesian product
#'   is NOT asserted to be the joint identified set.
#' @export
#' @examples
#' unanchored_cdf(c(.2, .3, .5), gamma = 2, eta = .1)
unanchored_cdf <- function(q, gamma = 1, eta = 0) {
  .prob(q, "q"); .scalar(gamma, "gamma", 1)
  .scalar(eta, "eta", 0, 1)
  if (eta >= 1) stop("eta must be below one.")
  Q <- cumsum(q)[seq_len(length(q) - 1L)]
  data.frame(threshold = seq_along(Q), lower = .cdf_map(Q, gamma, eta, FALSE),
             upper = .cdf_map(Q, gamma, eta, TRUE))
}

#' Check one candidate against the anchored joint feasible system
#'
#' This checks a witness. It does not search for a witness, certify emptiness,
#' or produce sharp endpoints. The audit distribution pi0 is fixed at the
#' supplied candidate; membership in an external audit region is caller-owned.
#' @param q,g,r Cell-by-category matrices. First row is the anchor.
#' @param pi0 Positive anchor population distribution.
#' @param lambda Anchor-to-target projective drift caps; anchor entry must be one.
#' @param eta_cap Cell contamination caps, each below one.
#' @param epsilon Shared intensity-weighted contamination allowance.
#' @param omega Known positive population visible-intensity weights.
#' @param kappa_pi,kappa_s Positive quantitative anchor overlap restrictions.
#' @param t Optional known total trace-to-encounter intensity ratios.
#' @param tolerance Numerical feasibility tolerance; not an optimization certificate.
#' @return Feasibility, maximum violation, candidate latent distributions, and
#'   the check's restricted scope.
#' @export
check_joint_witness <- function(q, g, r, pi0, lambda, eta_cap, epsilon, omega,
                                kappa_pi = 1e-6, kappa_s = 1e-6,
                                t = NULL, tolerance = 1e-8) {
  if (!is.matrix(q) || !is.matrix(g) || !is.matrix(r) ||
      !identical(dim(q), dim(g)) || !identical(dim(q), dim(r)) ||
      any(!is.finite(g)) || any(!is.finite(r)) || any(r <= 0))
    stop("q, g, r require matching finite matrix dimensions and positive r.")
  apply(q, 1L, .prob, name = "q row")
  .prob(pi0, "pi0", TRUE)
  C <- nrow(q); K <- ncol(q)
  if (length(pi0) != K || C < 2 || K < 2) stop("Invalid witness dimensions.")
  lambda <- .expand(lambda, C, "lambda"); omega <- .expand(omega, C, "omega")
  eta_cap <- .expand(eta_cap, C, "eta_cap")
  .scalar(epsilon, "epsilon", 0, 1); .scalar(tolerance, "tolerance", 0)
  .scalar(kappa_pi, "kappa_pi", .Machine$double.eps, 1 / K)
  .scalar(kappa_s, "kappa_s", .Machine$double.eps, 1 / K)
  if (any(lambda < 1) || lambda[1] != 1 || any(omega <= 0) ||
      any(eta_cap < 0 | eta_cap >= 1)) stop("Invalid sensitivity restrictions.")
  mass <- rowSums(g); Z <- rowSums(g / r)
  if (any(Z <= 0) || any(g[1, ] <= 0))
    return(list(feasible = FALSE, violation = Inf, pi = NULL,
                claim = "candidate_check_only"))
  r0 <- (g[1, ] / pi0) / (g[1, 1] / pi0[1])
  violations <- c(-g, g - q, 1 - eta_cap - mass, mass - 1,
    kappa_pi - pi0, kappa_s * mass[1] - g[1, ],
    abs(r[, 1] - 1), abs(log(r[1, ]) - log(r0)),
    vapply(seq_len(C), function(c) projective_distance(r[c, ], r[1, ]) -
             log(lambda[c]), numeric(1)),
    sum(omega * (1 - mass)) / sum(omega) - epsilon)
  if (!is.null(t)) {
    t <- .expand(t, C, "t")
    if (any(t <= 0)) stop("Known rates must be positive.")
    violations <- c(violations, t * Z * apply(r, 1, max) - 1)
  }
  violation <- max(0, violations)
  list(feasible = violation <= tolerance, violation = violation,
       pi = (g / r) / Z, claim = "candidate_check_only",
       audit_region_membership_checked = FALSE)
}
