.binomial_interval <- function(x, n, alpha) {
  if (n == 0) return(c(0, 1))
  c(if (x == 0) 0 else stats::qbeta(alpha / 2, x, n - x + 1),
    if (x == n) 1 else stats::qbeta(1 - alpha / 2, x + 1, n - x))
}
.simplex_intervals <- function(z) {
  # One simultaneous coordinate tightening; all expressions use the input box.
  lo <- z[, 1]; hi <- z[, 2]
  cbind(pmax(lo, 1 - (sum(hi) - hi), 0),
        pmin(hi, 1 - (sum(lo) - lo), 1))
}

#' Conservative simultaneous intervals for observable category probabilities
#'
#' Uses exact binomial marginals and a Bonferroni union bound. The implementation
#' accepts independent marked Poisson arrivals or a declared conditional
#' multinomial experiment, not arbitrary dependence or a finite archive census.
#' @param x Standardized trace_data.
#' @param alpha Familywise error level.
#' @param include_audit Include independent, fully responding iid outcome audits.
#'   Complex survey designs and nonignorable audit response are not implemented.
#' @return A trace_region with trace and audit intervals and design metadata.
#' @export
trace_region <- function(x, alpha = .05, include_audit = TRUE) {
  validate_trace_data(x); .scalar(alpha, "alpha", 0, 1)
  if (alpha <= 0 || alpha >= 1) stop("alpha must be between zero and one.")
  if (!is.logical(include_audit) || length(include_audit) != 1 || is.na(include_audit))
    stop("include_audit must be logical.")
  if (any(!x$cells$frame_status %in% c("observed", "observed_zero")))
    stop("Sampling projection cannot silently omit unavailable or unmatched cells.")
  if (any(!x$cells$count_regime %in% c("poisson_arrivals", "conditional_multinomial")))
    stop("This confidence region requires Poisson marks or conditional multinomial sampling.")
  a <- if (include_audit) x$audit else NULL
  if (!is.null(a) && any(a$design != "iid_complete"))
    stop("Audit nonresponse or complex survey designs require a separate audit region.")
  K <- nrow(x$categories)
  tests <- nrow(x$cells) * K + if (is.null(a)) 0 else nrow(a)
  delta <- alpha / tests
  intervals <- function(tab, kind) {
    if (is.null(tab)) return(NULL)
    do.call(rbind, lapply(unique(tab$cell_id), function(id) {
      z <- tab[tab$cell_id == id, ]
      z <- z[match(x$categories$category_id, z$category_id), ]
      N <- sum(z$n)
      v <- t(vapply(z$n, .binomial_interval, numeric(2), n = N, alpha = delta))
      v <- .simplex_intervals(v)
      data.frame(cell_id = id, category_id = z$category_id,
        lower = v[, 1], upper = v[, 2], n = z$n, total = N, kind = kind)
    }))
  }
  structure(list(trace = intervals(x$counts, "trace"), audit = intervals(a, "audit"),
    alpha = alpha, marginal_alpha = delta, sampling = unique(x$cells$count_regime),
    guarantee = "joint_reduced_form_coverage_under_declared_sampling"),
    class = "trace_region")
}

#' Conservative unanchored inference and abstaining comparisons
#'
#' Projects reduced-form intervals through monotone marginal CDF bounds.
#' The shared budget supplies a valid cellwise cap but its cross-cell coupling
#' is then relaxed. The result is an OUTER region, not the anchored joint sharp
#' set. Marginal mean and pairwise bounds may be conservative.
#' @param x Standardized trace_data with one common ordered measurement.
#' @param gamma Finite within-cell visibility-ratio caps.
#' @param eta_cap Cell contamination-share caps.
#' @param epsilon Shared contamination budget. NULL disables the shared constraint.
#' @param alpha Simultaneous error level.
#' @param assumptions Named list explicitly asserting positive_visibility,
#'   identity_measurement, and one_trace_per_unit, all TRUE. These assertions are
#'   requirements, not empirical validations performed by the function.
#' @return A trace_fit with CDF and mean outer bounds, assumptions, and scope labels.
#' @export
#' @examples
#' d <- prototype_trace("ratings", 22)$data
#' fit <- trace_fit(d, gamma = 3, eta_cap = .1, epsilon = .05,
#'   assumptions = list(positive_visibility = TRUE, identity_measurement = TRUE,
#'                      one_trace_per_unit = TRUE))
#' trace_rank(fit)
trace_fit <- function(x, gamma, eta_cap = 0, epsilon = NULL, alpha = .05,
                      assumptions = NULL) {
  validate_trace_data(x)
  required <- c("positive_visibility", "identity_measurement", "one_trace_per_unit")
  if (!is.list(assumptions) || !all(vapply(required, function(nm)
        isTRUE(assumptions[[nm]]), logical(1))))
    stop("Explicitly assert positive_visibility, identity_measurement, and one_trace_per_unit.")
  if (!x$ordered) stop("CDF/score comparisons require an ordered outcome.")
  for (nm in c("domain_id", "instrument_id", "unit_type"))
    if (length(unique(x$cells[[nm]])) != 1L)
      stop("Comparisons require a declared common domain, instrument, and target unit.")
  if (identical(x$provenance$measurement_identity, FALSE))
    stop("Known measurement error contradicts the identity-measurement assertion.")
  if (identical(x$provenance$structural_visibility_zero, TRUE))
    stop("Known structural zeros contradict the positive-visibility assertion.")
  C <- nrow(x$cells); K <- nrow(x$categories)
  gamma <- .expand(gamma, C, "gamma"); caps <- .expand(eta_cap, C, "eta_cap")
  if (any(gamma < 1) || any(caps < 0 | caps >= 1))
    stop("Need finite gamma >= 1 and contamination caps below one.")
  if (!is.null(epsilon)) {
    .scalar(epsilon, "epsilon", 0, 1)
    if (any(x$cells$weight_type != "known_intensity") || anyNA(x$cells$weight))
      stop("A sampling-population budget requires known intensity weights here, not sample counts.")
    caps <- pmin(caps, epsilon * sum(x$cells$weight) / x$cells$weight)
  }
  region <- trace_region(x, alpha, include_audit = FALSE)
  cdf <- means <- vector("list", C)
  scores <- x$categories$score
  for (c in seq_len(C)) {
    id <- x$cells$cell_id[c]
    z <- region$trace[region$trace$cell_id == id, ]
    lower <- upper <- numeric(K - 1L)
    for (k in seq_len(K - 1L)) {
      A <- seq_len(k); B <- (k + 1L):K
      Qlo <- max(sum(z$lower[A]), 1 - sum(z$upper[B]), 0)
      Qhi <- min(sum(z$upper[A]), 1 - sum(z$lower[B]), 1)
      lower[k] <- .cdf_map(Qlo, gamma[c], caps[c], FALSE)
      upper[k] <- .cdf_map(Qhi, gamma[c], caps[c], TRUE)
    }
    # Tiny outward slack protects subsequent strict comparisons from roundoff.
    lower <- pmax(0, lower - 1e-12); upper <- pmin(1, upper + 1e-12)
    cdf[[c]] <- data.frame(cell_id = id, threshold = seq_len(K - 1L),
                           lower = lower, upper = upper)
    means[[c]] <- data.frame(cell_id = id,
      lower = max(scores[1], scores[K] - sum(diff(scores) * upper)),
      upper = min(scores[K], scores[K] - sum(diff(scores) * lower)))
  }
  structure(list(cdf = do.call(rbind, cdf), means = do.call(rbind, means),
    region = region, gamma = gamma, effective_eta_cap = caps, alpha = alpha,
    assumptions = assumptions, mode = "unanchored_outer",
    certificate = "analytic_outer", feasibility = "nonempty_unanchored_model",
    sharp = FALSE, audit_used = FALSE, shared_budget_coupling_retained = FALSE),
    class = "trace_fit")
}

#' Safe pairwise declarations and conservative weak-rank intervals
#' @param fit A trace_fit from the implemented analytic outer procedure.
#' @param tolerance Positive comparison margin for numerical safety.
#' @return Pairwise lower bounds and abstaining decisions, plus weak-rank bounds.
#'   Familywise protection is conditional on the fit's structural and sampling
#'   assumptions. This is not a sharp joint rank set.
#' @export
trace_rank <- function(fit, tolerance = 1e-10) {
  if (!inherits(fit, "trace_fit") || fit$certificate != "analytic_outer" ||
      fit$feasibility != "nonempty_unanchored_model")
    stop("An implemented nonempty outer confidence region is required.")
  .scalar(tolerance, "tolerance", 0)
  m <- fit$means
  if (!nrow(m) || any(!is.finite(m$lower)) || any(!is.finite(m$upper)) ||
      any(m$lower > m$upper)) stop("Empty or invalid bounds: no ranking is permitted.")
  grid <- expand.grid(c = seq_len(nrow(m)), d = seq_len(nrow(m)))
  grid <- grid[grid$c != grid$d, , drop = FALSE]
  lower <- m$lower[grid$c] - m$upper[grid$d]
  pairs <- data.frame(cell_c = m$cell_id[grid$c], cell_d = m$cell_id[grid$d],
                      lower = lower, decision = ifelse(lower > tolerance, "better", "abstain"))
  ranks <- data.frame(cell_id = m$cell_id, lower = 1L, upper = nrow(m))
  for (i in seq_len(nrow(m))) {
    ranks$lower[i] <- 1L + sum(pairs$cell_d == m$cell_id[i] & pairs$decision == "better")
    ranks$upper[i] <- nrow(m) - sum(pairs$cell_c == m$cell_id[i] & pairs$decision == "better")
  }
  list(comparisons = pairs, ranks = ranks, alpha = fit$alpha,
       scope = "simultaneous_outer_weak_rank_intervals")
}
