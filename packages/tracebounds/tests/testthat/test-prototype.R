assertions <- list(positive_visibility = TRUE, identity_measurement = TRUE,
                   one_trace_per_unit = TRUE)

test_that("six schemas separate interpretation from representation", {
  for (name in c("ratings", "contacts", "consultation", "requests311", "complaints", "protest")) {
    z <- prototype_trace(name, 10)
    expect_true(validate_trace_data(z$data))
    expect_false("pi" %in% names(z$data))
    expect_equal(rowSums(z$truth$q), rep(1, 3), ignore_attr = TRUE)
    expect_equal(z$data$cells$unit_type[1], z$design$unit_type)
    expect_true(all(is.na(z$data$cells$exposure_value)))
    f <- z$stages
    for (id in z$data$cells$cell_id) {
      totals <- vapply(c("opportunity", "posting", "admission", "survival", "capture"),
                       function(stage) sum(f$n[f$cell_id == id & f$stage == stage]), numeric(1))
      expect_true(all(diff(totals) <= 0))
    }
  }
})

test_that("RNG state and deterministic seeds are respected", {
  set.seed(721); before <- .Random.seed
  z <- prototype_trace("contacts", 10)
  expect_identical(before, .Random.seed)
  expect_identical(z, prototype_trace("contacts", 10))
})

test_that("missing, zero, duplicate, and misaligned categories are not silently accepted", {
  x <- prototype_trace("ratings")$data
  x$counts$n[x$counts$cell_id == "cell1"] <- 0
  expect_error(validate_trace_data(x), "observed_zero")
  x$cells$frame_status[2] <- "observed_zero"
  expect_true(validate_trace_data(x))
  z <- trace_region(x, include_audit = FALSE)$trace
  expect_true(all(z$lower[z$cell_id == "cell1"] == 0))
  expect_true(all(z$upper[z$cell_id == "cell1"] == 1))
  x$cells$frame_status[2] <- "unavailable"
  expect_error(validate_trace_data(x), "NA")
  x$counts$n[x$counts$cell_id == "cell1"] <- NA
  expect_true(validate_trace_data(x))
  expect_error(trace_region(x), "unavailable")
  x$counts <- rbind(x$counts, x$counts[1, ])
  expect_error(validate_trace_data(x), "unique")
})

test_that("design and construct gates reject misleading inference", {
  x <- prototype_trace("contacts")$data
  expect_error(trace_fit(x, 3), "Explicitly assert")
  x$cells$weight_type <- "observed_count"
  expect_error(trace_fit(x, 3, .1, .05, assumptions = assertions), "known intensity")
  x <- prototype_trace("contacts")$data
  x$cells$instrument_id[2] <- "different_question"
  expect_error(trace_fit(x, 3, assumptions = assertions), "common domain")
  expect_error(trace_fit(prototype_trace("complaints")$data, 3,
                         assumptions = assertions), "structural zeros")
  expect_error(trace_fit(prototype_trace("protest")$data, 3,
                         assumptions = assertions), "ordered")
  x <- prototype_trace("ratings")$data
  x$cells$count_regime <- "fixed_corpus"
  expect_error(trace_region(x), "Poisson")
  x$cells$count_regime <- "dependent_records"
  expect_error(trace_region(x), "Poisson")
  d <- prototype_trace("ratings")$design
  d$audit_response[1, 1] <- .5
  x <- simulate_traces(d)$data
  expect_error(trace_region(x), "Audit nonresponse")
})

test_that("selection stages cannot be separated from static intensity alone", {
  pi <- rbind(c(.2, .3, .5), c(.4, .4, .2))
  a <- trace_design(pi, 500, posting = .2, survival = .5)
  b <- trace_design(pi, 500, posting = .5, survival = .2)
  expect_equal(a$rho, b$rho)
  expect_equal(a$q, b$q)
  expect_equal(a$omega, b$omega)
})

test_that("anchor inversion and projective invariance hold", {
  pi0 <- c(.2, .3, .5); p <- c(.4, .4, .2); r <- c(.3, .1, .2)
  q0 <- pi0 * r / sum(pi0 * r); q <- p * r / sum(p * r)
  for (ref in 1:3) expect_equal(anchor_invert(q0, pi0, q, ref)$pi, p)
  expect_equal(projective_distance(r, c(1, 2, 3)),
               projective_distance(7 * r, 9 * c(1, 2, 3)))
  expect_error(anchor_invert(c(0, 1), c(.5, .5), c(.2, .8)), "probability")
  expect_error(unanchored_cdf(c(.3, .8)), "probability")
  expect_error(unanchored_cdf(c(.3, .7), Inf), "range")
})

test_that("common selection preserves odds but need not preserve mean rankings", {
  qA <- c(.10, .10, .80); qB <- c(.01, .39, .60)
  inverse <- c(100, 1, 1)
  pA <- qA * inverse / sum(qA * inverse)
  pB <- qB * inverse / sum(qB * inverse)
  expect_gt(sum((1:3) * qA), sum((1:3) * qB))
  expect_lt(sum((1:3) * pA), sum((1:3) * pB))
  expect_equal(pA, c(100, 1, 8) / 109)
  expect_equal(pB, c(100, 39, 60) / 199)
})

test_that("contaminated CDF endpoints have explicit feasible witnesses", {
  q <- c(.1, .2, .3, .4); eta <- .15; gamma <- 3
  ans <- unanchored_cdf(q, gamma, eta)
  for (k in 1:3) {
    A <- 1:k; B <- (k + 1):4
    # Lower: remove as much as possible from A; upper: remove from B.
    for (upper in c(FALSE, TRUE)) {
      remove <- numeric(4); remaining <- eta
      priority <- if (upper) c(B, A) else c(A, B)
      for (h in priority) {
        remove[h] <- min(remaining, q[h]); remaining <- remaining - remove[h]
      }
      g <- q - remove
      weights <- rep(1, 4)
      weights[if (upper) A else B] <- gamma
      p <- g * weights / sum(g * weights)
      endpoint <- if (upper) ans$upper[k] else ans$lower[k]
      expect_equal(sum(p[A]), endpoint, tolerance = 1e-12)
      expect_equal(sum(remove), eta)
    }
  }
  expect_equal(unanchored_cdf(c(0, 1), 3, 0)$lower, 0)
  expect_equal(unanchored_cdf(c(1, 0), 3, 0)$upper, 1)
})

test_that("joint witnesses respect shared budgets and known encounter rates", {
  z <- prototype_trace("ratings", 9); t <- z$truth
  r <- t$rho / t$rho[, 1]
  w <- check_joint_witness(t$q, t$g, r, t$pi[1, ], lambda = 1,
      eta_cap = .1, epsilon = t$budget_share, omega = t$omega,
      t = t$omega / t$opportunity)
  expect_true(w$feasible)
  expect_equal(w$pi, t$pi, ignore_attr = TRUE)
  expect_false(w$audit_region_membership_checked)
  q <- matrix(.5, 3, 2); rr <- matrix(1, 3, 2)
  g <- q; g[2, ] <- c(.3, .5)
  args <- list(q = q, r = rr, pi0 = c(.5, .5), lambda = 1,
               eta_cap = c(0, .2, .2), epsilon = 1 / 15, omega = 1)
  expect_true(do.call(check_joint_witness, c(args, list(g = g)))$feasible)
  g[3, ] <- c(.3, .5)
  expect_false(do.call(check_joint_witness, c(args, list(g = g)))$feasible)
})

test_that("forward measurement is not inverse-matrix correction", {
  d <- prototype_trace("contacts")$design
  d <- trace_design(d$pi, 500, d$posting,
       measurement = matrix(c(.5, .5, 0, .5, .5, 0, 0, 0, 1), 3))
  z <- simulate_traces(d)
  expect_equal(rowSums(z$truth$q), rep(1, 3), ignore_attr = TRUE)
  expect_error(trace_fit(z$data, 3, assumptions = assertions), "measurement error")
})

test_that("rank decisions abstain for empty invalid or wide regions", {
  f <- trace_fit(prototype_trace("ratings")$data, 3, .1, .05, assumptions = assertions)
  expect_false(f$sharp)
  expect_false(f$shared_budget_coupling_retained)
  expect_true(all(trace_rank(f)$comparisons$decision == "abstain"))
  f$means$lower[1] <- 9
  expect_error(trace_rank(f), "Empty or invalid")
  f$feasibility <- "empty"
  expect_error(trace_rank(f), "nonempty")
})

test_that("inference is invariant to input row ordering", {
  x <- prototype_trace("contacts")$data
  f <- trace_fit(x, 3, .1, .05, assumptions = assertions)
  x$counts <- x$counts[nrow(x$counts):1, ]
  h <- trace_fit(x, 3, .1, .05, assumptions = assertions)
  expect_equal(f$cdf, h$cdf)
  expect_equal(f$means, h$means)
})
