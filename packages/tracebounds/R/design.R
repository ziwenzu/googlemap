.stage <- function(x, dims, name) {
  if (length(x) == 1L) x <- matrix(x, dims[1L], dims[2L])
  if (!is.matrix(x) || !is.numeric(x) || !identical(dim(x), dims) ||
      any(!is.finite(x)) || any(x < 0 | x > 1))
    stop(name, " needs probabilities in a cell-by-category matrix or a scalar.")
  x
}
.with_seed <- function(seed, code) {
  .scalar(seed, "seed", 0, .Machine$integer.max)
  if (seed != floor(seed)) stop("seed must be an integer.")
  had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  if (had_seed) old_seed <- get(".Random.seed", envir = .GlobalEnv)
  on.exit(if (had_seed) assign(".Random.seed", old_seed, envir = .GlobalEnv)
          else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE))
            rm(".Random.seed", envir = .GlobalEnv), add = TRUE)
  set.seed(seed)
  force(code)
}

#' Declare a multistage finite-category data-generating process
#'
#' The baseline has at most one genuine record per target unit. Posting,
#' admission, survival, and capture are sequential Bernoulli stages conditional
#' on category. Independent Poisson contaminating records enter after capture.
#' An independent iid audit samples target outcomes; response may be informative.
#' @param pi Cell-by-category probability matrix, with unique row names.
#' @param opportunity Expected qualifying-unit counts (Poisson), or fixed counts.
#' @param posting,admission,survival,capture Scalars or cell-by-category stage
#'   probabilities. Only their product is identified from a static snapshot.
#' @param eta Contamination shares in the population visible intensity, below one.
#' @param contaminant Cell-by-category contaminant mark distributions; default uniform.
#' @param audit_n Independent audit sample sizes per cell.
#' @param audit_response Category-specific audit response probabilities.
#' @param measurement Column-stochastic recorded-by-latent category matrix.
#' @param regime Either poisson_arrivals or fixed_opportunities.
#' @param scenario,domain_id,instrument_id,unit_type Explicit interpretation labels.
#' @param labels Category labels.
#' @param scores Numeric scores, or NA for unordered categories.
#' @param ordered Whether the outcome is ordered.
#' @return A trace_design containing a completely specified artificial DGP.
#' @export
trace_design <- function(pi, opportunity, posting, admission = 1, survival = 1,
                         capture = 1, eta = 0, contaminant = NULL, audit_n = 0,
                         audit_response = 1, measurement = NULL,
                         regime = c("poisson_arrivals", "fixed_opportunities"),
                         scenario = "synthetic", domain_id = "one_domain",
                         instrument_id = "common_measure", unit_type = "eligible_unit",
                         labels = colnames(pi), scores = seq_len(ncol(pi)),
                         ordered = TRUE) {
  regime <- match.arg(regime)
  if (!is.matrix(pi) || !is.numeric(pi) || nrow(pi) < 1 || ncol(pi) < 2)
    stop("pi must be a cell-by-category matrix.")
  apply(pi, 1L, .prob, name = "pi row")
  if (is.null(rownames(pi))) rownames(pi) <- paste0("cell", seq_len(nrow(pi)) - 1L)
  .ids(rownames(pi), "pi row names")
  if (is.null(labels)) labels <- paste0("category", seq_len(ncol(pi)))
  if (length(labels) != ncol(pi) || anyNA(labels) ||
      length(scores) != ncol(pi)) stop("Invalid category labels or scores.")
  if (!is.logical(ordered) || length(ordered) != 1L || is.na(ordered) ||
      !is.numeric(scores) || (ordered && (any(!is.finite(scores)) || any(diff(scores) <= 0))))
    stop("Invalid ordering or scores.")
  C <- nrow(pi); K <- ncol(pi)
  stages <- lapply(list(posting = posting, admission = admission, survival = survival,
                       capture = capture, audit_response = audit_response),
                   .stage, dims = dim(pi), name = "stage")
  opportunity <- .expand(opportunity, C, "opportunity")
  eta <- .expand(eta, C, "eta")
  audit_n <- .expand(audit_n, C, "audit_n")
  if (any(opportunity <= 0) || any(eta < 0 | eta >= 1) ||
      any(audit_n < 0 | audit_n != floor(audit_n)) ||
      (regime == "fixed_opportunities" && any(opportunity != floor(opportunity))))
    stop("Invalid opportunity, eta, or audit size.")
  if (is.null(contaminant)) contaminant <- matrix(1 / K, C, K)
  if (!is.matrix(contaminant) || !identical(dim(contaminant), dim(pi)))
    stop("contaminant must have the same dimensions as pi.")
  apply(contaminant, 1L, .prob, name = "contaminant row")
  if (is.null(measurement)) measurement <- diag(K)
  if (!is.matrix(measurement) || !is.numeric(measurement) ||
      !identical(dim(measurement), c(K, K))) stop("Invalid measurement matrix.")
  apply(measurement, 2L, .prob, name = "measurement column")
  rho <- Reduce("*", stages[1:4])
  v <- rowSums(pi * rho)
  if (any(v <= 0)) stop("Each cell needs positive genuine visible intensity.")
  sY <- (pi * rho) / v
  s <- sY %*% t(measurement)
  q <- (1 - eta) * s + eta * contaminant
  omega <- opportunity * v / (1 - eta)
  structure(c(list(pi = pi, opportunity = opportunity, eta = eta,
                   contaminant = contaminant, audit_n = audit_n, measurement = measurement,
                   regime = regime, scenario = scenario, domain_id = domain_id,
                   instrument_id = instrument_id, unit_type = unit_type,
                   labels = labels, scores = scores, ordered = ordered,
                   rho = rho, v = v, s = s, q = q, omega = omega), stages),
            class = "trace_design")
}

#' Generate standardized observed data and separately held oracle truth
#' @param design A trace_design.
#' @param seed Nonnegative integer seed. The caller's random state is restored.
#' @return List with data, truth, stages, and audit_flow. Truth is never stored
#'   inside the analysis data object. Known intensity weights are supplied as
#'   declared design information; opportunity denominators are not exposed.
#' @export
#' @examples
#' z <- prototype_trace("ratings", seed = 4)
#' trace_diagnostics(z$data)
simulate_traces <- function(design, seed = 1L) {
  if (!inherits(design, "trace_design")) stop("Expected trace_design.")
  .with_seed(seed, {
    d <- design; C <- nrow(d$pi); K <- ncol(d$pi)
    cats <- paste0("k", seq_len(K)); ids <- rownames(d$pi)
    counts <- audits <- flows <- audit_flows <- vector("list", C)
    for (c in seq_len(C)) {
      N <- if (d$regime == "poisson_arrivals") stats::rpois(1L, d$opportunity[c])
           else d$opportunity[c]
      latent <- as.numeric(stats::rmultinom(1L, N, d$pi[c, ]))
      stages <- list(opportunity = latent)
      now <- latent
      for (nm in c("posting", "admission", "survival", "capture")) {
        now <- stats::rbinom(K, now, d[[nm]][c, ])
        stages[[nm]] <- now
      }
      genuine <- numeric(K)
      for (h in seq_len(K))
        genuine <- genuine + as.numeric(stats::rmultinom(1L, now[h], d$measurement[, h]))
      B <- stats::rpois(1L, d$omega[c] * d$eta[c])
      bad <- as.numeric(stats::rmultinom(1L, B, d$contaminant[c, ]))
      counts[[c]] <- data.frame(cell_id = ids[c], category_id = cats, n = genuine + bad)
      flows[[c]] <- do.call(rbind, lapply(names(stages), function(nm)
        data.frame(cell_id = ids[c], category_id = cats, stage = nm, n = stages[[nm]])))
      if (d$audit_n[c] > 0) {
        a <- as.numeric(stats::rmultinom(1L, d$audit_n[c], d$pi[c, ]))
        responded <- stats::rbinom(K, a, d$audit_response[c, ])
        audits[[c]] <- data.frame(cell_id = ids[c], category_id = cats, n = responded,
          design = if (all(d$audit_response[c, ] == 1)) "iid_complete" else "iid_nonresponse")
        audit_flows[[c]] <- data.frame(cell_id = ids[c], sampled = sum(a),
                                      responded = sum(responded))
      }
    }
    counts <- do.call(rbind, counts)
    total <- vapply(ids, function(id) sum(counts$n[counts$cell_id == id]), numeric(1))
    cells <- data.frame(cell_id = ids, domain_id = d$domain_id,
      instrument_id = d$instrument_id, unit_type = d$unit_type,
      frame_status = ifelse(total == 0, "observed_zero", "observed"),
      count_regime = d$regime, weight = d$omega, weight_type = "known_intensity",
      exposure_value = NA_real_, exposure_status = "unknown")
    categories <- data.frame(category_id = cats, label = d$labels, score = d$scores)
    observed <- trace_data(cells, categories, counts, do.call(rbind, audits), d$ordered,
      list(synthetic = TRUE, scenario = d$scenario, seed = seed,
           unit_mapping = "one_genuine_trace_per_unit_by_construction",
           stages_observable = FALSE, weights = "known_in_simulation_design",
           structural_visibility_zero = any(d$rho == 0),
           measurement_identity = isTRUE(all.equal(d$measurement, diag(K)))))
    truth <- list(pi = d$pi, rho = d$rho, s = d$s, q = d$q,
                  g = (1 - d$eta) * d$s,
                  g_latent = (1 - d$eta) * (d$pi * d$rho) / d$v,
                  eta = d$eta, omega = d$omega, opportunity = d$opportunity,
                  measurement = d$measurement,
                  budget_share = sum(d$omega * d$eta) / sum(d$omega),
                  positive_support = all(d$rho > 0))
    list(data = observed, truth = truth, stages = do.call(rbind, flows),
         audit_flow = do.call(rbind, audit_flows))
  })
}

#' Six artificial political applications under the same data contract
#' @param scenario One of ratings, contacts, consultation, requests311, complaints,
#'   or protest. Complaints deliberately include a structural reporting zero;
#'   protest is nominal and supplies no event-rate identification.
#' @param seed Simulation seed.
#' @return Output of simulate_traces() plus its design.
#' @export
prototype_trace <- function(scenario = c("ratings", "contacts", "consultation",
                            "requests311", "complaints", "protest"), seed = 1L) {
  scenario <- match.arg(scenario)
  if (scenario == "ratings") {
    pi <- rbind(c(.15, .15, .2, .25, .25), c(.1, .1, .15, .25, .4),
                c(.25, .2, .2, .2, .15))
    labels <- c("very_bad", "bad", "middle", "good", "very_good")
    p <- c(.3, .15, .1, .15, .25); unit <- "service_encounter"
  } else {
    pi <- rbind(c(.3, .3, .4), c(.2, .2, .6), c(.5, .3, .2))
    p <- c(.1, .06, .3)
    labels <- switch(scenario, contacts = c("oppose", "neutral", "support"),
      consultation = c("oppose", "neutral", "support"),
      requests311 = c("minor_problem", "moderate_problem", "severe_problem"),
      complaints = c("no_mistreatment", "minor_mistreatment", "severe_mistreatment"),
      protest = c("peaceful", "confrontation", "violence"))
    unit <- switch(scenario, contacts = "resident_issue_period",
      consultation = "eligible_person_proposal", requests311 = "service_problem_episode",
      complaints = "police_contact", protest = "distinct_protest_event")
    if (scenario == "complaints") p[1L] <- 0
  }
  rownames(pi) <- c("cell0", "cell1", "cell2")
  d <- trace_design(pi, opportunity = c(600, 900, 450),
    posting = matrix(rep(p, each = 3L), 3L), admission = .95, survival = .9,
    capture = .98, eta = c(.02, .08, .03),
    audit_n = if (scenario %in% c("ratings", "contacts", "consultation"))
      c(200, 0, 0) else 0, scenario = scenario, domain_id = scenario,
    instrument_id = paste0(scenario, "_synthetic_scale"), unit_type = unit,
    labels = labels, scores = if (scenario == "protest") rep(NA_real_, 3L)
      else seq_len(ncol(pi)), ordered = scenario != "protest")
  result <- simulate_traces(d, seed)
  result$design <- d
  result
}
