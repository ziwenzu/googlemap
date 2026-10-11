# Internal checks deliberately do not silently renormalize invalid user inputs.
.prob <- function(x, name, positive = FALSE) {
  if (!is.numeric(x) || !length(x) || any(!is.finite(x)) ||
      any(x < 0) || (positive && any(x <= 0)) ||
      abs(sum(x) - 1) > 1e-8)
    stop(name, " must be a probability vector.", call. = FALSE)
  invisible(x)
}
.scalar <- function(x, name, lower = -Inf, upper = Inf) {
  if (!is.numeric(x) || length(x) != 1L || !is.finite(x) ||
      x < lower || x > upper)
    stop(name, " is outside its allowed scalar range.", call. = FALSE)
  invisible(x)
}
.expand <- function(x, n, name) {
  if (length(x) == 1L) x <- rep(x, n)
  if (!is.numeric(x) || length(x) != n || any(!is.finite(x)))
    stop(name, " needs one value or exactly one value per cell.", call. = FALSE)
  as.numeric(x)
}
.fields <- function(x, required, name) {
  if (!is.data.frame(x) || !all(required %in% names(x)))
    stop(name, " requires: ", paste(required, collapse = ", "), call. = FALSE)
}
.keys <- function(a, b) paste(a, b, sep = "\034")
.ids <- function(x, name) {
  if (!is.character(x) || anyNA(x) || any(!nzchar(x)) || anyDuplicated(x) ||
      any(grepl("\034", x, fixed = TRUE)))
    stop(name, " must be unique, nonempty character identifiers.", call. = FALSE)
}

#' Standardized aggregate political-trace data
#'
#' A complete cell-by-category grid distinguishes observed zero counts from
#' unavailable cells. This constructor never imputes nonparticipants, discovers
#' a target population, deduplicates records, or infers a sampling design.
#' @param cells Data frame with cell_id, domain_id, instrument_id, unit_type,
#'   frame_status, count_regime, weight, and weight_type.
#' @param categories Data frame with category_id, label, and numeric score.
#'   Row order defines the category order. Scores may be NA for nominal outcomes.
#' @param counts Complete grid with cell_id, category_id, and integer n.
#' @param audit Optional complete category grids for audited cells, with
#'   cell_id, category_id, n, and design. Nonresponse is not an iid-complete audit.
#' @param ordered Whether categories have an interpretable order.
#' @param provenance List recording source, unit mapping, and measurement choices.
#' @return An object of class trace_data.
#' @export
#' @examples
#' x <- prototype_trace("contacts", seed = 12)$data
#' validate_trace_data(x)
trace_data <- function(cells, categories, counts, audit = NULL,
                       ordered = TRUE, provenance = list()) {
  x <- structure(list(cells = cells, categories = categories, counts = counts,
                      audit = audit, ordered = ordered, provenance = provenance),
                 class = "trace_data")
  validate_trace_data(x)
  x
}

#' Validate the aggregate data contract
#' @param x An object returned by trace_data().
#' @return Invisibly TRUE; invalid data raise an error.
#' @export
validate_trace_data <- function(x) {
  if (!inherits(x, "trace_data")) stop("Expected trace_data.", call. = FALSE)
  .fields(x$cells, c("cell_id", "domain_id", "instrument_id", "unit_type",
                    "frame_status", "count_regime", "weight", "weight_type"), "cells")
  .fields(x$categories, c("category_id", "label", "score"), "categories")
  .fields(x$counts, c("cell_id", "category_id", "n"), "counts")
  c <- x$cells; h <- x$categories; z <- x$counts
  .ids(c$cell_id, "cell_id"); .ids(h$category_id, "category_id")
  if (!is.character(h$label) || anyNA(h$label) || any(!nzchar(h$label)))
    stop("Category labels must be nonempty text.")
  if (!nrow(c) || nrow(h) < 2L) stop("Need cells and at least two categories.")
  if (!is.logical(x$ordered) || length(x$ordered) != 1L || is.na(x$ordered))
    stop("ordered must be TRUE or FALSE.")
  for (nm in c("domain_id", "instrument_id", "unit_type", "count_regime", "weight_type"))
    if (!is.character(c[[nm]]) || anyNA(c[[nm]]) || any(!nzchar(c[[nm]])))
      stop("Invalid cell metadata: ", nm)
  statuses <- c("observed", "observed_zero", "unavailable", "unmatched", "not_operating")
  if (anyNA(c$frame_status) || any(!c$frame_status %in% statuses))
    stop("Unknown frame_status.")
  if (!is.numeric(c$weight) || any(!is.na(c$weight) &
      (!is.finite(c$weight) | c$weight <= 0))) stop("Weights must be positive or NA.")
  if (!is.numeric(h$score) || any(!is.na(h$score) & !is.finite(h$score)))
    stop("Scores must be finite or NA.")
  if (x$ordered && (anyNA(h$score) || any(diff(h$score) <= 0)))
    stop("Ordered outcomes need strictly increasing finite scores.")
  expected <- as.vector(outer(c$cell_id, h$category_id, .keys))
  key <- .keys(z$cell_id, z$category_id)
  if (anyNA(z$cell_id) || anyNA(z$category_id) ||
      anyDuplicated(key) || !setequal(key, expected) || length(key) != length(expected))
    stop("counts must contain the complete, unique cell-by-category grid.")
  if (!is.numeric(z$n) || any(!is.na(z$n) &
      (!is.finite(z$n) | z$n < 0 | z$n != floor(z$n))))
    stop("Counts must be nonnegative integers or NA.")
  for (i in seq_len(nrow(c))) {
    v <- z$n[z$cell_id == c$cell_id[i]]
    seen <- c$frame_status[i] %in% c("observed", "observed_zero")
    if (seen && anyNA(v)) stop("Observed cells need counts, including explicit zeros.")
    if (!seen && !all(is.na(v))) stop("Unavailable cells need NA, not artificial zero counts.")
    if (c$frame_status[i] == "observed_zero" && sum(v) != 0)
      stop("observed_zero contradicts positive counts.")
    if (c$frame_status[i] == "observed" && sum(v) == 0)
      stop("A fully observed zero-total cell must be labeled observed_zero.")
  }
  if (!is.null(x$audit)) {
    a <- x$audit
    .fields(a, c("cell_id", "category_id", "n", "design"), "audit")
    if (!nrow(a) || anyNA(a$cell_id) || anyNA(a$category_id) ||
        any(!a$cell_id %in% c$cell_id) || any(!a$category_id %in% h$category_id) ||
        anyDuplicated(.keys(a$cell_id, a$category_id)) ||
        !is.numeric(a$n) || anyNA(a$n) || any(!is.finite(a$n)) ||
        any(a$n < 0 | a$n != floor(a$n))) stop("Invalid audit counts or keys.")
    for (id in unique(a$cell_id)) {
      aa <- a[a$cell_id == id, ]
      if (!setequal(aa$category_id, h$category_id) || nrow(aa) != nrow(h) ||
          anyNA(aa$design) || length(unique(aa$design)) != 1L)
        stop("Each audit needs a complete category grid and one declared design.")
    }
  }
  invisible(TRUE)
}

#' Trace-count diagnostics without a representativeness claim
#' @param x Standardized trace_data.
#' @return Cell-level observed totals and zero-category counts. No latent
#'   satisfaction or population event rate is estimated.
#' @export
trace_diagnostics <- function(x) {
  validate_trace_data(x)
  do.call(rbind, lapply(x$cells$cell_id, function(id) {
    v <- x$counts$n[x$counts$cell_id == id]
    data.frame(cell_id = id, total = if (anyNA(v)) NA_real_ else sum(v),
               zero_categories = if (anyNA(v)) NA_integer_ else sum(v == 0))
  }))
}
