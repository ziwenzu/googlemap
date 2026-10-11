#!/usr/bin/env Rscript
# Run from the repository root. Artificial data only; no Google records are read.
# Usage: Rscript simulation/run-prototype-study.R 1000
args <- commandArgs(trailingOnly = TRUE)
B <- if (length(args)) as.integer(args[1]) else 1000L
if (length(B) != 1 || is.na(B) || B < 20L) stop("Supply at least 20 replications.")
if (!file.exists("packages/tracebounds/DESCRIPTION")) stop("Run from the project root.")
for (f in list.files("packages/tracebounds/R", pattern = "\\.R$", full.names = TRUE)) source(f)
out <- "simulation/prototype-output"
dir.create(out, recursive = TRUE, showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)
dir.create("tables", showWarnings = FALSE)
assumptions <- list(positive_visibility = TRUE, identity_measurement = TRUE,
                    one_trace_per_unit = TRUE)
seed_base <- 20260921L
scenarios <- c("ratings", "contacts", "consultation", "requests311", "complaints", "protest")
prototypes <- setNames(lapply(seq_along(scenarios), function(i)
  prototype_trace(scenarios[i], seed_base + i)), scenarios)
saveRDS(prototypes, file.path(out, "six-prototypes.rds"))

# A three-cell, five-category artificial population. These are design inputs,
# not estimates calibrated to any of the named political settings.
pi <- rbind(c(.42, .22, .16, .12, .08),
            c(.20, .20, .20, .20, .20),
            c(.08, .12, .16, .22, .42))
rownames(pi) <- c("cell0", "cell1", "cell2")
rho <- rbind(c(.12, .10, .08, .06, .04),
             c(.04, .06, .08, .10, .12),
             c(.04, .06, .08, .10, .12))
gamma <- 3
grid <- expand.grid(expected_visible = c(25, 100, 500),
                    regime = c("clean", "concentrated"), stringsAsFactors = FALSE)
raw <- summaries <- vector("list", nrow(grid))
designs <- vector("list", nrow(grid))
for (s in seq_len(nrow(grid))) {
  eta <- if (grid$regime[s] == "clean") rep(0, 3) else c(0, .12, 0)
  eta_cap <- if (grid$regime[s] == "clean") 0 else .15
  epsilon <- if (grid$regime[s] == "clean") 0 else .04
  # Total population visible intensities are equal by construction. Known
  # omega is a design input; realized review counts never become budget weights.
  U <- grid$expected_visible[s] * (1 - eta) / rowSums(pi * rho)
  contaminant <- rbind(c(.2, .2, .2, .2, .2),
                      c(1, 0, 0, 0, 0), c(.2, .2, .2, .2, .2))
  d <- trace_design(pi, U, posting = rho, eta = eta, contaminant = contaminant,
       audit_n = 0, scenario = paste(grid$regime[s], grid$expected_visible[s], sep = "_"))
  designs[[s]] <- d
  template <- simulate_traces(d, seed_base + s)$data
  mu <- as.numeric(pi %*% 1:5)
  caps <- pmin(rep(eta_cap, 3), epsilon * sum(d$omega) / d$omega)
  population_envelope <- do.call(rbind, lapply(1:3, function(c)
    unanchored_cdf(d$q[c, ], gamma, caps[c])))
  # row-by-cell order, matching trace_fit$cdf:
  truth_cdf <- unlist(lapply(1:3, function(c) cumsum(pi[c, ])[1:4]), use.names = FALSE)
  metrics <- matrix(NA_real_, B, 8, dimnames = list(NULL,
    c("reduced_form_covered", "population_envelope_covered", "truth_cdf_covered",
      "false_order", "abstention", "mean_width", "naive_false_order", "realized_n")))
  set.seed(seed_base + 10000L * s)
  for (b in seq_len(B)) {
    # Exact marginal equivalent of the independent Poisson thinning and marking
    # construction. All stages remain explicit in simulate_traces() prototypes.
    counts <- matrix(stats::rpois(length(d$q), as.vector(d$q * d$omega)), 3, 5)
    x <- template
    for (c in 1:3) {
      x$counts$n[x$counts$cell_id == rownames(pi)[c]] <- counts[c, ]
      x$cells$frame_status[c] <- if (sum(counts[c, ]) == 0) "observed_zero" else "observed"
    }
    fit <- trace_fit(x, gamma, eta_cap, epsilon, assumptions = assumptions)
    ranks <- trace_rank(fit)
    z <- fit$region$trace
    qvec <- unlist(lapply(1:3, function(c) d$q[c, ]), use.names = FALSE)
    rf <- all(z$lower <= qvec + 1e-12 & z$upper >= qvec - 1e-12)
    env <- all(fit$cdf$lower <= population_envelope$lower + 1e-12 &
               fit$cdf$upper >= population_envelope$upper - 1e-12)
    cov <- all(fit$cdf$lower <= truth_cdf + 1e-12 & fit$cdf$upper >= truth_cdf - 1e-12)
    dec <- ranks$comparisons
    cidx <- match(dec$cell_c, rownames(pi)); didx <- match(dec$cell_d, rownames(pi))
    false <- any(dec$decision == "better" & mu[cidx] <= mu[didx])
    naive <- trace_rank(trace_fit(x, 1, 0, 0, assumptions = assumptions))$comparisons
    naive_false <- any(naive$decision == "better" & mu[cidx] <= mu[didx])
    unresolved <- 1 - sum(dec$decision == "better") / choose(nrow(pi), 2)
    metrics[b, ] <- c(rf, env, cov, false, unresolved,
                     mean(fit$means$upper - fit$means$lower), naive_false, mean(rowSums(counts)))
  }
  raw[[s]] <- as.data.frame(metrics)
  estimate <- colMeans(metrics)
  mcse <- apply(metrics, 2, stats::sd) / sqrt(B)
  summaries[[s]] <- data.frame(scenario = d$scenario, metric = names(estimate),
    estimate = as.numeric(estimate), mcse = as.numeric(mcse), replications = B,
    seed = seed_base + 10000L * s)
  # Exact MC intervals for Bernoulli performance indicators, including all-zero
  # errors. Zero Monte Carlo SE at a boundary is not proof of zero risk.
  for (metric in c("reduced_form_covered", "population_envelope_covered",
                   "truth_cdf_covered", "false_order", "naive_false_order")) {
    idx <- summaries[[s]]$metric == metric
    ci <- stats::binom.test(sum(metrics[, metric]), B)$conf.int
    summaries[[s]]$mc_lower[idx] <- ci[1]
    summaries[[s]]$mc_upper[idx] <- ci[2]
  }
  message("Completed ", d$scenario, ": ", B, " artificial replications.")
}
results <- do.call(rbind, summaries)
saveRDS(list(grid = grid, designs = designs, raw = raw, summary = results,
             seed_base = seed_base, B = B, R = R.version.string),
        file.path(out, "monte-carlo.rds"))
writeLines(capture.output(sessionInfo()), file.path(out, "session-info.txt"))

# Self-contained native TeX table, following the manuscript's table conventions.
rows <- vapply(seq_len(nrow(grid)), function(s) {
  v <- setNames(summaries[[s]]$estimate, summaries[[s]]$metric)
  sprintf("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f \\\\",
    if (grid$regime[s] == "clean") "Clean" else "Concentrated",
    grid$expected_visible[s], v["population_envelope_covered"], v["false_order"],
    v["naive_false_order"], v["abstention"], v["mean_width"])
}, character(1))
table_text <- c(
  "% Generated by simulation/run-prototype-study.R; synthetic prototype only.",
  "\\begin{table}[H]", "\\centering", "\\captionsetup{skip=0pt}",
  "\\setlength{\\abovecaptionskip}{0pt}",
  "\\caption{Artificial-Data Performance of the Conservative Unanchored Prototype}",
  "\\label{tab:prototype-mc}", "\\footnotesize",
  "\\begin{adjustbox}{max width=\\linewidth}\\begin{threeparttable}",
  "\\setstretch{1}\\renewcommand{\\arraystretch}{1.10}\\setlength{\\tabcolsep}{3pt}",
  "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
  "Design & Expected $N$ & Envelope coverage & False order & Naive false order & Abstention & Width\\\\",
  "\\midrule", rows, "\\bottomrule\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  paste0("\\item Notes: ", B, " independent replications per design; three cells and five categories.",
    " Nominal simultaneous coverage is 0.95. Expected $N$ is per-cell visible count;",
    " realized counts are Poisson. Envelope coverage tests containment of all population",
    " marginal CDF endpoints after relaxing cross-cell budget dependence, not numerical",
    " recovery of the anchored joint sharp set. False order is the probability of any",
    " wrong directed comparison. The naive procedure sets selection and contamination",
    " bounds to zero; it is misspecified in these designs. Abstention is the mean fraction",
    " of unordered pairs for which neither direction is established. Width is the mean outer score-mean interval width",
    " on the 1--5 scale. Monte Carlo SEs and exact intervals for binary metrics are",
    " retained in the simulation object. These designs are not calibrated to real populations."),
  "\\end{tablenotes}\\end{threeparttable}\\end{adjustbox}\\end{table}")
writeLines(table_text, "tables/prototype-monte-carlo.tex")

# Compare target and population-visible distributions, not sampled histograms.
# Times, grayscale, vector PDF only; title and explanation live in TeX captions.
grDevices::pdf("figures/prototype-selection-distributions.pdf", width = 7, height = 3.5,
               family = "Times", useDingbats = FALSE)
graphics::par(mfrow = c(1, 3), mar = c(4, 4, 1.4, .7), mgp = c(2.4, .7, 0),
              cex = .85, las = 1)
for (c in 1:3) {
  graphics::plot(1:5, pi[c, ], type = "b", pch = 16, lty = 1, ylim = c(0, .6),
                 xlab = "Outcome category", ylab = if (c == 1) "Probability" else "",
                 xaxt = "n", bty = "l")
  graphics::axis(1, at = 1:5)
  graphics::lines(1:5, designs[[3]]$q[c, ], type = "b", pch = 1, lty = 2, col = "gray40")
  graphics::mtext(LETTERS[c], side = 3, adj = 0, line = .15)
  if (c == 1) graphics::legend("topright", c("Target", "Visible"), pch = c(16, 1),
                               lty = c(1, 2), col = c("black", "gray40"),
                               bty = "n", cex = .8)
}
grDevices::dev.off()
writeLines(c("\\begin{figure}[!htbp]", "\\centering",
  "\\includegraphics[width=\\linewidth]{../figures/prototype-selection-distributions.pdf}",
  "\\caption{Target and Visible Outcome Distributions under Informative Trace Generation}",
  "\\label{fig:prototype-selection}",
  "\\figuremeta{Notes: Constructed population probabilities, not empirical estimates.",
  "Panels A--C denote the three artificial cells in the clean simulation design.",
  "Solid lines are target category probabilities; dashed lines are their population",
  "visible probabilities after selective trace generation. There is no sampling",
  "uncertainty in these design quantities. Contamination is absent in this figure.}",
  "\\end{figure}"), "tables/prototype-distribution-figure.tex")

lines <- c("# Prototype simulation run", "",
  paste("Generated with", R.version.string), "",
  paste("Fixed replications per design:", B, "; six designs."),
  "This is an engineering pilot for the conservative unanchored outer procedure.",
  "It is not validation of the general anchored joint optimizer, which is not implemented.",
  "No real Google Maps data or other human records were used.", "",
  "## Performance and Monte Carlo uncertainty", "",
  "| Design | Metric | Estimate | MC SE | MC 95% interval (binary metrics) |",
  "|---|---|---:|---:|---|")
for (i in seq_len(nrow(results))) {
  z <- results[i, ]
  interval <- if (is.na(z$mc_lower)) "not a binary metric" else
    sprintf("[%.4f, %.4f]", z$mc_lower, z$mc_upper)
  lines <- c(lines, sprintf("| %s | %s | %.4f | %.4f | %s |",
    z$scenario, z$metric, z$estimate, z$mcse, interval))
}
lines <- c(lines, "", "## Scope",
  "Population-envelope coverage evaluates the entire relaxed marginal-CDF envelope,",
  "not only the generating vector. It is a sufficient containment check for the",
  "shared-budget population set, but not an evaluation of a sharp joint optimizer.",
  "The naive comparator uses the same observed counts and sampling-region construction,",
  "but intentionally assumes away informative selection and contamination.",
  "Conservatism, wide intervals, and abstention are retained as results, not tuned away.",
  "Abstention uses unordered pairs: it is zero only if every pair is ordered in one direction.",
  "The initial six DGPs do not create a population mean-order reversal. Zero false-order",
  "rates for both procedures therefore do not establish superiority over the naive comparator.",
  "Separate reversal and misspecification experiments are required; this pilot is retained.",
  "All-zero observed false-order rates retain nonzero upper Monte Carlo limits.",
  "General drift, uncertain anchors, dependence, and estimated weights remain separate",
  "future experiments; six application prototypes are schema examples, not six validated models.")
writeLines(lines, file.path(out, "run-summary.md"))
# Keep manuscript summaries and the final-size vector figure synchronized.
source("scripts/export-prototype-summary.R")
print(results[results$metric %in% c("population_envelope_covered", "false_order",
                                  "naive_false_order", "abstention"), ])
