#!/usr/bin/env Rscript
# Export manuscript numbers and MC uncertainty from the saved simulation, without
# rerunning or selecting replications. Run from the project root.
mc <- readRDS("simulation/prototype-output/monte-carlo.rds")
stopifnot(nrow(mc$grid) == 6L, length(mc$raw) == 6L,
          all(vapply(mc$raw, nrow, integer(1)) == mc$B))
value <- function(s, metric) {
  z <- mc$summary[mc$summary$scenario == mc$designs[[s]]$scenario &
                    mc$summary$metric == metric, ]
  stopifnot(nrow(z) == 1L)
  # Verify the export against retained replications, not only a summary table.
  stopifnot(abs(z$estimate - mean(mc$raw[[s]][[metric]])) < 1e-12)
  z
}
metric <- function(name) vapply(seq_len(6), function(s) value(s, name)$estimate, numeric(1))
env <- metric("population_envelope_covered")
zero <- value(1, "false_order")
stopifnot(all(metric("false_order") == 0), all(metric("naive_false_order") == 0))
numbers <- c(PilotB = format(mc$B, big.mark = "{,}", scientific = FALSE),
  PilotEnvelopeMin = sprintf("%.3f", min(env)),
  PilotEnvelopeMax = sprintf("%.3f", max(env)),
  PilotCleanWidth = sprintf("%.3f", value(3, "mean_width")$estimate),
  PilotContaminatedWidth = sprintf("%.3f", value(6, "mean_width")$estimate),
  PilotCleanAbstention = sprintf("%.3f", value(3, "abstention")$estimate),
  PilotContaminatedAbstention = sprintf("%.3f", value(6, "abstention")$estimate),
  PilotZeroUpper = sprintf("%.4f", zero$mc_upper))
writeLines(c("% Generated from monte-carlo.rds by scripts/export-prototype-summary.R.",
  sprintf("\\providecommand{\\%s}{%s}", names(numbers), numbers)),
  "tables/prototype-summary-macros.tex")

# Regenerate the main results table too, so changing a saved study cannot leave
# table values stale while updating the prose and uncertainty appendix.
main_rows <- vapply(seq_len(6), function(s) {
  sprintf("%s & %d & %.3f & %.3f & %.3f & %.3f & %.3f \\\\",
    if (mc$grid$regime[s] == "clean") "Clean" else "Concentrated",
    mc$grid$expected_visible[s], value(s, "population_envelope_covered")$estimate,
    value(s, "false_order")$estimate, value(s, "naive_false_order")$estimate,
    value(s, "abstention")$estimate, value(s, "mean_width")$estimate)
}, character(1))
writeLines(c("% Generated from monte-carlo.rds by scripts/export-prototype-summary.R.",
  "\\begin{table}[!htbp]", "\\centering", "\\captionsetup{skip=0pt}",
  "\\setlength{\\abovecaptionskip}{0pt}",
  "\\caption{Artificial-Data Performance of the Conservative Unanchored Prototype}",
  "\\label{tab:prototype-mc}", "\\footnotesize",
  "\\begin{adjustbox}{max width=\\linewidth}\\begin{threeparttable}",
  "\\setstretch{1}\\renewcommand{\\arraystretch}{1.10}\\setlength{\\tabcolsep}{3pt}",
  "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
  "Design & Expected $N$ & Envelope coverage & False order & Naive false order & Abstention & Width\\\\",
  "\\midrule", main_rows, "\\bottomrule\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  paste0("\\item Notes: ", mc$B, " independent replications per design; three cells and five categories.",
    " Nominal simultaneous coverage is 0.95. Expected $N$ is per-cell visible count;",
    " realized counts are Poisson. Envelope coverage tests containment of all population",
    " marginal CDF endpoints after relaxing cross-cell budget dependence, not numerical",
    " recovery of the anchored joint sharp set. False order is the probability of any",
    " wrong directed comparison. The naive procedure uses $\\Gamma=1$ and no contamination;",
    " it is misspecified. Abstention is the mean fraction of unordered pairs with neither",
    " direction established. Width is the mean outer score-mean interval width on the",
    " 1--5 scale. Monte Carlo uncertainty is reported in Appendix H and retained in the",
    " simulation object. These designs are not calibrated to real populations."),
  "\\end{tablenotes}\\end{threeparttable}\\end{adjustbox}\\end{table}"),
  "tables/prototype-monte-carlo.tex")

# Compact uncertainty table: each coverage column reports estimate (MC SE).
rows <- vapply(seq_len(6), function(s) {
  fmt <- function(name) {
    z <- value(s, name)
    sprintf("%.3f (%.3f)", z$estimate, z$mcse)
  }
  sprintf("%s & %d & %s & %s & %s \\\\",
    if (mc$grid$regime[s] == "clean") "Clean" else "Concentrated",
    mc$grid$expected_visible[s], fmt("reduced_form_covered"),
    fmt("population_envelope_covered"), fmt("truth_cdf_covered"))
}, character(1))
writeLines(c("% Generated from the complete saved simulation; no manual estimates.",
  "\\begin{table}[!htbp]", "\\centering", "\\captionsetup{skip=0pt}",
  "\\setlength{\\abovecaptionskip}{0pt}",
  "\\caption{Distinct Coverage Targets and Monte Carlo Standard Errors}",
  "\\label{tab:prototype-mc-uncertainty}", "\\footnotesize",
  "\\begin{threeparttable}",
  "\\setstretch{1}\\renewcommand{\\arraystretch}{1.10}\\setlength{\\tabcolsep}{4pt}",
  "\\begin{tabular}{@{}lrrrr@{}}\\toprule",
  "Design & Expected $N$ & Reduced form & CDF envelope & True CDF vector\\\\",
  "\\midrule", rows, "\\bottomrule\\end{tabular}",
  "\\begin{tablenotes}[flushleft]",
  paste0("\\item Notes: ", mc$B, " replications per design. Entries are coverage",
    " estimates with Monte Carlo standard errors in parentheses. Reduced-form",
    " coverage concerns all visible category probabilities; envelope containment",
    " concerns all relaxed population CDF endpoints; true-vector coverage concerns",
    " only the generating latent CDFs. These are different events. Exact binomial",
    " Monte Carlo intervals and all replication-level metrics are retained in the",
    " simulation object. An estimate of 1.000 with displayed SE 0.000 does not",
    " establish certain coverage."),
  "\\end{tablenotes}\\end{threeparttable}\\end{table}"),
  "tables/prototype-mc-uncertainty.tex")

# Refresh the population-design figure at its final manuscript size. The source
# coordinates come from the saved design; this draws no new random observations.
grDevices::pdf("figures/prototype-selection-distributions.pdf", width = 6.5,
               height = 3.4, family = "Times", pointsize = 11, useDingbats = FALSE)
graphics::par(mfrow = c(1, 3), mar = c(4, 3.5, 1.4, .6), mgp = c(2.1, .6, 0),
              cex = 1, las = 1)
for (c in 1:3) {
  graphics::plot(1:5, mc$designs[[3]]$pi[c, ], type = "b", pch = 16, lty = 1,
    ylim = c(0, .6), xlab = "Outcome category",
    ylab = if (c == 1) "Probability" else "", xaxt = "n", bty = "l")
  graphics::axis(1, at = 1:5)
  graphics::lines(1:5, mc$designs[[3]]$q[c, ], type = "b", pch = 1,
                  lty = 2, col = "gray40")
  graphics::mtext(LETTERS[c], side = 3, adj = 0, line = .15)
  if (c == 1) graphics::legend("topright", c("Target", "Visible"),
    pch = c(16, 1), lty = c(1, 2), col = c("black", "gray40"), bty = "n", cex = .85)
}
invisible(grDevices::dev.off())
message("Exported saved pilot numbers, both results tables, and vector design figure.")
