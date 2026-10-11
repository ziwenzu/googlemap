#!/usr/bin/env Rscript
# Deterministic analytical examples, not observations or Monte Carlo estimates.
# Run from the repository root. Base R only; PDF/Times matches the prototype.
dir.create("figures", showWarnings = FALSE)
open_pdf <- function(name, width = 6.5, height = 3.8, panels = c(1, 1)) {
  grDevices::pdf(file.path("figures", name), width = width, height = height,
                family = "Times", useDingbats = FALSE, pointsize = 12)
  graphics::par(mfrow = panels, mar = c(4, 4.1, .9, .8),
                mgp = c(2.5, .65, 0), las = 1, tcl = -.25, bty = "l", cex = 1)
}

# Comparison 1: identical visible data laws can have different latent outcomes.
q <- c(.25, .25, .5)
pi <- rbind(A = c(.6, .3, .1), B = c(.1, .3, .6))
rho <- rbind(.1 * q / pi[1, ], .1 * q / pi[2, ])
stopifnot(all(rho > 0 & rho <= 1),
          max(abs(pi * rho - rep(.1 * q, each = 2))) < 1e-14)
open_pdf("theory-observational-equivalence.pdf", height = 3.4, panels = c(1, 3))
for (j in 1:3) {
  z <- switch(j, pi, rho, rbind(q, q))
  plot(1:3, z[1, ], type = "o", pch = 16, lwd = 1.2,
       xlim = c(.85, 3.15), ylim = c(0, .68), xaxt = "n",
       xlab = "Outcome category", ylab = c("Latent probability", "Visibility probability",
                                            "Visible probability")[j])
  axis(1, at = 1:3)
  lines(1:3, z[2, ], type = "o", pch = 1, lty = 2, lwd = 1.2)
  mtext(LETTERS[j], 3, adj = 0, line = -.1)
}
dev.off()

# Comparison 2: all-pair log-ratio restrictions cut a hexagon from a reference box.
open_pdf("theory-projective-drift.pdf", width = 4.7, height = 4.5)
plot(NA, xlim = c(-1.28, 1.28), ylim = c(-1.28, 1.28), asp = 1,
     xlab = expression(u[2]/L), ylab = expression(u[3]/L),
     xaxt = "n", yaxt = "n")
axis(1, c(-1, 0, 1)); axis(2, c(-1, 0, 1))
rect(-1, -1, 1, 1, col = "grey95", border = "grey50", lty = 2)
hx <- c(-1, 0, 1, 1, 0, -1); hy <- c(-1, -1, 0, 1, 1, 0)
polygon(hx, hy, col = "grey75", border = "black", lwd = 1.3)
abline(h = 0, v = 0, col = "grey90", lty = 3)
text(0, .4, "All-pair drift set", cex = .95)
points(1, -1, pch = 4, cex = 1.2, lwd = 1.4)
text(.93, -.78, "Range = 2L", adj = 1, cex = .9)
stopifnot(all(vapply(seq_along(hx), function(i)
  diff(range(c(0, hx[i], hy[i]))) == 1, logical(1))))
dev.off()

# Minimum mixture-contamination share for binary q=(1-q2,q2), r=(1,1).
# Interior arguments suffice for these examples; the proof defines endpoints.
cost <- function(p, q2) pmax(0, 1 - q2 / p, 1 - (1 - q2) / (1 - p))
limits <- function(q2, b) c(max(0, (q2 - b) / (1 - b)), min(1, q2 / (1 - b)))
boundary <- function(q1, q2, b) {
  bounds <- limits(q1, b)
  x <- sort(unique(c(seq(bounds[1], bounds[2], length.out = 2001), q1)))
  left <- pmax(0, b - cost(x, q1))
  lo <- pmax(0, (q2 - left) / (1 - left))
  hi <- pmin(1, q2 / (1 - left))
  stopifnot(max(abs(cost(x, q1) + cost(lo, q2) - b)) < 1e-12,
            max(abs(cost(x, q1) + cost(hi, q2) - b)) < 1e-12)
  list(x = x, lo = lo, hi = hi)
}
open_pdf("theory-joint-budget.pdf", height = 3.75, panels = c(1, 2))
for (i in 1:2) {
  q1 <- c(.5, .65)[i]; q2 <- .5; b <- .2
  z <- boundary(q1, q2, b)
  xrange <- limits(q1, b); yrange <- limits(q2, b)
  plot(NA, xlim = if (i == 1) c(.35, .65) else c(.535, .84),
       ylim = c(.35, .65), asp = 1, xlab = expression(p[1]),
       ylab = expression(p[2]), xaxt = "n", yaxt = "n")
  axis(1, at = if (i == 1) c(.375, .5, .625) else c(.5625, .65, .8125),
       labels = if (i == 1) c(".375", ".500", ".625") else c(".5625", ".650", ".8125"),
       cex.axis = .88)
  axis(2, at = c(.375, .5, .625), labels = c(".375", ".500", ".625"), cex.axis = .88)
  rect(xrange[1], yrange[1], xrange[2], yrange[2],
       col = "grey95", border = "grey50", lty = 2)
  polygon(c(z$x, rev(z$x)), c(z$lo, rev(z$hi)),
          col = "grey75", border = "black", lwd = 1.1)
  abline(a = 0, b = 1, lty = 3, col = "grey35")
  mtext(LETTERS[i], 3, adj = 0, line = -.1)
  if (i == 1) {
    points(.625, .625, pch = 4, cex = 1.2, lwd = 1.3)
    text(.62, .585, "Infeasible corner", adj = 1, cex = .86)
    stopifnot(abs(2 * cost(.625, .5) - .4) < 1e-14)
  } else {
    points(.65, .625, pch = 16)
    text(.665, .61, "Gap = .025", adj = 0, cex = .87)
    stopifnot(min(z$x - z$hi) >= .025 - 1e-12,
              abs(q1 - q2 / (1 - b) - .025) < 1e-14)
  }
}
dev.off()
message("Three vector PDFs generated; analytical boundary checks passed. R ", getRversion())
