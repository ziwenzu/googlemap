#!/usr/bin/env Rscript
# Build and check only this package in a temporary directory, not the whole repo.
root <- normalizePath(".", mustWork = TRUE)
pkg <- file.path(root, "packages", "tracebounds")
if (!file.exists(file.path(pkg, "DESCRIPTION"))) stop("Run from the project root.")
build_dir <- tempfile("tracebounds-check-")
dir.create(build_dir)
old <- setwd(build_dir)
Sys.setenv("_R_CHECK_CRAN_INCOMING_" = "false",
           "_R_CHECK_FORCE_SUGGESTS_" = "false")
r <- file.path(R.home("bin"), "R")
status <- system2(r, c("CMD", "build", "--no-build-vignettes", shQuote(pkg)))
if (status != 0) stop("R package build failed; inspect ", build_dir)
archive <- list.files(build_dir, pattern = "\\.tar\\.gz$", full.names = TRUE)
if (length(archive) != 1) stop("Unexpected package archive set.")
status <- system2(r, c("CMD", "check", "--no-manual", "--no-build-vignettes", shQuote(archive)))
setwd(old)
out <- file.path(root, "simulation", "prototype-output")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
file.copy(file.path(build_dir, "tracebounds.Rcheck", "00check.log"),
          file.path(out, "package-check.txt"), overwrite = TRUE)
file.copy(archive, file.path(out, basename(archive)), overwrite = TRUE)
message("Check log and local source archive saved in ", out)
if (status != 0) stop("R CMD check reported a failure.")
