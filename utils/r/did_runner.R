# Callaway-Sant'Anna estimation with the R `did` package (Callaway & Sant'Anna's reference
# implementation), driven from Python by utils/r_did_bridge.py.
#
# Usage: Rscript did_runner.R <runs.csv> <long.csv> <cells.csv> <out_dir>
#   runs.csv : run_id, control_flag, control_group, est_method, use_covariates, anticipation,
#              control_fraction, seed, min_e, max_e [, faster_mode, trim_level, treat_flag]
#              treat_flag (optional): a trt_* column of cells.csv; only cells with g > 0 and flag == 1 are treated
#              in that run (default: every cell with g > 0). Used for per-cohort (stacked) and per-ring runs.
#              est_method "dr_trim" = DRDID::drdid_panel with trim.level = trim_level, passed to att_gt as a
#              custom estimator (did's documented option); custom estimators also run in faster_mode.
#   long.csv : uid, year, y [, time-varying covariates]   (outcome panel; did uses each covariate's value in the
#              comparison's base period: g-1-anticipation for post-treatment ATT(g,t), t-1 for varying-base pre-periods)
#   cells.csv: uid, g, <ctrl_* flags>, <trt_* flags>, <covariates>   (g = first treatment year, 0 = not a treated cohort)
#
# Per run it writes, into <out_dir>/<run_id>_*:
#   gt.csv        group, time, att, se                    (ATT(g,t), analytic SE)
#   agg.csv       type, e, att, se                        (simple overall ATT and dynamic event-time ATTs)
#   ids.bin       int32 unit ids in the row order of the influence functions
#   if_simple.bin float64 influence function of the overall ATT (length n)
#   if_dynamic.bin float64 n x E influence functions of the event-time ATTs (column-major)
#   meta.csv      n, E, seconds, n_warnings
#   warnings.txt  every warning raised during estimation
#   error.txt     only if the run failed; the remaining runs still go ahead (read_run raises on it)
suppressMessages({library(did); library(data.table)})
args <- commandArgs(trailingOnly = TRUE)
runs <- fread(args[1]); long <- fread(args[2]); cells <- fread(args[3]); out_dir <- args[4]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
cov_cols <- setdiff(names(cells), c("uid", "g", grep("^(ctrl|trt)_", names(cells), value = TRUE)))   # time-invariant
cov_long <- setdiff(names(long), c("uid", "year", "y"))                                              # time-varying
cov_all <- c(cov_cols, cov_long)
setkey(long, uid)

for (i in seq_len(nrow(runs))) {
  r <- runs[i]
  t0 <- Sys.time()
  p <- file.path(out_dir, r$run_id)
  unlink(paste0(p, "_error.txt"))
  tf <- if ("treat_flag" %in% names(r) && !is.na(r$treat_flag) && nzchar(r$treat_flag)) r$treat_flag else ""
  treated <- if (nzchar(tf)) cells[g > 0 & get(tf) == 1, uid] else cells[g > 0, uid]
  tryCatch({
  if (r$control_flag == "none") {
    ctrl <- integer(0)
  } else {
    ctrl <- cells[get(r$control_flag) == 1, uid]
    if (r$control_fraction < 1) { set.seed(r$seed); ctrl <- sort(sample(ctrl, ceiling(length(ctrl) * r$control_fraction))) }
  }
  keep <- c(treated, ctrl)
  d <- long[J(keep), nomatch = 0L]
  d <- merge(d, cells[, c("uid", "g", cov_cols), with = FALSE], by = "uid")
  d <- as.data.frame(d); d$g <- as.numeric(d$g)
  f <- if (isTRUE(as.logical(r$use_covariates))) as.formula(paste("~", paste(cov_all, collapse = " + "))) else NULL
  fm <- if ("faster_mode" %in% names(r) && !is.na(r$faster_mode)) as.logical(r$faster_mode) else TRUE
  est <- r$est_method
  if (est == "dr_trim") {
    tl <- as.numeric(r$trim_level)
    est <- function(y1, y0, D, covariates, i.weights = NULL, inffunc = TRUE, ...) {
      DRDID::drdid_panel(y1 = y1, y0 = y0, D = D, covariates = covariates, i.weights = i.weights, inffunc = TRUE, trim.level = tl)
    }
    # custom estimators work in did 2.3.0 fast mode (checked: identical to built-in "dr" on mpdta)
  }
  warns <- character(0)
  a <- withCallingHandlers(
    att_gt(yname = "y", tname = "year", idname = "uid", gname = "g", xformla = f, data = d,
           control_group = r$control_group, anticipation = r$anticipation, est_method = est,
           bstrap = FALSE, cband = FALSE, faster_mode = fm),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  ids <- unique(as.data.frame(a$DIDparams$data)[["uid"]])
  s <- withCallingHandlers(aggte(a, type = "simple", bstrap = FALSE, na.rm = TRUE),
                           warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  e <- withCallingHandlers(aggte(a, type = "dynamic", bstrap = FALSE, na.rm = TRUE, min_e = r$min_e, max_e = r$max_e),
                           warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") })
  fwrite(data.table(group = a$group, time = a$t, att = a$att, se = a$se), paste0(p, "_gt.csv"))
  fwrite(rbind(data.table(type = "simple", e = NA_integer_, att = s$overall.att, se = s$overall.se),
               data.table(type = "dynamic", e = e$egt, att = e$att.egt, se = e$se.egt)), paste0(p, "_agg.csv"))
  con <- file(paste0(p, "_ids.bin"), "wb"); writeBin(as.integer(ids), con, size = 4); close(con)
  con <- file(paste0(p, "_if_simple.bin"), "wb"); writeBin(as.numeric(s$inf.function$simple.att), con, size = 8); close(con)
  dyn <- as.matrix(e$inf.function$dynamic.inf.func.e)
  con <- file(paste0(p, "_if_dynamic.bin"), "wb"); writeBin(as.numeric(dyn), con, size = 8); close(con)
  writeLines(if (length(warns)) warns else "none", paste0(p, "_warnings.txt"))
  secs <- round(as.numeric(difftime(Sys.time(), t0, units = "secs")))
  fwrite(data.table(n = length(ids), E = ncol(dyn), seconds = secs, n_warnings = length(warns)), paste0(p, "_meta.csv"))
  cat(sprintf("%s: ATT %.1f SE %.1f | n %d | %d s | %d warnings\n", r$run_id, s$overall.att, s$overall.se, length(ids), secs, length(warns)))
  rm(a, s, e, d, dyn)
  }, error = function(err) {
    writeLines(conditionMessage(err), paste0(p, "_error.txt"))
    cat(sprintf("%s: ERROR %s
", r$run_id, conditionMessage(err)))
  })
  gc()
}
