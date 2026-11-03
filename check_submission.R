# provide path to csv:
# (única modificación respecto al script del curso: la ruta se puede pasar como argumento
#  -> Rscript check_submission.R output/20261104_MiEquipo.csv)
args <- commandArgs(trailingOnly = TRUE)
path <- if (length(args) >= 1) args[1] else "/home/johannes/Documents/Teaching/Ensemble_Seminar5/submission_template.csv"

# define expected properties:
EXPECTED_COLS <- c("forecast_date", "target", "horizon",
                   "q0.025", "q0.25", "q0.5", "q0.75", "q0.975")
LEN_EXP_COLS <- length(EXPECTED_COLS)

TARGETS <- c("demand", "res_demand", "temperature", "wind")

HORIZON_VALS <- paste0(c(36, 40, 44, 60, 64, 68), " hour")

TARGET_VALS <- list(
  demand      = HORIZON_VALS,
  res_demand  = HORIZON_VALS,
  temperature = HORIZON_VALS,
  wind        = HORIZON_VALS
)

TARGET_LEN <- lapply(TARGET_VALS, length)

TARGET_PLAUS <- list(
  demand      = c(0, 200),
  res_demand  = c(-200, 200),
  temperature = c(-50, 50),
  wind        = c(0, 70)
)

COLS_QUANTILES <- c("q0.025", "q0.25", "q0.5", "q0.75", "q0.975")

# a function that read in the csv, conducts checks and
# prints potential problems:
check_df <- function(path) {
  
  df <- tryCatch({
    read.csv(path, sep = ",", stringsAsFactors = FALSE)
  }, error = function(e) {
    if (!file.exists(path)) {
      cat("File not found.\n")
      cat("Stopping early...\n")
      quit(status = 1)
    } else {
      cat("Parse error or no data in given file.\n")
      cat("Stopping early...\n")
      quit(status = 1)
    }
  })
  
  if (nrow(df) == 0) {
    cat("No data in given file.\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  cat("File loaded! Start Checking...\n")
  cat("---------------------------\n")
  col_names <- colnames(df)
  
  ## ---- Checking the Columns ----
  cat("Checking the Columns...\n")
  
  if (length(col_names) != LEN_EXP_COLS) {
    cat("Dataset contains ", length(col_names), "columns. Required are", LEN_EXP_COLS, "\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  if (!setequal(col_names, EXPECTED_COLS)) {
    cat("Dataset does not contain the required columns (or more).\n")
    missing_cols <- setdiff(EXPECTED_COLS, col_names)
    cat("The missing columns are:", missing_cols, "\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  for (i in seq_along(EXPECTED_COLS)) {
    if (EXPECTED_COLS[i] != col_names[i]) {
      cat("Columns not in correct order. Order should be:", EXPECTED_COLS, "\n")
      cat("Your order is:", col_names, "\n")
      cat("Stopping early...\n")
      quit(status = 1)
    }
  }
  
  ## ---- Checking type of columns ----
  cat("Checking type of columns...\n")
  
  forecast_date <- tryCatch({
    as.Date(df$forecast_date, format = "%Y-%m-%d")
  }, error = function(e) NA)
  
  if (any(is.na(forecast_date))) {
    cat("Could not parse Date in format YYYY-MM-DD\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  df$forecast_date <- forecast_date
  
  df$target  <- as.character(df$target)
  df$horizon <- as.character(df$horizon)
  
  # Coerce quantile columns to numeric
  df[COLS_QUANTILES] <- lapply(df[COLS_QUANTILES], function(x) suppressWarnings(as.numeric(x)))
  
  nan_mask <- is.na(df[COLS_QUANTILES])
  
  if (any(nan_mask)) {
    row_has_na  <- which(apply(nan_mask, 1, any))
    first_idx   <- row_has_na[1]
    first_quantile <- COLS_QUANTILES[which(nan_mask[first_idx, ])[1]]
    
    cat(sprintf(
      "Some quantile values are missing or not numeric! First occurrence: target '%s', horizon '%s', quantile '%s'.\n",
      df$target[first_idx], df$horizon[first_idx], first_quantile
    ))
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  # Reject infinity
  if (any(is.infinite(as.matrix(df[COLS_QUANTILES])))) {
    cat("Quantile values cannot be infinite.\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  ## ---- Checking if the Dates make sense ----
  cat("Checking if the Dates make sense...\n")
  
  if (length(unique(df$forecast_date)) > 1) {
    cat("forecast_date needs to be the same in all rows.\n")
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  if (df$forecast_date[1] < Sys.Date()) {
    cat("----WARNING: Forecast date should not be in the past.\n")
    cat("\n")
  }
  
  # Python weekday(): Monday=0 ... Wednesday=2. In R use %u: Monday=1 ... Wednesday=3
  if (as.integer(format(df$forecast_date[1], "%u")) != 3) {
    cat("----WARNING: Forecast date should be a Wednesday.\n")
    cat("\n")
  }
  
  ## ---- Checking targets ----
  cat("Checking targets...\n")
  
  if (!all(df$target %in% TARGETS)) {
    cat(sprintf("Target column can only contain %s. Check spelling.\n",
                paste(TARGETS, collapse = ", ")))
    cat("Stopping early...\n")
    quit(status = 1)
  }
  
  for (target in TARGETS) {
    sub_df <- df[df$target == target, ]
    
    if (nrow(sub_df) != TARGET_LEN[[target]]) {
      cat(sprintf("Exactly %d rows need to have target = %s\n", TARGET_LEN[[target]], target))
      cat("Stopping early...\n")
      quit(status = 1)
    }
    
    if (!identical(sub_df$horizon, TARGET_VALS[[target]])) {
      cat("Target", target, "horizons need to be (in this order):",
          paste(TARGET_VALS[[target]], collapse = ", "), "\n")
      cat("Stopping early...\n")
      quit(status = 1)
    }
    
    plaus <- TARGET_PLAUS[[target]]
    vals <- as.matrix(sub_df[COLS_QUANTILES])
    if (any(vals < plaus[1]) || any(vals > plaus[2])) {
      cat("----WARNING: Implausible values for", target, "detected. You may want to re-check.\n")
      cat("\n")
    }
  }
  
  ## ---- Checking quantiles ----
  cat("Checking quantiles...\n")
  
  for (i in seq_len(nrow(df))) {
    row_vals <- as.numeric(df[i, COLS_QUANTILES])
    diffs <- diff(row_vals)
    
    if (any(is.na(diffs))) {
      cat("Something is wrong with your quantiles.\n")
      cat("Stopping early...\n")
      quit(status = 1)
    }
    
    if (any(diffs < 0)) {
      cat("Predictive quantiles in row", i, "are not ordered correctly (need to be non-decreasing)\n")
      cat("Stopping early...\n")
      quit(status = 1)
    }
  }
  
  cat("---------------------------\n")
  cat("Looks good!\n")
}

# apply that function:
check_df(path)
