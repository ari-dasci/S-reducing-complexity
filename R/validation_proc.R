#!/usr/bin/env Rscript
if (sys.nframe() == 0){
  args <- commandArgs(TRUE)
  purrr::quietly(devtools::load_all)(".")
  verbose <- as.logical(args[3])
  options(keras.fit_verbose = if (verbose) 2 else 0)

  dataset_f <- dataset_list()[[args[1]]]

  requiredArgs <- list(
    dataset_f = dataset_f,
    method = args[2],
    verbose = verbose,
    name = args[1],
    autosave = as.logical(args[4]),
    loadsave = as.logical(args[5])
  )

  extraArgs <- list()
  if (length(args) > 6) {
    namevalues <- strsplit(args[7:length(args)], split = "=", fixed = T)
    extraArgs <- lapply(namevalues, function(x) x[-1])
    argnames <- lapply(namevalues, function(x) x[1])
    names(extraArgs) <- argnames

    if (!is.null(extraArgs$weight)) {
      extraArgs$weight <- as.numeric(extraArgs$weight)
    }
    if (!is.null(extraArgs$epochs)) {
      extraArgs$epochs <- as.integer(extraArgs$epochs)
    } else {
      extraArgs$epochs <- 200
    }
    if (!is.null(extraArgs$folds)) {
      extraArgs$folds <- as.integer(extraArgs$folds)
    }
  }

  if (is.null(dataset_f)) stop("Did not find dataset!")

  # print(c(requiredArgs, extraArgs))

  retval <- do.call(experiment_validation, c(requiredArgs, extraArgs))

  print(retval$classifiers)

  saveRDS(retval, args[6])
}
