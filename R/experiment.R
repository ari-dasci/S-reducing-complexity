#' @importFrom ruta input dense output train autoencoder encode
#' @importFrom purrr partial compose %>%
#' @importFrom dimRed LLE Isomap embed
#' @importFrom caret knn3
train_reduction <- function(train_x, train_y, method, normalized = TRUE, ...) {
  args <- list(...)
  max_for_10ppd <- ceiling(0.1 * nrow(train_x))
  ten_percent <- ceiling(0.1 * ncol(train_x))
  squareroot <- ceiling(sqrt(ncol(train_x)))
  # At least 2 generated features
  # hidden_dim <- max(min(max_for_10ppd, ten_percent), 2)
  hidden_dim <- max(min(max_for_10ppd, squareroot), 2)
  # hidden_dim <- max(sqrt(ncol(train_x)), 2)

  activation <- if (normalized) "sigmoid" else "linear"

  middle_layer <- 2
  network <- if (squareroot <= max_for_10ppd && squareroot <= 10) {
    input() + dense(hidden_dim, "relu") + output(activation)
  } else {
    middle_layer <- 3
    middle <- floor(sqrt(ncol(train_x) * hidden_dim))
    middle_act <- if (method %in% c("skaler", "skaler2")) "sigmoid" else "relu"
    input() + dense(middle, "relu") + dense(hidden_dim, middle_act) + dense(middle, "relu") + output(activation)
  }



  # Do not use binary crossentropy (and sigmoid activation) *unless* the data has been
  # accordingly normalized (to the [0, 1] interval)
  loss <- if (normalized) "binary_crossentropy" else "mean_squared_error"
  epochs <- if (is.null(args$epochs)) 200 else args$epochs

  # reduction_f <- function(x) x
  reduction_f <-
    switch(method,
           baseline = {
             function(x)
               x
           },
           pca = {
             pca <- train_x %>% prcomp(scale = FALSE, center = FALSE)
             function(x) predict(pca, x)[, 1:hidden_dim]
           },
           isomap = {
             embedding <- embed(train_x, "Isomap", .mute = c("message", "output"), ndim = hidden_dim)
             function(x) {
               if (nrow(x) == nrow(train_x) && all(x == train_x)) embedding@data@data
               else embedding@apply(x)@data
             }
           },
           lle = {
             function(x) {
               # learns embedding with train and test dataset, apparently
               # the only way to do it with LLE, but valid because the class
               # info is not used
               embedding <- embed(rbind(x, train_x), "LLE", .mute = c("message", "output"), ndim = hidden_dim)
               embedding@data@data[1:nrow(x), ]
             }
           },
           autoencoder = {
             feature_extractor <- autoencoder(network, loss = loss) %>%
               train(train_x, epochs = epochs)
             partial(encode, learner = feature_extractor, .lazy = FALSE)
           },
           scorer = {
             weight <- if (is.null(args$weight)) 0.01 else args$weight
             feature_extractor <- Scorer$new(network, loss = loss, weight = weight)
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           slicer = {
             weight <- if (is.null(args$weight)) 0.1 else args$weight
             feature_extractor <- Slicer$new(network, loss = loss, weight = weight)
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           skaler = {
             weight <- if (is.null(args$weight)) .1 else args$weight
             feature_extractor <- Skaler$new(network, loss = loss, weight = weight, penalty_type="kld")
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           skaler3 = {
             weight <- if (is.null(args$weight)) .1 else args$weight
             feature_extractor <- Skaler$new(network, loss = loss, weight = weight, penalty_type="mutualinf")
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           skaler4 = {
             weight <- if (is.null(args$weight)) .1 else args$weight
             feature_extractor <- Skaler$new(network, loss = loss, weight = weight, penalty_type="kldsparsity")
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           skaler2 = {
             weight <- if (is.null(args$weight)) .01 else args$weight
             feature_extractor <- Skaler2$new(network, loss = loss, weight = weight)
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           },
           combined = {
             feature_extractor <- Combined$new(network, loss = loss, slicer_weight = 1, scorer_weight = 0.01)
             feature_extractor$train(train_x, classes = as.numeric(train_y) - 1, epochs = epochs)
             feature_extractor$encode
           })

  compose(name, expand_dims, reduction_f)
}

#' @import caret
#' @importFrom purrr partial
train_classifier <- function(train_x, train_y, classifier) {
  ctrl <- trainControl(method = "none")
  pars <- list(
    train_x,
    train_y,
    method = classifier,
    trControl = ctrl,
    num.threads = 1
  )
  if (classifier == "knn") {
    pars["use.all"] <- FALSE
  }

  classifier <- tryCatch(
    do.call(caret::train, pars),
    error = function(e) NULL
  )
  # if predicting fails, predict the same class for all rows
  function(dat) tryCatch(
    predict(classifier, dat),
    error = function(e) rep(train_y[1], nrow(dat))
  )
}

#' @importFrom caret createFolds
#' @export
experiment_validation <- function(
  dataset_f, method, name, classifiers = c("knn", "svmRadial", "mlp"),
  seed = 4242, folds = 5, verbose = TRUE, autosave = TRUE, loadsave = TRUE, omit = FALSE,
  # The following parameters take advantage of R's lazy parameter evaluation in order
  # to skip dataset loading unless it is needed
  dataset = dataset_f(), folds_idx = set_folds(), ...) {
  log <- function(...) {
    if (verbose) cat(...)
  }

  log("Method:", method, "|| ")

  # Lazy computation of dataset folds
  set_folds <- function() {
    set.seed(seed)
    createFolds(dataset$y, k = folds)
  }

  results <- list()
  results$features <- list()
  cls <- list()

  for (i in 1:folds) {
    dirname <- file.path("checkpoints", method)
    filename <- file.path(dirname, paste0(name, "_fold_", i, ".rds"))
    evalname <- file.path(dirname, paste0(name, "_eval_", i, ".rds"))

    # all work is done
    if (loadsave && dir.exists(dirname) && file.exists(filename) && file.exists(evalname)) {
      current <- readRDS(evalname)
      results$features[[i]] <- current$features
      cls[[i]] <- current$classifiers
    } else if (omit) {
      results$features[[i]] <- NA
      cls[[i]] <- structure(map(classifiers, ~NA), names = classifiers)
    } else { # need to train and/or evaluate, so load datasets
      train_x = dataset$x[-folds_idx[[i]],]
      train_y = dataset$y[-folds_idx[[i]]]
      test_x = dataset$x[folds_idx[[i]],]
      test_y = dataset$y[folds_idx[[i]]]

      if (loadsave && dir.exists(dirname) && file.exists(filename)) {
        reduced <- readRDS(filename)
      } else {
        log("Fold", i, paste0("(", sum(test_y == 1), "/", length(test_y), " positives)"), ">> ")

        if (dataset$normalize) {
          log("normalizing")
          mx <- apply(train_x, 2, max)
          mn <- apply(train_x, 2, min)
          # Avoid division by zero
          range_n <- max(mx - mn, keras::k_epsilon())

          train_x <- t(apply(train_x, 1, function(x) (x - mn) / range_n))
          test_x <- t(apply(test_x, 1, function(x) (x - mn) / range_n))
          log(" >> ")
        }

        reduction <- purrr::quietly(train_reduction)(train_x, train_y, method, dataset$normalize, ...)$result

        reduced <- list(
          train = purrr::quietly(reduction)(train_x)$result,
          test = purrr::quietly(reduction)(test_x)$result
        )

        if (autosave) {
          dir.create(dirname, showWarnings = FALSE, recursive = TRUE)
          saveRDS(reduced, file = filename)
        }
      }

      log("evaluating: ")
      results$features[[i]] <- evaluate_features(reduced$test, test_y)
      cls[[i]] <- map(classifiers, function(cl) {
        log(cl, "(training)")
        model <- purrr::quietly(train_classifier)(reduced$train, train_y, cl)$result
        log(" (predicting) ")
        predictions <- model(reduced$test)
        purrr::quietly(evaluate_model)(test_y, predictions)$result
      })
      names(cls[[i]]) <- classifiers

      if (autosave) {
        saveRDS(list(features = results$features[[i]], classifiers = cls[[i]]), file = evalname)
      }
      log("\n")
    }

    gc()
  }

  results$classifiers <- map(classifiers, function(cl) {
    cls %>% map(cl)
  })
  names(results$classifiers) <- classifiers

  log("OK\n")

  structure(results, class = results_dataset)
}

weight_optimization <- function(datasets = dataset_list(except = c("IMDB", "Internet", "Riccardo")),
                                method = "slicer",
                                weights = 10 ** seq(-4, 2, by = 1),
                                metric = "knn.fscore",
                                epochs = 20) {

  results <- array(dim = c(length(datasets), length(weights)))
  dimnames(results) <- list(names(datasets), as.character(weights))
  is_classifier_metric <- grepl(".", metric, fixed=T)

  classifier <- if (is_classifier_metric) strsplit(metric, ".", fixed=T)[[1]][1] else character(0)
  metric <- if (is_classifier_metric) strsplit(metric, ".", fixed=T)[[1]][2] else metric
  verbose <- F

  for (d in names(datasets)) {
    for (w in weights) {
      save_partial <- file.path(tempdir(), "partial.rds")
      system2("/usr/bin/env", c("Rscript", "R/validation_proc.R",
                                d, method, as.character(verbose), as.character(FALSE), as.character(FALSE), save_partial, paste0("weight=", w), paste0("epochs=", epochs), "folds=2"))
      partial <- readRDS(save_partial)
      results[d, as.character(w)] <-
        if (is_classifier_metric) {
          partial$classifiers[[classifier]] %>% map(~.[metric]) %>% unlist() %>% mean(na.rm=T)
        } else {
          partial$features %>% map( ~ .[metric]) %>% unlist() %>% mean(na.rm = T)
        }
      print(paste(d, w, " | ", results[d, as.character(w)]))
    }
  }

  saveRDS(results, paste0(method, "_weight_opt.rds"))
  results
}


#' @export
experiment_all <-
  function(datasets = dataset_list(),
           folder = paste0("results_", format(Sys.time(), "%y-%m-%d_%H:%M")),
           methods = c("baseline",
                       "pca",
                       "lle",
                       "isomap",
                       "autoencoder",
                       "scorer",
                       "slicer",
                       "skaler",
                       "skaler3",
                       "combined"),
           verbose = T,
           autosave = T,
           loadsave = T,
           omit = F,
           ...) {
  log <- function(...) {
    if (verbose) cat(...)
  }

  options(keras.fit_verbose = 0)
  # library(tensorflow)
  # gpu <- tensorflow::tf$config$experimental$get_visible_devices('GPU')[[1]]
  # tensorflow::tf$config$experimental$set_memory_growth(device = gpu, enable = TRUE)

  dir.create(folder, recursive = TRUE)

  results <- map2(datasets, names(datasets), function(dataset_f, name) {
    save_name <- file.path(folder, paste0(name, ".rds"))
    if (loadsave && file.exists(save_name)) {
      log("Skipping", name, "\n")
      readRDS(save_name)
    } else {
      # keras::backend()$clear_session()
      # config = tensorflow::tf_config()
      # keras::backend()$set_session()
      gc()

      log("Entering", name, ">> ")
      dataset_f2 <- function() {
        ds <- dataset_f()
        log("read dataset >> ")
        ds
      }

      this_dataset <- map(methods, function(m) {
        save_partial <- file.path(folder, paste0(name, "_", m, ".rds"))
        system2("/usr/bin/env", c("Rscript", "R/validation_proc.R",
                name, m, as.character(verbose), as.character(autosave), as.character(loadsave), save_partial, paste("omit", as.character(omit), sep = "=")))
        tryCatch(
          readRDS(save_partial),
          error = function(e) NULL
        )
      })
        # experiment_validation(
        #   dataset_f = dataset_f2,
        #   method = m,
        #   verbose = verbose,
        #   name = name,
        #   autosave = autosave,
        #   loadsave = loadsave,
        #   ...
        # ))
      if (is.null(this_dataset)) {
        log(name, "will be missing")
      } else {
        names(this_dataset) <- methods

        log("All tests ok. Saving...\n")
        if (autosave) {
          saveRDS(this_dataset, file = save_name)
        }
      }
      this_dataset
    }
  })

  results <- structure(results, class = results_experiment)

  saveRDS(results, file = file.path(folder, "results.rds"))
  invisible(results)
}


# experiment_all(dataset_list(except=c("Arcene","IMDB")), methods=c("pca","isomap","autoencoder","scorer","skaler2","skaler","slicer"))
