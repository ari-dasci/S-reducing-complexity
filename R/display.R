format_results <- function(results_dir=NULL, results=NULL) {
  if (is.null(results))
    results <- readRDS(file.path(results_dir, "results.rds"))
  n_datasets <- length(results)
  index_datasets <- names(results)

  if (n_datasets == 0) stop("No datasets found.")

  n_methods <- length(results[[1]])
  index_methods <- names(results[[1]])

  if (n_methods == 0) stop("No methods found.")

  n_metric_types <- 2
  index_metric_types <- c("features", "classifiers")

  n_folds                  <- length(results[[1]][[1]]$features)
  n_feature_metrics        <- length(results[[1]][[1]]$features[[1]])
  index_feature_metrics    <- names(results[[1]][[1]]$features[[1]])
  n_classifier_metrics     <- length(CLASSIFIERS) * length(CL_METRICS)
  index_classifiers        <- CLASSIFIERS
  index_classifier_metrics <- paste(rep(index_classifiers, each=length(CL_METRICS)), CL_METRICS, sep=".")
  n_total_metrics          <- n_feature_metrics + n_classifier_metrics
  index_total_metrics      <- c(index_feature_metrics, index_classifier_metrics)

  metric_arr <- array(dim=c(n_total_metrics, n_datasets, n_methods))
  dimnames(metric_arr) <- list(index_total_metrics, index_datasets, index_methods)

  for (i_dataset in index_datasets) {
    for (i_method in index_methods) {
      for (i_metric in index_feature_metrics) {
        values <- rep(NA, n_folds)
        for (i_fold in 1:n_folds) {
          values[i_fold] <- results[[i_dataset]][[i_method]]$features[[i_fold]][i_metric]
        }
        metric_arr[i_metric, i_dataset, i_method] <- mean(unlist(values), na.rm = TRUE)
      }
      # cat(i_dataset, "/\n")
      for (i_classifier in index_classifiers) {
        # cat(i_classifier, ": ")
        for (i_metric in CL_METRICS) {
          values <- rep(NA, n_folds)
          for (i_fold in 1:n_folds) {
            val <- results[[i_dataset]][[i_method]]$classifiers[[i_classifier]][[i_fold]][i_metric]
            if (!is.null(val)) values[i_fold] <- val
          }
          i_cl_metric <- paste(i_classifier, i_metric, sep=".")
          # cat(i_cl_metric, "= ")
          metric_arr[i_cl_metric, i_dataset, i_method] <- mean(unlist(values), na.rm = TRUE)
          # cat(metric_arr[i_cl_metric, i_dataset, i_method], "| ")
        }
        # cat("\n")
      }
    }
  }

  metric_arr
}


printable <- function(x, ...) UseMethod("printable")

printable.list_metrics <- function(x, ...) {
  structure(as.numeric(unlist(x)), names = names(x))
}

print.list_metrics <- function(x, ...) {
  print(printable(x), ...)
  invisible(x)
}

printable.results_dataset <- function(x, ...) {
  average <- function(obj) {
    map(obj, printable) %>% do.call(rbind, .) %>% colMeans(na.rm = TRUE)
  }
  printable_features <- average(x$features)
  printable_cls <- map(x$classifiers, average)

  do.call(c, c(list(printable_features), printable_cls))
}

print.results_dataset <- function(x, ...) {
  cat("Measures averaged over", length(x[[1]]), "folds:\n")
  print(printable(x))
  invisible(x)
}

printable.results_experiment <- function(x, ...) {
  x %>% map(~ map(., printable)) %>% map(~ do.call(rbind, .))
}

print.results_experiment <- function(x, ...) {
  reslist <- printable(x)
  walk2(reslist, names(x), function(obj, name) {
    cat("================\n", name, "\n================\n", sep = "")

    print(obj)
  })

  invisible(x)
}

printable.default <- function(x, ...) x

# x: results_experiment: { dataset = { method = { "features" = [ { metric = numeric } ], "classifiers" = [ { metric = numeric } ] } } }
# return: { dataset = { method = [ { metric = numeric } ] } }
select_classifier <- function(x, cl) {
  map(x, function(dataset)
    map(dataset, function(method) {
      #print(method$classifiers[[cl]])
      listzip(method$features, method$classifiers[[cl]])
    }))
}

get_tables <- function(results) {
  results %>% map(function(ds) {
    metrics <- names(ds[[1]][[1]])
    #print(metrics)
    table <- metrics %>% map(function(metric) {
      #ds %>% map(metric) %>% as.numeric %>% mean
      ds %>%
        map(function(k) k %>% map(metric) %>% as.numeric() %>% mean(na.rm = TRUE)) %>%
        do.call(cbind, .)
    }) %>% do.call(rbind, .) %>% t()
    colnames(table) <- metrics
    table
  })
}

get_table_bests <- function(tables) {
  tables %>%
    map(function(tab) {
      row.names(tab)[apply(tab, 2, function(col) match(max(col), col))]
    }) %>%
    do.call(rbind, .)
}

print_tables <- function(tables, filename = NULL) {
  if (!is.null(file)) sink(filename)
  for (table in names(tables)) {
    c("\n##", table) %>% paste0(collapse = " ") %>% cat()
    tables[[table]] %>% knitr::kable(., format = "markdown") %>% print()
  }
  if (!is.null(file)) sink()
}

myrank <- function(x, decreasing = FALSE, ...) {
  if (decreasing) x <- -x
  rank(x, ...)
}

# use decreasing=9:17 to invert the ranking in classifier metrics
get_rankings <- function(tables, decreasing = 8:16) {
  rables <- tables
  rables[decreasing, , ] <- -rables[decreasing, , ]

  # return same structure than tables
  aperm(apply(rables, c(1, 2), rank), c(2, 3, 1))
}

display_names <- list(
  pca="PCA", lle="LLE", isomap="Isomap", autoencoder="AE", skaler="Skaler", scorer="Scorer", slicer="Slicer",
  fisher="F1", efficiency="F3", errorlin="L2", nonlin="L3", errorknn="N3", onb_total="ONBtot", onb_avg="ONBavg",
  knn.auc="kNN AUC", knn.fscore="kNN F-score", knn.kappa="kNN Kappa",
  svmRadial.auc="SVM AUC", svmRadial.fscore="SVM F-score", svmRadial.kappa="SVM Kappa",
  mlp.auc="MLP AUC", mlp.fscore="MLP F-score", mlp.kappa="MLP Kappa"
)
get_plots <- function(tables, which=c("fisher", "efficiency", "knn.fscore", "knn.kappa")) {
  # pars <- list(nrow = floor(length(which)/2))
  # if (length(which) == 4) {
  #   pars[["layout_matrix"]] <- matrix(c(1,1,1,2,2,2,2,2,3,3,3,4,4,4,4,4),nrow=2,byrow=T)
  # }
  #
  # plots <- map2(which, rep(c(FALSE, TRUE), length.out=length(which)),
  #   ~ scmamp::plotDensities(tables[.x, , ], na.rm = TRUE, show.legend = .y) + ggplot2::xlab(.x))
  # do.call(gridExtra::grid.arrange, c(plots, pars))

  tables_wide <- lapply(dimnames(tables)[[1]], function(metric) {
    # metric_results <- tables[metric, , ]
    blocks <- lapply(dimnames(tables)[[3]], function(method) {
      data.frame(
        score = tables[metric, , method],
        method = display_names[[method]]
      )
    })

    do.call(rbind, blocks)
  })
  names(tables_wide) <- dimnames(tables)[[1]]

  map(which, ~
        ggplot2::ggplot(tables_wide[[.]], ggplot2::aes(x=score, y=method)) +
        ggridges::geom_density_ridges(rel_min_height = 0.005, quantile_lines = TRUE, quantiles = 2, scale=3) +
        ggplot2::scale_y_discrete(expand = c(0.01, 0), name="") +
        ggplot2::scale_x_continuous(expand = c(0.01, 0), breaks=c(0,.25,.5,.75,1),limits=c(-.25,1.25), name=display_names[[.]]) +
        ggridges::theme_ridges() + ggplot2::theme(text = element_text(size = 20)))

}

higher_is_better <- list(
  fisher           = FALSE,
  volume           = FALSE,
  efficiency       = FALSE,
  errorknn         = FALSE,
  errorlin         = FALSE,
  nonlin           = FALSE,
  ir               = FALSE,
  onb_total        = FALSE,
  onb_avg          = FALSE,
  ppd              = TRUE,
  knn.fscore       = TRUE,
  knn.kappa        = TRUE,
  knn.auc          = TRUE,
  svmRadial.fscore = TRUE,
  svmRadial.kappa  = TRUE,
  svmRadial.auc    = TRUE,
  mlp.fscore       = TRUE,
  mlp.kappa        = TRUE,
  mlp.auc          = TRUE
)

get_wins <- function(tables) {
  # get which metrics are better when lower
  inverse_metrics <- names(higher_is_better)[!unlist(higher_is_better)]
  inverse_metrics <- intersect(inverse_metrics, dimnames(tables)[[1]])
  # set those values in negative so that the maximum is the best
  tables[inverse_metrics,,] <- -tables[inverse_metrics,,]
  # find the maximum for each case
  best_metrics <- apply(tables, 2, function(t){ apply(t, 1, max) })
  # get boolean tables for
  which_won <- lapply(dimnames(tables)[[3]], function(m) tables[,,m] == best_metrics)
  structure(sapply(which_won, sum), names=dimnames(tables)[[3]])
}

#' @import scmamp
#' @import reshape2
get_tests <- function(tables, quiet = T, alpha = 0.01) {
  # control_i <- which(dimnames(tables)[[3]] == control)
  different_table <- array(FALSE, dim = dim(tables[, 1, ]))
  dimnames(different_table) <- list(dimnames(tables)[[1]], dimnames(tables)[[3]])
  # molten_tables <- reshape2::melt(tables, varnames=c("metric", "dataset", "method"))
  # resultsdf <- reshape2::dcast(molten_tables, metric + dataset ~ method)
  # resultsdf$dataset <- NULL
  # resultsdf$metric <- NULL

  for (metric in dimnames(tables)[[1]]) {
    if (!quiet) cat("\nNow evaluating", metric, "\n")
    metric_results <- tables[metric, , ]

    comptest <- scmamp::friedmanAlignedRanksTest(metric_results)
    if (comptest$p.value < alpha) {
      if (!quiet) cat("With confidence > 0.99, not all algorithms perform the same.\n")
      test <- scmamp::postHocTest(metric_results, test="aligned ranks", use.rank=TRUE)

      opt <- (if (higher_is_better[[metric]]) min else max)(test$summary)
      # cat(test$summary, "\n best:", which(test$summary == opt), "(", opt,")\n")
      # print(test$corrected.pval)
      different_table[metric,] <- test$corrected.pval[which(test$summary == opt), ] < alpha

      update_names <- function(nlist) sapply(nlist, function(n) display_names[[n]])
      dimnames(test$summary) <- lapply(dimnames(test$summary), update_names)
      dimnames(test$corrected.pval) <- lapply(dimnames(test$corrected.pval), update_names)
      pdf(paste0("alignedranks_", sub(".", "", metric, fixed = T), ".pdf"), width = 6, height = 2)
      scmamp::plotRanking(test$corrected.pval, summary=test$summary, alpha=alpha, cex=1)
      dev.off()
    } else {
      if (!quiet) cat("All methods performed the same.")
    }
  }

  different_table
}

get_global_test <- function(tables) {
  # get which metrics are better when lower
  inverse_metrics <- names(higher_is_better)[!unlist(higher_is_better)]
  inverse_metrics <- intersect(inverse_metrics, dimnames(tables)[[1]])
  # set those values in negative so that the maximum is the best
  tables[inverse_metrics,,] <- -tables[inverse_metrics,,]
  molten_tables <- reshape2::melt(tables, varnames=c("metric", "dataset", "method"))
  resultsdf <- reshape2::dcast(molten_tables, metric + dataset ~ method)
  resultsdf$dataset <- NULL
  resultsdf$metric <- NULL
  test <- scmamp::postHocTest(resultsdf, test="aligned ranks", use.rank=TRUE)
  update_names <- function(nlist) sapply(nlist, function(n) display_names[[n]])
  dimnames(test$summary) <- lapply(dimnames(test$summary), update_names)
  dimnames(test$corrected.pval) <- lapply(dimnames(test$corrected.pval), update_names)
  pdf(paste0("alignedranks_global.pdf"), width = 6, height = 2)
  scmamp::plotRanking(test$corrected.pval, summary = test$summary, alpha=0.01)
  dev.off()
  test
}

get_bayesian_tests <- function(tables, quiet = T) {
  test_results <- list()
  for (metric in dimnames(tables)[[1]]) {
    if (!quiet) cat("\nNow evaluating", metric, "\n")
    metric_results <- tables[metric, , ]
    res <- scmamp::bPlackettLuceModel(metric_results, min=!higher_is_better[[metric]])
    print(metric)
    print(res$expected.win.prob)
    print(res$expected.mode.rank)
    test_results[[metric]] <- res
  }
  test_results
}

get_analysis <-  function(tables, quiet = T, control = "slicer", exclude = character(0)) {

  metric_axis  <- 1
  dataset_axis <- 2
  method_axis  <- 3
  tab <- tables

  include <- setdiff(dimnames(tab)[[method_axis]], exclude)
  print(include)
  tab <- tab[,,include]

  metrics <- dimnames(tab)[[metric_axis]]
  #--------------------------------------------------------
  plots <- get_plots(tab)
  #--------------------------------------------------------


  ranking_table <- get_rankings(tab, unlist(higher_is_better[dimnames(tab)[[1]]]))
  #--------------------------------------------------------
  different_table <- get_tests(tab, quiet, control)

  list(
    plots     = plots,
    ranking   = ranking_table,
    different = different_table
  )
}

print_all_tests <- function() {
  files <- c(dir("config_200ep_gain", full.names = T), dir("config_200ep_loss", full.names = T))
  for (f in files) {
    cat("\n\n#", f, "\n")
    t <- get_tests(f)
    cat(t %>% knitr::kable())
  }
}
