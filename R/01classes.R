# list_metrics: { numeric }
list_metrics <- "list_metrics"

# results_dataset: { "features"    = [ list_metrics ],
#                    "classifiers" = { [ list_metrics ] } }
results_dataset <- "results_dataset"

# results_experiment: { dataset = { method = results_dataset } }
results_experiment <- "results_experiment"

CLASSIFIERS <- c("knn", "svmRadial", "mlp")
CL_METRICS  <- c("fscore", "auc", "kappa")
