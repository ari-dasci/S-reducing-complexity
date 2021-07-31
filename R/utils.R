expand_dims <- function(features) {
  if (is.null(dim(features)) || length(dim(features)) == 1)
    array(features, c(length(features), 1))
  else
    features
}

name <- function(features) {
  colnames(features) <- paste0("h", 1:dim(features)[2])
  features
}

listzip <- function(...) {
  purrr::pmap(list(...), c)
}

#' @import purrr
#' @export
preparation <- function(dataset, class_name = "class", value_positive = 1) {
  filtered <- if (is_character(class_name)) match(class_name, names(dataset)) else class_name
  x <- dataset[, -filtered]
  cat_inds <- apply(x, 2, is.factor)
  x[, cat_inds] <- as.numeric(x[, cat_inds])
  x <- data.matrix(x)
  list(
    x = x,
    y = (dataset[[class_name]] %in% value_positive) %>% as.integer() %>% as.factor(),
    normalize = any(x > 1) || any(x < 0)
  )
}

#' @export
class_first <- purrr::partial(preparation, class_name = 1)

#' @export
class_last <- function(dataset, value_positive = 1)
  preparation(dataset, class_name = ncol(dataset), value_positive = value_positive)

#' @export
vs <- function(dataset, class_pos, positive, negative) {
  dataset[dataset[,class_pos] %in% c(positive, negative), ] %>%
    preparation(class_pos, positive)
}

#' @export
vs_first <- purrr::partial(vs, class_pos = 1)

#' @export
vs_last <- function(dataset, positive, negative)
  vs(dataset, class_pos = ncol(dataset), positive, negative)
