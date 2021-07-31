SVMLayer <- R6::R6Class("R6Layer",
  inherit = keras::KerasLayer,
  public = list(
    # output_dim = NULL,
    kernel = NULL,
    bias = NULL,
    initialize = function() {
      # self$output_dim <- output_dim
    },
    build = function(input_shape) {
      self$kernel <- self$add_weight(
        name = "svm_kernel",
        shape = list(input_shape[[2]], 1L),
        initializer = keras::initializer_lecun_normal(),
        trainable = TRUE
      )
      self$bias <- self$add_weight(
        name = "svm_bias",
        shape = list(1L, 1L),
        initializer = keras::initializer_lecun_normal(),
        trainable = TRUE
      )
    },
    call = function(x, mask = NULL) {
      keras::k_dot(x, self$kernel) + self$bias
    },
    compute_output_shape = function(input_shape) {
      list(input_shape[[1]], 1L)
    }
  )
)


layer_svm <- function(object, name = NULL, trainable = TRUE) {
  keras::create_layer(SVMLayer, object, list(
    name = name,
    trainable = trainable,
    dtype = "float32"
  ))
}
