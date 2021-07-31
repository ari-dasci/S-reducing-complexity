#' @include slicer.R
Combined <- R6::R6Class("Combined",
  inherit = Slicer,
  private = list(
    scorer = NULL,
    skaler = NULL,
    to_keras = function(input_shape) {
      super$to_keras(input_shape)

      private$scorer$set_autoencoder(private$autoencoder)
      private$skaler$set_autoencoder(private$autoencoder)
      private$loss <- private$loss + private$scorer$penalty() + private$skaler$penalty()
    }
  ),
  public = list(
    initialize = function(network, loss, slicer_weight = .1, scorer_weight = 0.01, skaler_weight = 0.01) {
      super$initialize(network, loss, slicer_weight)
      private$scorer <- Scorer$new(network, loss, scorer_weight)
      private$skaler <- Skaler$new(network, loss, skaler_weight)
    }
  )
)

