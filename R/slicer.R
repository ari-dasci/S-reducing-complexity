#' Slicer
#'
#' Supervised linear classifier error reduction. This autoencoder uses a Support
#' Vector Machine-based penalty in order to increase class separability.
#'
#' @section Arguments:
#'
#'   `network` Ruta network object indicating the desired architecture.
#'
#'   `loss` Character scalar (e.g. `"mean_squared_error"`) or Ruta loss object.
#'
#'   `weight` Weight of the applied penalty.
#'
#' @section Methods:
#'
#'   `$new()` Initialize new object
#'
#'   `$set_autoencoder()` Save a Keras model as internal autoencoder.
#'
#'   `$train()` Train the autoencoder with data.
#'
#'   `$encode()` Once trained, encode new data.
#'
#'   `$penalty()` Retrieve the Keras tensor which computes the penalty for the
#'   loss.
#'
#' @name Slicer
NULL

#' @include custom-autoencoder.R
#' @include svm_layer.R
#' @export
Slicer <- R6::R6Class("Slicer",
  inherit = CustomAutoencoder,
  private = list(
    to_keras = function(input_shape) {
      learner <- ruta::autoencoder(private$network, private$reconstruction_loss)
      learner$input_shape <- input_shape
      models <- ruta:::to_keras(learner)

      encoding <- keras::get_layer(models$autoencoder, "encoding") %>% keras::get_output_at(1)

      svmtrainer <- layer_svm(encoding, "svm_layer")
      # svmtrainer <- keras::k_sum(encoding, name = "svm_layer")

      # class input accepts ones and zeros
      class_pos <- keras::layer_input(list(1))
      models$autoencoder <- keras::keras_model(
        list(models$autoencoder$input, class_pos),
        list(models$autoencoder$output, svmtrainer)
      )

      private$autoencoder <- models$autoencoder
      private$encoder <- models$encoder
      private$decoder <- models$decoder

      start <- private$autoencoder$inputs[[1]]
      end <- private$autoencoder$outputs[[1]]
      rec_loss <- (ruta:::to_keras(learner$loss, learner))(start, end)
      private$loss <- keras::k_mean(rec_loss) + self$penalty()
    }
  ),
  public = list(
    penalty = function() {
      class_pos <- private$autoencoder$inputs[[2]]
      svm_ly <- keras::get_layer(private$autoencoder, "svm_layer")
      svm_weight <- svm_ly$weights[[1]]
      svm_out <- keras::get_output_at(svm_ly, 1)
      t_n <- 2 * class_pos - 1 # class \in {-1, 1}

      sum_term <- keras::k_square(1 - svm_out * t_n)
      svm_loss <- keras::k_sum(sum_term)

      weight_reg <- keras::k_sum(svm_weight * svm_weight) / 2

      private$weight * (weight_reg + svm_loss)
    }
  )
)
