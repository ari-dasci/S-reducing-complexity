#' CustomAutoencoder
#'
#' A common abstract class for custom autoencoder implementations.
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
#' @name CustomAutoencoder
NULL

#' @export
CustomAutoencoder <- R6::R6Class("CustomAutoencoder",
  private = list(
    network = NULL,
    reconstruction_loss = NULL,
    weight = NULL,
    autoencoder = NULL,
    encoder = NULL,
    decoder = NULL,
    loss = NULL
  ),
  public = list(
    initialize = function(network, loss, weight) {
      private$network <- network
      private$reconstruction_loss <- loss
      private$weight <- weight
    },

    set_autoencoder = function(autoencoder) {
      private$autoencoder <- autoencoder
    },

    train = function(data, classes, epochs = 100, optimizer = keras::optimizer_adam(), ...) {
      input_shape <- dim(data)[-1]
      private$to_keras(input_shape)

      private$autoencoder$add_loss(private$loss)
      private$autoencoder$compile(optimizer = optimizer)

      keras::fit(
        private$autoencoder,
        x = list(data, classes),
        validation_split=0.15,
        epochs = epochs,
        ...
      )

      invisible(self)
    },

    encode = function(data) {
      private$encoder$predict(data)
    }
  )
)
