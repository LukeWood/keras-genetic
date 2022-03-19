class Individual:
    """Individual represents a single individual.

    Args:
        weights: array of np.array weights.
        score: score the individual received.
        model: keras model to load the weights into.

    Attributes:
        weights: array of np.arrays representing weights of the neural network.
        score: score the individual received during training.
    """

    def __init__(self, weights, score=None, model=None):
        self.weights = weights
        self.score = score
        self.model = model

    def load_model(self):
        """load_model() prepares the keras model for use.

        Usage:
        ```python
        model = individual.load_model()
        y_test = model.predict(x_test)
        ```
        """
        self.model.load_weights(self.weights)
        return self.model
