class Individual:
    """Individual represents a single individual.

    Args:
        weights: array of np.array weights.
        model: keras model to load the weights into.
        score: score the individual received.

    Attributes:
        weights: array of np.arrays representing weights of the neural network.
        score: score the individual received during training.
    """

    def __init__(self, weights, model, score=None):
        self.weights = weights
        self.model = model
        self.score = score

    def load_model(self):
        """load_model() prepares the keras model for use.

        Usage:
        ```python
        model = individual.load_model()
        y_test = model.predict(x_test)
        ```
        """
        self.model.set_weights(self.weights)
        return self.model

    def __gt__(self, other):
        return self.score > other.score

    def __lt__(self, other):
        return self.score < other.score
