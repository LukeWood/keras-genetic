import numpy as np
from tensorflow import keras


@dataclass
class SingleResult:
    weights: [numpy.array]
    score: float


@dataclass
class SearchResult:
    """SearchResult contains the results for a search call."""

    results: List[SingleResult]
    model: keras.Model

    @property
    def best_weights(self):
        return results[0].weights

    @property
    def best_model(self):
        model.set_weights(self.best_weights)
        return model
