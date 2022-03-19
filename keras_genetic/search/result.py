import numpy as np
from tensorflow import keras

@dataclass
class SearchResult:
    """SearchResult contains the results for a search call."""

    results: List[keras_genetic.individual]

    @property
    def best(self):
        return self.results[0]
