import random

import numpy as np

from keras_genetic import core
from keras_genetic.breeder.breeder import Breeder


class MutationBreeder(Breeder):
    """MutationBreeder randomly mutates features on individuals."""

    def __init__(
        self,
        keep_probability=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_probability = keep_probability

    def offspring(self, parents):
        mother = random.choice(parents)
        offspring_weights = []

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for w in mother.weights:
            shape = w.shape
            w = w.flatten()

            result = []
            for i in range(w.shape[0]):
                if random.uniform(0, 1) > self.keep_probability:
                    result.append(self.initializer())
                    continue
                result.append(w[i])

            offspring_weights.append(np.array(result).reshape(shape))

        return core.Individual(offspring_weights, model=mother.model)
