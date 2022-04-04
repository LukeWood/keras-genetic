import random

import numpy as np

from keras_genetic import core
from keras_genetic import utils
from keras_genetic.breeder.breeder import Breeder


class MutationBreeder(Breeder):
    """MutationBreeder randomly multiplies weights by mutation factors."""

    def __init__(
        self,
        model,
        parents_per_generation,
        keep_probability=0.9,
        learning_rate=0.1,
        keep_parents=True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.keep_probability = keep_probability
        self.parents_per_generation = parents_per_generation
        self.keep_parents = keep_parents
        self.learning_rate = learning_rate
        self._parents = None

    def update_state(self, generation):
        self._parents = generation[: self.parents_per_generation]

    def offspring(self):
        parents = self._parents
        if not parents:
            raise RuntimeError(
                "`MutationBreeder.offspring()` called before "
                "`update_state()`.  Please call `update_state()` at least once before "
                "calling `offspring()`."
            )
        mother = random.choice(parents)

        weights = utils.flatten(mother.weights)
        mutation = 1.0 + (self.initializer((self.num_params,)) * self.learning_rate)
        weights = np.multiply(weights, mutation)
        weights = utils.conform_weights_to_shape(weights, mother.weights)

        return core.Individual(weights, model=mother.model)

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        if self.keep_parents:
            result = result + self._parents
        return result
