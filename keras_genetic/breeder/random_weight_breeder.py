import random

import numpy as np

from keras_genetic import core
from keras_genetic import utils
from keras_genetic.breeder.breeder import Breeder


class RandomWeightBreeder(Breeder):
    """RandomWeightBreeder randomly mutates features on individuals.

    The mutations are initialized to completely random values drawn from the
    `initializer` passed in the constructor.  This breeder can be used to ensure that
    the algorithm always has a chance to find the global optimum, but converges
    incredibly slow.  It should only be used within the context of a combination
    breeder.
    """

    def __init__(
        self,
        model,
        parents_per_generation,
        mutation_rate=0.1,
        keep_parents=True,
        **kwargs,
    ):
        super().__init__(model, **kwargs)
        self.mutation_rate = mutation_rate
        self.parents_per_generation = parents_per_generation
        self.keep_parents = keep_parents
        self.parents = None

    def update_state(self, generation):
        self._parents = generation[: self.parents_per_generation]

    def offspring(self):
        parents = self._parents
        if not parents:
            raise RuntimeError(
                "`RandomWeightBreeder.offspring()` called before "
                "`update_state()`.  Please call `update_state()` at least once before "
                "calling `offspring()`."
            )
        mother = random.choice(parents)
        mother_weights = utils.flatten(mother.weights)
        choices = (
            np.random.uniform(0, 1, size=mother_weights.shape) < self.mutation_rate
        )
        mutations = self.initializer(mother_weights.shape)
        result = np.where(choices, mutations, mother_weights)
        return core.Individual(
            utils.conform_weights_to_shape(result, mother.weights), model=self.model
        )

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        if self.keep_parents:
            result = result + self._parents
        return result
