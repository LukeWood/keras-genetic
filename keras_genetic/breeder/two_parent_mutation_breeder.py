import random

import numpy as np

from keras_genetic import core
from keras_genetic.breeder.breeder import Breeder


class TwoParentMutationBreeder(Breeder):
    """TwoParentMutationBreeder randomly mutates features on individuals."""

    def __init__(
        self,
        keep_probability=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.keep_probability = keep_probability

    def offspring(self, parents):
        mother, father = random.sample(parents, 2)
        offspring_weights = []

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for m, f in zip(mother.weights, father.weights):
            shape = m.shape

            # easier to traverse them as flattened arrays
            m = m.flatten()
            f = f.flatten()

            result = []
            for i in range(m.shape[0]):
                if random.uniform(0, 1) > self.keep_probability:
                    result.append(self.initializer())
                    continue
                result.append(random.choice([m[i], f[i]]))

            offspring_weights.append(np.array(result).reshape(shape))

        return core.Individual(offspring_weights, model=mother.model)
