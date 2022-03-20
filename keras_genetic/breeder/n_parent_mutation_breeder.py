import random

import numpy as np

from keras_genetic import core
from keras_genetic.breeder.breeder import Breeder


class NParentMutationBreeder(Breeder):
    """NParentMutationBreeder randomly mutates features on individuals."""

    def __init__(
        self,
        n,
        keep_probability=0.9,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if n < 1:
            raise ValueError(
                f"NParentMutationBreeder.__init__() received n={n}.  Want n > 1"
            )
        self.n = n
        self.keep_probability = keep_probability

    def offspring(self, parents):
        parents = random.sample(parents, self.n)
        offspring_weights = []

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for parent_weights in zip(parents):
            shape = parent_weights[0].shape
            parent_weights = [w.flatten() for w in parent_weights]
            # easier to traverse them as flattened arrays
            result = []
            for i in range(parent_weights[0].shape[0]):
                if random.uniform(0, 1) > self.keep_probability:
                    result.append(self.initializer())
                    continue
                choice = random.choice(parent_weights)
                result.append(choice[i])

            offspring_weights.append(np.array(result).reshape(shape))

        return core.Individual(offspring_weights, model=parents[0].model)
