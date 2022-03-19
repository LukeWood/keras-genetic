import random

import numpy as np

import keras_genetic


class Breeder:
    """Breeder is responsible for creating offspring."""

    def __init__(self, initializer=None):
        if initializer is None:
            initializer = keras_genetic.initializers.random_normal.RandomNormal()
        self.initializer = initializer

    def offspring(self, mother, father):
        """`offspring()` creates a new offspring from a mother and father model.

        Args:
            mother: keras_genetic.Individual.
            father: keras_genetic.Individual.
        Returns:
            a new keras_genetic.Individual representing the offspring.
        """
        raise NotImplementedError(
            "`offspring()` must be implemented on the breeder class."
        )

    def population_from_parents(self, parents, population_size):
        result = []
        for _ in range(population_size):
            mother, father = random.sample(parents, 2)
            result.append(self.offspring(mother, father))
        return result

    def fully_random_weight_set(self, mother):
        offspring_weights = []

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for m in mother.weights:
            shape = m.shape

            # easier to traverse them as flattened arrays
            m = m.flatten()
            result = []
            for i in range(m.shape[0]):
                result.append(self.initializer())

            offspring_weights.append(np.array(result).reshape(shape))

        return keras_genetic.Individual(offspring_weights, model=mother.model)
