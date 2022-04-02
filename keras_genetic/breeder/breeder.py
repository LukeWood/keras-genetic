import numpy as np

import keras_genetic


class Breeder:
    """Breeder is responsible for creating offspring."""

    def __init__(self, initializer=None):
        if initializer is None:
            initializer = keras_genetic.initializers.random_normal.RandomNormal()
        self.initializer = initializer
        self._parents = None
        self._template_model = None

    def offspring(self):
        """`offspring()` creates a new offspring from a mother and father model.

        Args:
            parents: list of keras_genetic.Individual.
        Returns:
            a new keras_genetic.Individual representing the offspring.
        """
        raise NotImplementedError(
            "`offspring()` must be implemented on the breeder class."
        )

    def update_state(self, generation):
        raise NotImplementedError(
            "`update_state()` must be implemented on the breeder class."
        )

    def population(self, population_size):
        raise NotImplementedError(
            "`population()` must be implemented on the breeder class."
        )

    def fully_random_weight_set(self, mother):
        offspring_weights = []

        for m in mother.weights:
            shape = m.shape

            # easier to traverse them as flattened arrays
            m = m.flatten()
            result = []
            for i in range(m.shape[0]):
                result.append(self.initializer())

            offspring_weights.append(np.array(result).reshape(shape))

        return keras_genetic.Individual(offspring_weights, model=mother.model)
