import keras_genetic
import random
from tensorflow import keras
from tensorflow.keras import initializers

class RandomFeatureMutationBreeder(keras_genetic.Breeder):
    """RandomFeatureMutationBreeder randomly mutates features on individuals."""

    def __init__(self,
        keep_probability=0.9,
        initializer=None,
    ):
        self.keep_probability = keep_probability
        self.initializer = initializer

    def offspring(self, mother, father):
        offspring_weights = []

        # each 'mother.weight', 'father.weight' can have a full shape
        # we must traverse the entire array

        for m, f in zip(mother.weights, father.weights):
            shape = m.shape
            m = m.flatten()
            f = f.flatten()

            result = []

            if random.uniform() > self.keep_probability:
                offspring_weights.append(self.initializer())
                continue
            offspring_weights.append(random.choice([m, f]))
