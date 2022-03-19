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

        # each 'mother.weight', 'father.weight' we have a vector of weights
        # we must traverse the entire array and sample a random one for each
        for m, f in zip(mother.weights, father.weights):
            shape = m.shape

            # easier to traverse them as flattened arrays
            m = m.flatten()
            f = f.flatten()

            result = []
            for i in range(m.shape[0]):
                if random.uniform() > self.keep_probability:
                    result.append(self.initializer())
                    continue
                result.append(random.choice([m[i], f[i]]))

            offspring_weights.append(np.array(result).reshape(shape))

        return kersa_genetic.Individual(offspring_weights, model=mother.model)
