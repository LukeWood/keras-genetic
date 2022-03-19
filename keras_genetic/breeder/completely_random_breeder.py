import random

from tensorflow import keras
from tensorflow.keras import initializers

import keras_genetic


class CompletelyRandomBreeder(keras_genetic.Breeder):
    """CompletelyRandomBreeder creates fully random weight sets."""

    def offspring(self, mother, _2):
        return self.fully_random_weight_set(mother)
