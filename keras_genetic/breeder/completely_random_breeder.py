import keras_genetic


class CompletelyRandomBreeder(keras_genetic.Breeder):
    """CompletelyRandomBreeder creates fully random weight sets."""

    def offspring(self, parents):
        mother = parents[0]
        return self.fully_random_weight_set(mother)
