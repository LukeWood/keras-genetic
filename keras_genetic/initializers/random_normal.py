import numpy as np

from keras_genetic.initializers.initializer import Initializer


class RandomNormal(Initializer):
    def __init__(self, mean=0, std=10):
        self.mean = mean
        self.std = std

    def __call__(self):
        return np.random.normal(self.mean, self.std)
