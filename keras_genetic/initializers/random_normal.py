import keras_genetic
import numpy as np

class RandomNormal(keras_genetic.initializers.Initializer):
  def __call__(self):
    return np.random_normal()
