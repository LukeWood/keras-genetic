import keras_genetic
import numpy as np

class RandomNormal(keras_genetic.initializers.Initializer):
  def call(self):
    return np.random_normal()
