import keras_genetic.breeder.cma_breeder
import keras
from keras import layers


model = keras.Sequential(layers.Dense(5))
model.build((None, 1,))

breeder = keras_genetic.breeder.cma_breeder.CMABreeder(model=model, recombination_parents=2, population_size=20)

individual = keras_genetic.Individual(fitness=0, weights=model.get_weights(), model=model)
breeder.update_state([individual])
breeder.offspring()
