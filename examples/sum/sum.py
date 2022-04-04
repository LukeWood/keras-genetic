import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import keras_genetic

model = keras.Sequential(
    [
        keras.Input(shape=(1,)),
        layers.Dense(3, activation="relu", use_bias=False),
    ]
)


def evaluate_model(individual: keras_genetic.Individual):
    return -(np.abs(keras_genetic.utils.weights.flatten(individual.weights))).sum()


print(
    "Sum of weights before:",
    -(np.abs(keras_genetic.utils.weights.flatten(model.get_weights()))).sum()
)

results = keras_genetic.search(
    model=model,
    # computational cost is evaluate*generations*population_size
    evaluator=evaluate_model,
    generations=1000,
    population_size=10,
    breeder=keras_genetic.breeder.CMABreeder(model, 4),
    return_best=1,
)

model = results.best.load_model()

print(
    "Sum of weights after:",
    -(np.abs(keras_genetic.utils.weights.flatten(model.get_weights()))).sum()
)
