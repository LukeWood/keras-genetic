"""search is the primary entrypoint to keras-genetic."""

from typing import List

import numpy as np
from dataclass import dataclass
from tensorflow import keras

from keras_genetic.search.result import SearchResult
from keras_genetic.search.result import SingleResult

def search(
    model,
    generations,
    population_size,
    breeder=None,
    evaluator=None,
    return_best=5,
):
    """`search()` is the primary entrypoint to keras-genetic.

    `search()` takes in a keras.Model, a breeder, and an evaluator.
    Using these three parameters, it searches the weight space and returns the
    n-best weights according to the `return_best` parameter.

    Args:
        model: the keras.Model to run the seach over.
        generations: number of generations to run the genetic algorithm for.
        population_size: size of the population.
        breeder: `keras_genetic.Breeder` instance used to produce new offspring.
            Defaults to `None`, which causes the framework to provide fully
            random weights for each offspring.
        evaluator: `keras_genetic.Evaluator` instance, or function.  Must take
            a model instance and return a number indicating the fitness score
            of the individual.  If you wish to minimize a loss, simply
            invert the result of your loss function.
    Returns:
        `keras_genetic.search.SearchResult` object containing all of the search results
    """
    pass
