"""search is the primary entrypoint to keras-genetic."""

from typing import List

import numpy as np
import tqdm
from tensorflow import keras

from keras_genetic import core
from keras_genetic.search.result import SearchResult


def search(
    model,
    generations,
    population_size,
    n_parents_from_population,
    evaluator,
    breeder=None,
    return_best=1,
):
    """`search()` is the primary entrypoint to keras-genetic.

    `search()` takes in a keras.Model, a breeder, and an evaluator.
    Using these three parameters, it searches the weight space and returns the
    n-best weights according to the `return_best` parameter.

    Args:
        model: the keras.Model to run the seach over.
        generations: number of generations to run the genetic algorithm for.
        population_size: size of the population.
        n_parents_from_population: number of parents to keep from population.
        evaluator: `keras_genetic.Evaluator` instance, or callable.  Must take
            a model instance and return a number indicating the fitness score
            of the individual.  If you wish to minimize a loss, simply
            invert the result of your loss function.
        breeder: `keras_genetic.Breeder` instance (or callable) used to produce
            new offspring.  Defaults to `RandomFeatureMutationBreeder`.
        return_best: number of `keras_genetic.Individual` to return.
    Returns:
        `keras_genetic.search.SearchResult` object containing all of the search results
    """
    breeder = breeder or keras_genetic.CompletelyRandomBreeder()
    search_manager = _SearchManager(
        generations,
        population_size,
        n_parents_from_population,
        breeder,
        evaluator,
        return_best,
    )
    return search_manager.run(model)


class _SearchManager:
    def __init__(
        self,
        generations,
        population_size,
        n_parents_from_population,
        breeder,
        evaluator,
        return_best,
    ):
        self.generation = 0
        self.generations = generations
        self.population_size = population_size
        self.n_parents_from_population = n_parents_from_population
        self.breeder = breeder
        self.evaluator = evaluator
        self.return_best = return_best

    def initial_generation(self, model, dummy_individual):
        return [
            self.breeder.fully_random_weight_set(dummy_individual)
            for _ in range(self.population_size)
        ]

    def run_generation(self, population, parents, keep):
        for individual in population:
            individual.score = self.evaluator(individual)

        result_population = sorted(population + parents, reverse=True)
        return result_population[:keep]

    def run(self, model):
        dummy_individual = core.Individual(model.get_weights(), model)
        population = self.initial_generation(model, dummy_individual)
        parents = []

        for _generation in tqdm.tqdm(range(self.generations)):
            parents = self.run_generation(
                population, parents, keep=self.n_parents_from_population
            )
            population = self.breeder.population_from_parents(
                parents, self.population_size
            )

        final = self.run_generation(population, [], keep=self.return_best)
        return SearchResult(final)
