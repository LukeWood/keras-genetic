"""search is the primary entrypoint to keras-genetic."""

from keras_genetic import core
from keras_genetic.callbacks import ProgBarLogger
from keras_genetic.search.result import SearchResult


def search(
    model,
    generations,
    population_size,
    evaluator,
    breeder,
    initial_population=None,
    return_best=1,
    verbose=1,
    callbacks=(),
):
    """`search()` is the primary entrypoint to keras-genetic.

    `search()` takes in a keras.Model, a breeder, and an evaluator.
    Using these three parameters, it searches the weight space and returns the
    n-best weights according to the `return_best` parameter.

    Args:
        model: the keras.Model to run the seach over.
        generations: number of generations to run the genetic algorithm for.
        population_size: size of the population.
        evaluator: `keras_genetic.Evaluator` instance, or callable.  Must take
            a model instance and return a number indicating the fitness
            of the individual.  If you wish to minimize a loss, simply
            invert the result of your loss function.
        breeder: `keras_genetic.Breeder` instance (or callable) used to produce
            new offspring.
        return_best: number of `keras_genetic.Individual` to return.
        verbose: verbosity to use.  1 for logging, 0 for silent.
        initial_population: an initial population of `keras_genetic.Individual` to use
            in training.  If `None`, a random population is created.
        callbacks: list of `keras_genetic.Callback` for use in training.
    Returns:
        `keras_genetic.search.SearchResult` object containing all of the search results
    """
    if not (isinstance(callbacks, list) or isinstance(callbacks, tuple)):
        callbacks = [callbacks]
    callbacks = list(callbacks)
    if verbose == 1:
        callbacks = callbacks + [ProgBarLogger(generations)]
    search_manager = _SearchManager(
        generations,
        population_size,
        initial_population,
        breeder,
        evaluator,
        return_best,
        callbacks,
    )
    result = search_manager.run(model)
    if verbose == 1:
        print(
            f"🎉 search() complete in {search_manager.generations} generations, "
            f"best fitness score: {result.best.fitness} 🎉"
        )
    return result


class _SearchManager:
    def __init__(
        self,
        generations,
        population_size,
        initial_population,
        breeder,
        evaluator,
        return_best,
        callbacks,
    ):
        self.best_individuals = []

        self.generations = generations
        self.initial_population = initial_population
        self.population_size = population_size
        self.breeder = breeder
        self.evaluator = evaluator
        self.return_best = return_best
        self.callbacks = callbacks
        self.should_stop = False

        for cb in self.callbacks:
            cb._manager = self

    def initial_generation(self, model, dummy_individual):
        return [
            self.breeder.fully_random_weight_set() for _ in range(self.population_size)
        ]

    def stop(self):
        self.should_stop = True

    def run(self, model):
        initial_parent = core.Individual(model.get_weights(), model)
        population = self.initial_population or self.initial_generation(
            model, initial_parent
        )
        parents = []

        for g in range(self.generations):
            # Call generation begin callbacks
            temp_result = SearchResult(population)
            for callback in self.callbacks:
                callback.on_generation_begin(g, temp_result)

            # evaluate all individuals in the current population
            for individual in population:
                if individual.fitness is None:
                    individual.fitness = self.evaluator(individual)

            # rank individuals by score
            population = sorted(population, reverse=True)
            # update breeder with results
            self.breeder.update_state(population)
            # store the best individuals from all runs
            # these are ultimately returned to users
            # also they are used in on_generation_end callbacks
            self.best_individuals = sorted(
                self.best_individuals + population, reverse=True
            )[: self.return_best]

            temp_result = SearchResult(self.best_individuals)
            for callback in self.callbacks:
                callback.on_generation_end(g, temp_result)

            if self.should_stop or (g == self.generations - 1):
                break

            population = self.breeder.population(self.population_size)

        return SearchResult(self.best_individuals)
