import math

import numpy as np

from keras_genetic import core
from keras_genetic import utils
from keras_genetic.breeder.breeder import Breeder


class CMABreeder(Breeder):
    """CMABreeder produces offspring using the CMA-ES Algorithm.

    Covariance matrix adaptation evolution strategy (CMA-ES) is an evoluationary
    algoirthm for use in optimization problems.  Implementation is based on the
    mathematical overview provided in the wikipedia article.

    References:
    - https://en.wikipedia.org/wiki/CMA-ES
    - https://github.com/CMA-ES/pycma
    """

    def __init__(
        self,
        model,
        recombination_parents,
        discount_factor_sigma=0.01,
        discount_factor_covariance=0.01,
        **kwargs
    ):
        super().__init__(model, **kwargs)

        # strategy settings
        self.N = self.num_params
        self.mean = np.random.normal(size=(self.num_params,))
        self.sigma = 0.3
        self.lamb = 4 + math.floor(3 * math.log(self.N))

        self.mu = recombination_parents

        self.weights = np.log(self.mu + 1 / 2) - np.log(np.arange(0, self.mu) + 1)
        self.weights = self.weights / self.weights.sum()
        self.mueff = self.weights.sum() ** 2 / (self.weights * self.weights).sum()

        # adaptation settings
        self.cc = (4 + self.mueff / self.N) + (self.N + 4 + 2 * self.mueff / self.N)
        self.cs = (self.mueff + 2) / (self.N + self.mueff + 5)
        self.c1 = 2 / ((self.N + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1 - self.c1,
            2 * (self.mueff - 2 + 1 / self.mueff) / ((self.N + 2) ** 2 + self.mueff),
        )
        self.damps = (
            1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.N + 1)) - 1) + self.cs
        )

        # dynamic internal parameters and constants
        self.pc = np.zeros((self.N,))
        self.ps = np.zeros((self.N,))
        self.B = np.identity(self.N)
        self.D = np.ones((self.N, ))

        self.C = self.B @ np.diag(self.D * self.D) @ self.B.transpose()
        self.invsqrtC = self.B @ np.diag(1 / self.D) @ self.B.transpose()
        self.eigeneval = 0
        self.chiN = np.sqrt(self.N) * (1 - 1 / (4 * self.N) + 1 / (21 * self.N**2))

        self.counteval = 0

    def offspring(self):
        """offspring() samples from a multivariate normal.

        offspring() ignores parents, and instead relies on `self.mean` and
        `self.covariance_matrix`  to produce a new offspring.

        In pseudocode, this function does:

        def offspring(self):
            return sample_multivariate_normal(mean=mean, covariance_matrix=sigma^2 * C)

        In reality, we also need to reshape the weights matrix to also fit the spec of
        parents[0].
        """
        weights = self._sample_weights()
        return core.Individual(
            weights=utils.conform_weights_to_shape(weights, self.model.get_weights()),
            model=self.model,
        )

    def update_state(self, generation):
        self.counteval += 1
        # template is used to pass the model down to the children
        xold = self.mean
        # parents are the individuals picked for use in the mean
        parents = generation[: self.recombination_parents]
        population_size = len(generation)

        self.mean = self._update_mean(xold, parents, self.recombination_weights)
        self.sigma_path = self._update_sigma_path(old_mean)
        hsig = self._compute_hsig(self.sigma_path, population_size)

    def _update_mean(self, old_mean, parents, recombination_weights):
        parents_concat = np.array(
            [utils.flatten(candidate.weights) for candidate in parents]
        )
        # broadcast mean over the batch dimension, which is the number of parents
        delta = parents_concat - self.mean[None, ...]
        # broadcast the weights across the num_params dimension
        updates = delta * recombination_weights[..., None]
        # sum over the individual num_parents axis
        update = updates.sum(axis=0)
        return old_mean + update

    def _compute_cached_covariance_by_products(self, covariance_matrix):
        w, v = np.linalg.eig(covariance_matrix)
        print("w", w)
        print("v", v)

        scaling = np.sqrt(np.diag(v))
        print("scaling", scaling)
        covariance_eigenvectors = w

        # update inverse_covariance_sqrt
        intermediary_scaling = np.divide(
            1,
            scaling,
            out=np.zeros_like(scaling),
            where=scaling != 0,
        )
        inverse_covariance_sqrt = w * np.diag(intermediary_scaling) * np.transpose(w)
        return scaling, covariance_eigenvectors, inverse_covariance_sqrt

    def _sample_weights(self):
        w, v = np.linalg.eig(self.covariance_matrix)
        scaling = np.sqrt(np.diag(v))

        sample = np.random.normal((self.num_params,)) * scaling
        weights = self.mean + sample * self.sigma
        return weights.flatten()

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        return result
