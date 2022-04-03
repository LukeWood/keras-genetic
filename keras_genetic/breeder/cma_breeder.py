import math

import numpy as np

import keras_genetic.utils
from keras_genetic import core
from keras_genetic.breeder.breeder import Breeder


class CMABreeder(Breeder):
    """CMABreeder produces offspring using the CMA-ES Algorithm.

    Covariance matrix adaptation evolution strategy (CMA-ES) is an evoluationary
    algoirthm for use in optimization problems.

    References:
    - https://en.wikipedia.org/wiki/CMA-ES
    - https://github.com/CMA-ES/pycma
    """

    def __init__(self, model, recombination_parents, population_size, **kwargs):
        super().__init__(model, **kwargs)

        self.mean = np.random.normal((n,))
        self.sigma = 0.3

        self.recombination_parents = recombination_parents  # mu
        self.population_size = population_size
        # initalize weights
        self.weights = np.log(self.recombination_parents + 1 / 2) - np.log(
            np.arange(1, self.recombination_parents)
        )
        self.recombination_effectiveness = (np.sum(self.weights) ** 2) / np.sum(
            np.square(self.weights)
        )

        self.covariance_path = np.zeros((n, 1))
        self.sigma_path = np.zeros((n, 1))

        self.scaling = np.ones((n, 1))
        self.covariance_matrix = np.diag(np.square(self.scaling))
        self.invsqrt_covariance = np.diag(1 / self.covariance_matrix)
        self.eigeneval = 0
        self.chiN = np.power(n, 0.5) * (1 - (1 / (4 * n) + 1 / (21 * n**2)))

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
        return self._format_offspring(weights, self._template)

    def update_state(self, generation):
        """`update_state(generation)` method updates the state of the breeder class.

        In pseudocode:

        x1...xn = x_s1... x_sn -> individuals sorted based on fitness

        mean_prime = mean // we later need mean - mean_prime and x_i - mean_prime
        mean = update_mean(x1...xn) // move the mean to better solutions

        // update "isotropic evolution path"
        p_sigma = update_p_sigma(p_sigma, sigma^-1 * C^(-1/2) * (m - m_prime))
        // update anisotropic evolution path
        p_c = update_p_c(p_c, sigma^-1*(m-m_prime), norm(p_sigma))

        // update covariance matrix
        C = update_C(C, p_c, (x1 - mean_prime) / sigma, ... (x_n - m_prime) / sigma)

        // update step-size using isotropic path length
        sigma = update_sigma(sigma, norm(p_sigma))
        """
        # template is used to pass the model down to the children
        self._template = generation[0]

        mean_old = self.mean
        self.mean = self._update_mean()
        self.sigma_path = self._update_sigma_path()
        self.covariance_path = self._update_covariance_path()
        self.covariance_matrix = self._update_covariance_matrix()
        self.sigma = self._update_sigma()

    def _update_mean(self):
        return self.mean

    def _update_sigma_path(self):
        return self.sigma_path

    def _update_covariance_path(self):
        return self.covariance_path

    def _update_covariance_matrix(self):
        return self.covariance_matrix

    def _update_sigma(self):
        return self.sigma

    def _sample_weights(self):
        sample = self.sigma * np.multiply(self.scaling, np.random.normal((self.n,)))
        weights = self.mean + sample
        return weights.flatten()
