import math

import numpy as np

from keras_genetic import core
from keras_genetic import utils
from keras_genetic.breeder.breeder import Breeder


class CMABreeder(Breeder):
    """CMABreeder produces offspring using the CMA-ES Algorithm.

    Covariance matrix adaptation evolution strategy (CMA-ES) is an evoluationary
    algoirthm for use in optimization problems.

    References:
    - https://en.wikipedia.org/wiki/CMA-ES
    - https://github.com/CMA-ES/pycma
    """

    def __init__(self, model, recombination_parents, population_size, discount_factor=0.01, **kwargs):
        super().__init__(model, **kwargs)

        self.mean = np.random.normal((self.num_params,))
        self.sigma = 0.3

        self.discount_factor = discount_factor

        self.recombination_parents = recombination_parents  # mu
        self.population_size = population_size
        # initalize weights
        self.recombination_weights = np.log(recombination_parents + 1 / 2) - np.log(
            np.arange(0, recombination_parents) + 1
        )
        self.recombination_weights = (
            self.recombination_weights / self.recombination_weights.sum()
        )

        self.covariance_path = np.zeros((self.num_params, 1))
        self.sigma_path = np.zeros((self.num_params, 1))
        self.covariance_matrix = np.identity(self.num_params)

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
        old_mean = self.mean
        parents = generation[: self.recombination_parents]
        population_size = len(generation)

        self.mean = self._update_mean(
            old_mean, parents
        )
        self.sigma_path = self._update_sigma_path(old_mean, parents, self.discount_factor, population_size)
        self.covariance_path = self._update_covariance_path()
        self.covariance_matrix = self._update_covariance_matrix()
        self.sigma = self._update_sigma()

    def _update_mean(self, old_mean, parents):
        parents_concat = np.array(
            [utils.flatten(candidate.weights) for candidate in parents]
        )
        print("parents_concat.shape", parents_concat.shape)
        # broadcast mean over the batch dimension, which is the number of parents
        delta = parents_concat - self.mean[None, ...]
        print("delta.shape", delta.shape)
        # broadcast the weights across the num_params dimension
        updates = delta * self.recombination_weights[..., None]
        print("updates.shape", updates.shape)
        # sum over the individual num_parents axis
        update = updates.sum(axis=0)
        print("update.shape", update.shape)
        return old_mean + update

    def _update_sigma_path(self, old_mean, parents, discount_factor, population_size):
        sigma_path = self.sigma_path

        # can be thought of as a slow degeneration
        discount = (1 - discount_factor) * sigma_path

        displacement_discount = math.sqrt(1 - (1 - discount_factor) ** 2)
        # we can probably just cache this, but localizing it makes the code more readable
        inverse_weight_effectiveness = math.sqrt(1 / (np.power(self.recombination_weights, 2)).sum())
        inverse_covariance_sqrt = self._inverse_covariance_sqrt()
        print('self.covariance_matrix.shape', self.covariance_matrix.shape)
        print('inverse_covariance_sqrt.shape', inverse_covariance_sqrt.shape)
        # should precompute and cache
        parents_concat = np.array(
            [utils.flatten(candidate.weights) for candidate in parents]
        )
        mean_displacement = parents_concat - self.mean[None, ...]
        mean_displacement = mean_displacement.sum(axis=0)
        print('mean_displacement.shape', mean_displacement.shape)

        displacement_factor = inverse_weight_effectiveness * np.matmul(
            inverse_covariance_sqrt, mean_displacement / self.sigma
        )

        return discount + (displacement_discount * displacement_factor)

    def _inverse_covariance_sqrt(self):
        w, v = np.linalg.eig(self.covariance_matrix)
        scaling = np.sqrt(np.diag(v))
        print('scaling.shape', scaling.shape)
        print('np.diag(scaling).shape', np.diag(scaling).shape)
        print('w.shape', w.shape)

        scaling = np.diag(scaling)
        scaling = np.divide(1, scaling, out=np.zeros_like(scaling), where=scaling !=0)
        return w * scaling * np.transpose(w)

    def _update_covariance_path(self):
        return self.covariance_path

    def _update_covariance_matrix(self):
        return self.covariance_matrix

    def _update_sigma(self):
        return self.sigma

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
