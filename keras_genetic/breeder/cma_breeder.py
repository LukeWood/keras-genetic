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

        self.mean = np.random.normal(size=(self.num_params,))
        self.sigma = 0.3

        self.discount_factor_sigma = discount_factor_sigma
        self.discount_factor_covariance = discount_factor_covariance

        self.recombination_parents = recombination_parents  # mu

        # initalize weights
        self.recombination_weights = np.log(recombination_parents + 1 / 2) - np.log(
            np.arange(0, recombination_parents) + 1
        )
        self.recombination_weights = (
            self.recombination_weights / self.recombination_weights.sum()
        )
        self.mu_effictive = (
            np.matmul(self.recombination_weights, self.recombination_weights)
            / np.multiply(self.recombination_weights, self.recombination_weights)
        ).sum()

        self.cs = (self.mu_effictive + 2) / (self.num_params + self.mu_effictive + 5)

        self.dampen_sigma = (
            1
            + 2 * max(0, math.sqrt((self.mu_effictive - 1) / (self.num_params + 1)) - 1)
            + self.cs
        )

        self.covariance_path = np.zeros((self.num_params,))
        self.sigma_path = np.zeros((self.num_params,))
        self.covariance_matrix = np.identity(self.num_params)
        (
            self.scaling,
            self.covariance_eigenvectors,
            self.inverse_covariance_sqrt,
        ) = self._compute_cached_covariance_by_products(self.covariance_matrix)

        self.expected_norm_of_random = (
            math.sqrt(self.num_params) * 1
            - (1 / 4 * self.num_params)
            + 1 / (21 * (self.num_params**2))
        )

        self.inverse_weight_effectiveness = 1 / (
            (self.recombination_weights * self.recombination_weights).sum()
        )

        self.counteval = 0

    def _compute_cached_covariance_by_products(self, covariance_matrix):
        self.covariance_matrix = np.triu(self.covariance_matrix) + np.transpose(
            np.triu(self.covariance_matrix, k=1)
        )
        v, w = np.linalg.eigh(covariance_matrix)
        # values in w
        # vectors in v
        print("w", w)
        print("v", v)
        scaling = np.sqrt(np.abs(v))
        print("scaling", scaling)
        covariance_eigenvectors = w
        print("covariance_eigenvectors", covariance_eigenvectors)

        # update inverse_covariance_sqrt
        intermediary_scaling = np.divide(
            1,
            np.diag(scaling),
            out=np.zeros_like(np.diag(scaling)),
            where=scaling != 0,
        )
        print("intermediary_scaling", intermediary_scaling)
        inverse_covariance_sqrt = w * np.diag(intermediary_scaling) * np.transpose(w)
        print("scaling", scaling)
        print("inverse_covariance_sqrt", inverse_covariance_sqrt)
        print("covariance_eigenvectors", covariance_eigenvectors)
        # input('cached covariance products')
        return scaling, covariance_eigenvectors, inverse_covariance_sqrt

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
        old_mean = self.mean
        # parents are the individuals picked for use in the mean
        parents = generation[: self.recombination_parents]
        population_size = len(generation)
        inverse_weight_effectiveness = self.inverse_weight_effectiveness

        self.mean = self._update_mean(old_mean, parents, self.recombination_weights)

        self.sigma_path = self._update_sigma_path(
            self.sigma_path,
            self.mean,
            old_mean,
            self.discount_factor_sigma,
            np.sqrt(inverse_weight_effectiveness),
            self.sigma,
        )
        self.covariance_path = self._update_covariance_path(
            self.covariance_path,
            self.mean,
            old_mean,
            self.sigma,
            self.discount_factor_covariance,
            self.sigma_path,
            math.sqrt(inverse_weight_effectiveness),
        )
        self.covariance_matrix = self._update_covariance_matrix(
            self.covariance_matrix,
            self.covariance_path,
            generation,
            population_size,
            old_mean,
            self.sigma,
            self.sigma_path,
            2 / (self.num_params**2),  # c1
            inverse_weight_effectiveness / (self.num_params**2),  # cs
            1 / (self.num_params / 4),
        )
        self.sigma = self._update_sigma(
            self.sigma,
            self.discount_factor_sigma,
            self.dampen_sigma,
            self.sigma_path,
            self.expected_norm_of_random,
        )
        print("dampen_sigma", self.dampen_sigma)
        input()

    def _compute_hsig(
        self,
        sigma_path,
        discount_factor,
        population_size,
    ):
        intermediary = np.linalg.norm(path_sigma) / math.sqrt(
            1 - (1 - discount_factor) ^ (2 * self.counteval / population_size)
        ) / expected_norm_of_random < 1.4 + 2 / (self.num_params + 1)

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

    def _update_sigma_path(
        self,
        sigma_path,
        new_mean,
        old_mean,
        discount_factor,
        weight_effectiveness,
        sigma,
    ):
        # can be thought of as a slow degeneration
        left_term = (1 - discount_factor) * sigma_path
        print("weight_effectiveness", weight_effectiveness)
        right_term = self._complement_discount_factor(discount_factor)
        print("complement_discount", right_term)

        print("inverse_covariance_sqrt", self.inverse_covariance_sqrt)
        print()
        right_term = (
            weight_effectiveness
            * right_term
            * self.inverse_covariance_sqrt
            @ ((new_mean - old_mean) / sigma)
        )
        print("left_term", left_term)
        print("right_term", right_term)
        print("sigma_path_result", left_term + right_term)
        return left_term + right_term

    def _update_covariance_path(
        self,
        covariance_path,
        new_mean,
        old_mean,
        sigma,
        discount_factor,
        path_sigma,
        inverse_weight_effectiveness,
    ):
        left_term = (1 - discount_factor) * covariance_path

        path_sigma_norm = np.linalg.norm(path_sigma)
        indicator_function = self._indicator_function(path_sigma_norm)

        complement_discount_factor = self._complement_discount_factor(discount_factor)
        right_term = inverse_weight_effectiveness * (old_mean - new_mean) / sigma

        return left_term + right_term * indicator_function

    def _complement_discount_factor(self, discount_factor):
        return math.sqrt(1 - (1 - discount_factor) ** 2)

    def _indicator_function(self, x):
        # alpha hard coded to 1.5
        if x >= 0 and x <= 1.5 * np.sqrt(self.num_params):
            return 1
        else:
            return 0

    def _update_covariance_matrix(
        self,
        covariance_matrix,
        path_covariance,
        population,
        population_size,
        old_mean,
        sigma,
        path_sigma,
        base_learning_rate,  # c1
        mu_learning_rate,  # cs
        c_c,  # C_c^-1 should be n/4, so this value is computed based on n
    ):
        # update covariance_matrix
        covariance_matrix = self.covariance_matrix

        path_sigma_norm = np.linalg.norm(path_sigma)
        indicator_function = self._indicator_function(path_sigma_norm**2)

        cs = 1 - indicator_function
        cs = cs * base_learning_rate * mu_learning_rate * (2 - c_c)

        left_term = (
            1 - (base_learning_rate) - mu_learning_rate + cs
        ) * covariance_matrix

        middle_term = base_learning_rate * np.matmul(
            path_covariance, np.transpose(path_covariance)
        )
        # mu = self.recombination_parents
        # lambda = population_size

        parents_concat = np.array(
            [
                utils.flatten(candidate.weights)
                for candidate in population[: self.recombination_parents]
            ]
        )
        # array call stacks to first axis, we want this on the last axis
        parents_concat = np.transpose(parents_concat)
        artmp = (1 / sigma) * (parents_concat - old_mean[..., None])
        right_hand_term = np.matmul(artmp, np.diag(self.recombination_weights))
        right_hand_term = np.matmul(right_hand_term, np.transpose(artmp))
        # broadcast mean over the batch dimension, which is the number of parents

        # TODO - correctness check right term
        # Should right term actually consist of a slice for delta - from 0:mu for each
        # mu in range 0, mu?
        right_term = mu_learning_rate * right_hand_term
        updated_covariance_matrix = left_term + middle_term + right_term

        # update scaling and eigenvectors
        (
            self.scaling,
            self.covariance_eigenvectors,
            self.inverse_covariance_sqrt,
        ) = self._compute_cached_covariance_by_products(updated_covariance_matrix)
        return updated_covariance_matrix

    def _update_sigma(
        self, sigma, discount_factor, dampen_sigma, sigma_path, expected_norm_of_random
    ):
        print("(discount_factor / dampen_sigma)", (discount_factor / dampen_sigma))
        print("sigma_path", self.sigma_path)
        print("(np.linalg.norm(sigma_path)", np.linalg.norm(sigma_path))
        print(
            "(np.linalg.norm(sigma_path)/expected_norm_of_random",
            np.linalg.norm(sigma_path) / expected_norm_of_random,
        )
        print(
            "sigma",
            sigma,
            "coeeff",
            math.exp(
                (discount_factor / dampen_sigma)
                * ((np.linalg.norm(sigma_path) / expected_norm_of_random) - 1)
            ),
        )
        result = sigma * math.exp(
            (discount_factor / dampen_sigma)
            * ((np.linalg.norm(sigma_path) / expected_norm_of_random) - 1)
        )
        print("result", result)
        # input()

        return result

    def _sample_weights(self):
        sample = np.random.normal((self.num_params,)) * self.scaling
        weights = self.mean + sample * self.sigma
        return weights.flatten()

    def population(self, population_size):
        result = []
        for _ in range(population_size):
            result.append(self.offspring())
        return result
