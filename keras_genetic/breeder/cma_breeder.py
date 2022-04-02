import numpy as np

import keras_genetic


class CMABreeder(keras_genetic.Breeder):
    """CMABreeder produces offspring using the CMA-ES Algorithm.

    Covariance matrix adaptation evolution strategy (CMA-ES) is an evoluationary
    algoirthm for use in optimization problems.

    References:
    - https://en.wikipedia.org/wiki/CMA-ES
    - https://github.com/CMA-ES/pycma
    """

    def __init__(self, n, initializer=None):
        super().__init__(initializer=initializer)

        self.mean = np.random_normal((n, 1))
        self.sigma = 0.3
        self.p_sigma = None
        self.p_c = None
        self.covariance_matrix = np.identity(n, np.float32)
        self.mu = 2

        self.d = np.ones((n, 1), np.float32)

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
        del parents

        # What the hell is the point of self.b?  Stolen from matlab code - isn't it just
        # an identity matrix?
        sample = np.matmul(
            np.matmul(self.sigma, self.b)
            * np.multiply(self.d, np.random.normal((self.n, 1)))
        )
        weights = self.mean + sample

        # Restructure weights to fit in the passed Keras model
        weights = weights.flatten()
        n_weights = weights.shape[0]

        mother = self._template

        idx = 0

        offspring_weights = []
        for template_weight in mother.weights:
            shape = template_weight.shape
            template_weight = template_weight.flatten()

            result = []
            for i in range(template_weight.shape[0]):
                if idx > n_weights:
                    raise ValueError(
                        "Weights exhausted during model population.  Did you pass "
                        "the correct value for `n`?  `n` must match "
                        f"dim(model.get_weights().flatten())."
                    )
                result.append(weights[idx])
                idx += 1

            offspring_weights.append(np.array(result).reshape(shape))

        if idx != n_weights:
            raise ValueError(
                "Not all weights were used when producing an offspring.  Did you pass "
                "the correct value for `n`?  `n` must match "
                "dim(model.get_weights().flatten())"
            )
        return core.Individual(offspring_weights, model=template_model)

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
        self._template = generation[0]

        pass
