import numpy as np


def flatten(weights):
    """converts a list of `np.array`s to a 1D `np.array`.

    Args:
        weights: list of `np.array`s to flatten.
    """
    return np.concatenate([w.flatten() for w in weights], axis=0)


def conform_weights_to_shape(weights, target_weights):
    """conform_weights_to_shape() conforms a 1D weight array to a target array shape.

    Takes a 1D array `weights` and a list of `target_weights` and returns a list with
    elements of type `np.array` where each element contains values taken
    sequentially from `weights` but the shapes match those of `target_weights`.

    Args:
        weights: 1D `np.array` of weights to use in the result.
        target_weights: an array with elements of type `np.array`.  This list is
        traversed and each elements' `.shape` attribute is used to target

    Returns:
        a list with elements of type `np.array` where each element contains values taken
        sequentially from `weights` but the shapes match those of `target_weights`.
    """
    n_weights = weights.shape[0]

    idx = 0
    final_result = []
    for template_weight in target_weights:
        shape = template_weight.shape
        template_weight = template_weight.flatten()

        result = []
        for i in range(template_weight.shape[0]):
            if idx > n_weights:
                raise ValueError(
                    "Weights exhausted during model population.  Did you pass "
                    "the correct model to your `Breeder()`?"
                )
            result.append(weights[idx])
            idx += 1
        final_result.append(np.array(result).reshape(shape))

    if idx != n_weights:
        raise ValueError(
            "Not all weights were used when producing an offspring.  Did you pass "
            "the correct model to your `Breeder()`?"
        )
    return final_result
