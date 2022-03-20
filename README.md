# Keras Genetic

*⚠️ Keras Genetic is Operating pre-release.  The API is currently unstable and will regularly change until the v1.0.0 release. ⚠️*

Keras Genetic allows you to easily train Keras models using genetic algorithms.

*Quick Links:*

- [Cartpole](examples/cartpole/cartpole.py)
- [MNIST Image Classification](examples/mnist/mnist.py)
- [Overview](#Overview)
- [Quickstart](#Quickstart)

## Background
Keras provides an elegant API for creating neural networks.  Typically, the
neural network weights are optimized by minimizing a loss function through the
process of gradient descent.

Keras Genetic takes a different approach to weight optimization by leveraging
genetic algorithms.  Genetic algorithms allow you to optimize a neural network
without in scenarios where there is no information about the loss landscape.

Some areas where genetic algorithms are applied today:

- [Reinforcement learning (WorldModels)](https://worldmodels.github.io/)
- Finance
- Computer architecture
- code breaking
- hardware bug searching
- and many more!

## Overview

The Keras genetic API is quick to get started with, but flexible enough to fit
any use case you may come up with.

There are three core components of the API that must be used to get started:

- the `Individual`
- the `Evaluator`
- the `Breeder`
- `search()`

### Individual

The `Individual` class represents an individual in the population.

The most important method on the `Individual` class is `load_model()`.
`load_model()` yields a Keras model with the weights stored on the `individual`
class loaded:

```python
model = individual.load_model()
model.predict(some_data)
```

### Evaluator

Next, lets go over the `Evaluator`.  The `Evaluator` is responsible for
determining the strength of an `Individual`.  Perhaps the simplest
evaluator is an accuracy evaluator for a classification task:

```python
def evaluate_accuracy(individual: keras_genetic.Individual):
    model = individual.load_model()
    result = model.evaluate(x_train[:100], y_train[:100], return_dict=True, verbose=0)
    return result["accuracy"]
```

The `evaluate_accuracy()` function defined above maps from an `Individual` to an
accuracy score.  This score can be used to select the individuals that will be
used in the next generation.

### Breeder

The `Breeder` is responsible with producing new individuals from a set of parent
individuals.  The details as to how each `Breeder` produces new individuals are
unique to the  breeder, but as a general rule some attributes of the parent are
preserved while some new attributes are randomly sampled.

For most users, the `TwoParentMutationBreeder` is sufficiently effective.

### search()
`search()` is akin to `model.fit()` in the core Keras framework.  The `search()` API
supports a wide variety of parameters.  For an in depth view, browse the API docs.

Here is a sample usage of the `search()` function:

```python
results = keras_genetic.search(
    model=model,
    # computational cost is evaluate*generations*population_size
    evaluator=evaluate_accuracy,
    generations=10,
    population_size=50,
    n_parents_from_population=5,
    breeder=keras_genetic.breeder.TwoParentMutationBreeder(),
    return_best=1,
)
```

### Further Reading
Check out the [examples](examples/) and guides (Coming Soon!).

## Quickstart
For now, the [MNIST Example](examples/mnist/mnist.py) serves as the Quickstart guide.
