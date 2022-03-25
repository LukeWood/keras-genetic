# Keras Genetic

[![](https://github.com/lukewood/keras-genetic/workflows/Tests/badge.svg?branch=master)](https://github.com/lukewood/keras-genetic/actions?query=workflow%3ATests+branch%3Amaster)
![Python](https://img.shields.io/badge/python-v3.7.0+-success.svg)
![Tensorflow](https://img.shields.io/badge/tensorflow-v2.8.0+-success.svg)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/lukewood/keras-genetic/issues)

Keras Genetic allows you to easily train Keras models using genetic algorithms.

*Quick Links:*

- [Cartpole](examples/cartpole/cartpole.py)
- [MNIST Image Classification](examples/mnist/mnist.py)
- [Overview](#Overview)
- [Quickstart](#Quickstart)

## Background
KerasGenetic allows you to leverage the elegent modeling API Keras while performing
training with genetic algorithms.  Typically, Keras neural network weights are optimized
by minimizing a loss function through the process of gradient descent.

Keras Genetic takes a different approach to weight optimization by leveraging
genetic algorithms.  Genetic algorithms allow you to optimize a neural network
without in scenarios where there is no information about the loss landscape.

Genetic algorithms can be used to train neural networks in niche cases where you need to
train a specialized controller with <1000 parameters.

Some areas where genetic algorithms are applied today:

- [Reinforcement learning (WorldModels)](https://worldmodels.github.io/)
- Finance

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
    breeder=keras_genetic.breeder.MutationBreeder(),
    return_best=1,
)
```

### Further Reading

Check out the [examples](examples/) and guides (Coming Soon!).

## Quickstart

For now, the [Cartpole Example](examples/cartpole/cartpole.py) serves as the Quickstart guide.


## Roadmap

I'd like to accomplish the following tasks:

- ✅ stabilize the base API
- ✅ support a callbacks API
- ✅ end to end MNIST example
- ✅ end to end CartPole example
- ✅ implement a ProgBarLogger
- ✅ implement EarlyStopping callback (can be used in CartPole example)
- have at least 3 distinct breeders
- autogenerate documentation
- thoroughly document each component
- offer implementations of  the most effective genetic algorithms
- implement unit tests for each component
- support random seeding
- thoroughly review the API per Keras core API design guidelines
- support custom initial populations (i.e. to model after a human imitation model)
- support keep_probability schedules

Feel free to contribute any of these.

## Citation

```bibtex
@misc{wood2022kerasgenetic,
	title        = {Keras Genetic},
	author       = {Luke Wood},
	year         = 2022,
	howpublished = {\url{https://github.com/lukewood/keras-genetic}}
}
```
