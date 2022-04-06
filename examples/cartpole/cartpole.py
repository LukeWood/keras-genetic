import gym
import numpy as np
from show_best import ShowBest
from tensorflow import keras
from tensorflow.keras import layers

import keras_genetic

env = gym.make("CartPole-v1")

num_inputs = 4
num_actions = 2
num_hidden = 128

inputs = layers.Input(shape=(num_inputs,))
hidden = layers.Dense(num_hidden, activation="relu")(inputs)
action = layers.Dense(num_actions, activation="softmax")(hidden)
model = keras.Model(inputs, action)


def evaluate_cartpole(individual: keras_genetic.Individual):
    model = individual.load_model()
    state = env.reset()
    total_reward = 0

    done = False
    while not done:
        action_probs = model(np.expand_dims(state, axis=0))
        action = np.argmax(np.squeeze(action_probs))
        state, reward, done, _ = env.step(action)
        total_reward += reward

    return total_reward


show_best = ShowBest(gym.make("CartPole-v1"))

results = keras_genetic.search(
    # computational cost is evaluate*generations*population_size
    model=model,
    evaluator=evaluate_cartpole,
    generations=20,
    population_size=20,
    breeder=keras_genetic.breeder.RandomWeightBreeder(model, parents_per_generation=3),
    return_best=1,
    callbacks=[keras_genetic.callbacks.EarlyStopping(goal=500.0), show_best]
    # note: the CartpoleGifCallback is super expensive because of the video
    # processing.  Only enable it if you really want to create the gifs.
    # callbacks=[CartpoleGifCallback(env)],
)

# show the best model one more time
show_best.on_generation_end(-1, results)
