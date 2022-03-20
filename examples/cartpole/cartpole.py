import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

import keras_genetic

env = gym.make("CartPole-v0")

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
    for _ in range(500):
        action_probs = model(np.expand_dims(state, axis=0))
        action = np.argmax(np.squeeze(action_probs))
        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break

    return total_reward


results = keras_genetic.search(
    # computational cost is evaluate*generations*population_size
    model=model,
    evaluator=evaluate_cartpole,
    generations=5,
    population_size=10,
    n_parents_from_population=3,
    breeder=keras_genetic.breeder.TwoParentMutationBreeder(),
    return_best=1,
    # note: the CartpoleGifCallback is super expensive because of the video
    # processing.  Only enable it if you really want to create the gifs.
    # callbacks=[CartpoleGifCallback(env)],
)

model = results.best.load_model()

state = env.reset()
done = False
while not done:
    env.render()
    action_probs = model(np.expand_dims(state, axis=0))
    action = np.argmax(np.squeeze(action_probs))
    state, reward, done, _ = env.step(action)
