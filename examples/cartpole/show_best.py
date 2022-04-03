import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import keras_genetic.callbacks


class ShowBest(keras_genetic.callbacks.Callback):
    def __init__(self, env):
        self.env = env

    def on_generation_end(self, generation, result):
        env = self.env
        state = env.reset()

        model = result.best.load_model()
        frames = []
        done = False
        while not done:
            env.render()
            action_probs = model(np.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_probs))
            state, reward, done, _ = env.step(action)
        for _ in range(100):
            env.render()
            action_probs = model(np.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_probs))
            state, reward, done, _ = env.step(action)
