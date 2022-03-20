import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import keras_genetic.callbacks


class CartpoleGifCallback(keras_genetic.callbacks.Callback):
    def __init__(self, env):
        self.env = env

    def save_frames_as_gif(self, frames, path, filename, generation):

        # Mess with this to change frame size
        plt.figure(
            figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72
        )

        patch = plt.imshow(frames[0])
        plt.axis("off")

        def animate(i):
            patch.set_data(frames[i])
            plt.title(f"Generation {generation}")

        anim = animation.FuncAnimation(
            plt.gcf(), animate, frames=len(frames), interval=50
        )
        anim.save(path + filename, writer="imagemagick", fps=60)

    def on_generation_end(self, generation, result):
        env = self.env
        state = env.reset()

        model = result.best.load_model()
        frames = []
        for _ in range(250):
            frames.append(env.render(mode="rgb_array"))
            action_probs = model(np.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_probs))
            state, reward, done, _ = env.step(action)

        self.save_frames_as_gif(
            frames,
            path=f"./artifacts/cartpole/",
            filename=f"cartpole-generation-{generation}.gif",
            generation=generation,
        )
