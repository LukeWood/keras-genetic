from math import floor

from keras_genetic.callbacks.callback import Callback


class ProgBarLogger(Callback):
    def __init__(self, generations):
        self.generations = generations
        self.progbar = ProgBar(generations=generations)

    def on_generation_end(self, generation, result):
        print(
            self.progbar.render(generation, result.best.fitness), end="\r", flush=True
        )
        if generation == self.generations:
            print()


class ProgBar:
    def __init__(
        self,
        generations,
        width=30,
    ):
        self.width = width
        self.generations = generations

    def render(self, generation, fitness):
        percent_done = generation / self.generations
        result = (["="] * floor(percent_done * self.width)) + (
            [" "] * floor((1 - percent_done) * self.width)
        )
        result = "".join(result)
        result = result[::-1].replace("=", ">", 1)[::-1]

        if len(result) < self.width:
            result += " "
        return f"({generation}/{self.generations}) [{result}] Best Fitness: {fitness}"
