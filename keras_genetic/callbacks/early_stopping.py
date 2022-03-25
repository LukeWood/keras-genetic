from keras_genetic.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(
        self,
        goal,
    ):
        self.goal = goal

    def on_generation_end(self, generation, result):
        if result.best.fitness >= self.goal:
            self._manager.stop()
