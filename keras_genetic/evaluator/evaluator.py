import keras_genetic

class Evaluator:
    def __call__(self, individual: keras_genetic.Individual):
        raise NotImplementedError("A Evaluator must implement __call__(individual)."
        )
