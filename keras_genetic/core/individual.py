import numpy as np

class Individual:
    def __init__(self, weights, score=None):
        self.weights = weights
        self.score = score
