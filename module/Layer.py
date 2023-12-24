import numpy as np

from module.Function import ReLU

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)

    def layer_forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output