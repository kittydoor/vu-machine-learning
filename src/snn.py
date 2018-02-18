#!/usr/bin/env python
import numpy as np

class SNN:
    def __init__(self, in_layer_n, out_layer_n):
        self._in_layer_n = in_layer_n
        self._out_layer_n = out_layer_n
        self._weights = np.random.randn(
                self._out_layer_n,
                self._in_layer_n
                )
        self._biases = np.random.randn(self._out_layer_n)
    
    def sigmoid(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def classify(self, inputs):
        return self.sigmoid(
                self._weights.dot(inputs) + self._biases
                )

    def descend(self, inputs, actual, expected):
        def derivative_of_sigmoid(x):
            return sigmoid(x) * (1 - sigmoid(x))

        def gradient_vector_biases(actual, expected):
            return (2 * (actual - expected)).dot(actual).dot(1-actual)

        def gradient_vector_weights(inputs, actual, expected):
            return [ layer for layer in weights ].dot().dot()

        def shift_weights():
            pass

        pass


def train():
    pass
