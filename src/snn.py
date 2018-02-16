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
    
    def classify(inputs):
        pass 

    def sigmoid(x):
        y = 1 / (1 + np.exp(-x))
        return y

def train():
    pass