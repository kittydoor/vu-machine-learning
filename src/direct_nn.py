#!/usr/bin/env python
import numpy as np

class DirectNN:

    def __init__(self, in_layer_n, out_layer_n, eval_fn=None):
        """Initializes a shallow neural network 
        with specified number of input and output layers

        in_layer_n -- the size of the input vector
        out_layer_n -- the size of the output vector
        eval_fn
        """
        self._in_layer_n = in_layer_n
        self._out_layer_n = out_layer_n
        self._weights = np.random.randn(
                self._out_layer_n,
                self._in_layer_n
                )
        self._biases = np.random.randn(self._out_layer_n)
        self.evaluate = eval_fn

    @staticmethod
    def read_data(filename):
        """Returns a list of tuples by reading a file
        Tuple: (inputs, outputs)
        inputs: A vector of size _in_layer_n
        outputs: A vector of size _out_layer_n bound within 1 and -1,
        generated from the label

        filename -- full path of file to read
        """
        pass
        # shuffle data?
        return [ (np.random.randn(self._in_layer_n), np.random.randn(self._out_layer_n)) for i in range(5) ]
    
    @staticmethod
    def sigmoid(x):
        """Returns a number bound within 1 and -1

        x -- function input
        """
        y = 1 / (1 + np.exp(-x))
        return y

    def feedforward(self, inputs):
        """Feedforward inputs and return output vector

        inputs -- A vector of size _in_layer_n
        """
        return self.sigmoid(
                self._weights.dot(inputs) + self._biases
                )

    def backpropogate(self, x, y):
        """Computes the partial derivatives of the cost function with respect
        to the weights and biases

        x -- inputs vector
        y -- desired output vector
        """
        a = self.feedforward(x)
        db = (a - y) * a * (1 - a)
        dw = np.outer(db, x)
        return(dw, db)

    def gradient_descent(self, training_data, epochs, batch_size, eta, test_data=None):
        """Apply the gradient descent algorithm using backpropogation

        training_data -- A list of inputs and labels
        epochs -- Number of times to fully consume training data
        batch_size -- Number of rows to train with in a single calculation
        eta -- Learning rate, positive real

        Keyword arguments:
        test_data -- Progress report on training per epoch (default None)
        """
        # shuffle data?
        pass
        for epoch in range(epochs):
            #split data into batches
            batches = [ data[i:i+batch_size] for i in range(0, len(training_data), batch_size) ]
            for batch in batches:
                self.process_batch(batch, eta) 

            self.evaluate(test_data)
            #if test_data:
            #    print(evaluate(test_data))

    def process_batch(self, batch, eta):
        """Backpropogates over all tuples in the batch,
        sums the gradient and shifts network weights and biases

        batch -- A list of inputs and labels, which is a subset of the entire data
        eta -- Learning rate, positive real
        """
        batch_dw = np.zeros(self._in_layer_n, self._out_layer_n)
        batch_db = np.zeros(self._out_layer_n)

        for inputs, outputs in batch:
            dw, db = backpropogate(inputs, outputs)
            batch_dw += dw
            batch_db += db

        self._weights += batch_dw
        self._biases += batch_db

    @staticmethod
    def mean_square_error():
        pass

#    def evaluate(self, test_data, fn=None):
#        """Feeds forward the inputs in given test data and compares with their labels,
#        in order to return the number of successful guesses.
#
#        test_data -- A list of inputs and matching labels
#        fn -- Evaluation function
#        """
#        pass
#
#    def gradient_vector_biases(actual, expected):
#        return (2 * (actual - expected)).dot(actual).dot(1-actual)
#
#    def gradient_vector_weights(inputs, actual, expected):
#        return [ layer for layer in weights ].dot().dot()
#
#    def derivative_of_sigmoid(x):
#        return sigmoid(x) * (1 - sigmoid(x))
