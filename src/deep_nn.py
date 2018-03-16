import numpy as np

# Documentation soon...
# Next steps: - Implement concurrency for speed-up
# - Functionality for CNN?

class DeepNN:

    def __init__(self, sizes):
        """DeepNN is a deep neural network implementation with infinitely scalable
        network depth, which handles the low level linear algebra behind
        the network calculations.

        sizes -- list of ints, defines network topology
        """
        self.sizes = sizes
        
        # network weights, defined as random
        self.w = [np.random.randn(i, j) for i, j in zip(sizes[1:], sizes[:-1])]

        # network biases, defined as random
        self.b = [np.random.randn(k) for k in sizes[1:]]

    @staticmethod
    def sigmoid(z):
        """A sigmoid function is used to shape the neuron outputs into a desired curve

        z -- input for the function
        """
        return 1 / (1 + np.exp(-z))


    def feedforward(self, a):
        """Feedforward takes an input matrix and feeds it through the network
        to get a resulting matrix of the size of the output layer
        
        a -- data matrix in the size of the input layer
        """
        for w, b in zip(self.w, self.b):
            a = DeepNN.sigmoid(w.dot(a) + b)
        return a

    def sgd(self, training_data, epochs, batch_size, eta, test_data=None):
        """Stochastic Gradient Descent, also known as incremental gradient descent,
        is an implementation in which multiple iterations of training are done,
        where every iteration consists of one batch which is evaluated all
        at once.

        training_data -- function to pull training data from
        epochs -- amount of times to consume entire training data
        batch_size -- amount of data to produce in one batch
        eta -- learning rate
        test_data -- function to pull test data from
        """
        n_training = len(training_data)
        if test_data:
            n_test = len(test_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            batches = [training_data[i:(i + batch_size)] \
                for i in range(0, n_training, batch_size)]
            for batch in batches:
                self.process_batch(batch, eta)
            if test_data:
                print("Epoch %d finished. Loss: %.6f" \
                    % (epoch, self.loss(test_data) / n_test))

    def process_batch(self, batch, eta):
        """Process batch takes a batch of data, and uses gradient descent in order
        to learn from the data and improve the weights and biases
        
        batch -- list of input and desired output tuples
        eta -- learning rate
        """
        gradient_w = [np.zeros(w.shape) for w in self.w]
        gradient_b = [np.zeros(b.shape) for b in self.b]
        for x, y in batch:
            d_gradient_w, d_gradient_b = self.backpropagate(x, y)
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, d_gradient_w)]
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, d_gradient_b)]
        self.w = [w - eta * dw / len(batch) for w, dw in zip(self.w, gradient_w)]
        self.b = [b - eta * db / len(batch) for b, db in zip(self.b, gradient_b)]

    def backpropagate(self, x, y):
        """Backpropogate takes an input matrix of the size of the input layer,
        and a desired output matrix, and calculates weight shifts necessary
        to minimize loss.

        x -- input matrix
        y -- output matrix
        """
        activations = [x]
        for w, b in zip(self.w, self.b):
            activations = [*activations,  DeepNN.sigmoid(w.dot(activations[-1]) + b)]

        delta = [(activations[-1] - y) * activations[-1] * (1 - activations[-1])]
        for w, a in zip(reversed(self.w), reversed(activations[:-1])):
            delta = [np.transpose(w).dot(delta[0]) * a * (1 - a), *delta]
        dw = [np.outer(d, a) for d, a in zip(delta[1:], activations[:-1])]
        return dw, delta[1:]

    def loss(self, test_data):
        """Loss function returns the error rate of the network
        given some test data
        """
        return sum(sum((self.feedforward(x) - y) ** 2) for (x, y) in test_data)
