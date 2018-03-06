import numpy as np

# Documentation soon...
# Next steps: - Implement concurrency for speed-up
# - Functionality for CNN?

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class DNN:

    def __init__(self, sizes):
        self.sizes = sizes
        self.w = [np.random.randn(i, j) for i, j in zip(sizes[1:], sizes[:-1])]
        self.b = [np.random.randn(k) for k in sizes[1:]]

    def feedforward(self, a):
        for w, b in zip(self.w, self.b):
            a = sigmoid(w.dot(a) + b)
        return a

    def sgd(self, training_data, epochs, batch_size, eta, test_data = None):
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
        gradient_w = [np.zeros(w.shape) for w in self.w]
        gradient_b = [np.zeros(b.shape) for b in self.b]
        for x, y in batch:
            d_gradient_w, d_gradient_b = self.backpropagate(x, y)
            gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, d_gradient_w)]
            gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, d_gradient_b)]
        self.w = [w - eta * dw / len(batch) for w, dw in zip(self.w, gradient_w)]
        self.b = [b - eta * db / len(batch) for b, db in zip(self.b, gradient_b)]

    def backpropagate(self, x, y):
        activations = [x]
        for w, b in zip(self.w, self.b):
            activations = [*activations,  sigmoid(w.dot(activations[-1]) + b)]

        delta = [(activations[-1] - y) * activations[-1] * (1 - activations[-1])]
        for w, a in zip(reversed(self.w), reversed(activations[:-1])):
            delta = [np.transpose(w).dot(delta[0]) * a * (1 - a), *delta]
        dw = [np.outer(d, a) for d, a in zip(delta[1:], activations[:-1])]
        return dw, delta[1:]


    def loss(self, test_data):
        return sum(sum((self.feedforward(x) - y) ** 2) for (x, y) in test_data)
