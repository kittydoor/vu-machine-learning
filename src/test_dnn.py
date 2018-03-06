import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from dnn import *

F_ARGS = 2
NETWORK_SIZES = (2, 3, 3, 1)
TRAINING_N = 100000
TEST_N = 1000
EPOCHS = 20
BATCH_SIZE = 20
ETA = 1.5

def f(x, y):
    """ Some arbitrary function f: [0, 1]*[0, 1] -> [0, 1] """
    return x * y * (1 - x - y + x * y) ** 2 * 45.5625

def generate_data(n):
    """ Generate n pairs x, y to be used as training and test data """
    return [(x, f(*x)) for x in np.random.random([n, F_ARGS])]

network = DNN(NETWORK_SIZES)
training_data = generate_data(TRAINING_N)
test_data = generate_data(TEST_N)
network.sgd(training_data, EPOCHS, BATCH_SIZE, ETA, test_data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xs = np.arange(0, 1, 0.05)
ys = np.arange(0, 1, 0.05)
x, y = np.meshgrid(xs, ys)
a = np.reshape([network.feedforward(np.array([x,y])) for x in xs for y in ys], [20,20])
z = f(x, y)
surf = ax.plot_surface(x, y, z, color = (1, 0, 0, 0.5))
surf2 = ax.plot_surface(x, y, a, color = (0, 0, 1, 0.5))
plt.show()
