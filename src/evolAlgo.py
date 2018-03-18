from deep_nn import DeepNN
from demo_deep_nn import demo
import Evolver as ev
import test
import numpy as np

F_ARGS = 2
NETWORK_SIZES = (2, 3, 3, 1)
TRAINING_N = 100000
TEST_N = 1000
EPOCHS = 20
BATCH_SIZE = 20
ETA = 1.5
SURVIVAL_RATE = 0.25
NB_GENERATIONS = 20


def f(x, y):
    """ Some arbitrary function f: [0, 1]*[0, 1] -> [0, 1] """
    return x * y * (1 - x - y + x * y) ** 2 * 45.5625

def generate_data(n):
    """ Generate n pairs x, y to be used as training and test data """
    return [(x, f(*x)) for x in np.random.random([n, F_ARGS])]

def evolution_algo():
    population = 100
    training_data = generate_data(TRAINING_N)
    test_data = generate_data(TEST_N)
    population_array = ev.firstGeneration(population, NETWORK_SIZES, test_data, len(test_data))
    for i in range (NB_GENERATIONS):
        x = ev.nextGeneration(population, SURVIVAL_RATE, population_array, NETWORK_SIZES, test_data, len(test_data))
    return x[0]
        
def main():
    print(evolution_algo().score)

##e1 = ev.Evolver(NETWORK_SIZES, test_data, len(test_data))
##e2 = ev.Evolver(NETWORK_SIZES, test_data, len(test_data))
##e3 = ev.Evolver(NETWORK_SIZES, test_data, len(test_data))
##e4 = ev.Evolver(NETWORK_SIZES, test_data, len(test_data))
##e5 = ev.Evolver(NETWORK_SIZES, test_data, len(test_data))
##
##l = [e1,e2,e3,e4,e5]

##l.sort(key=lambda x: x.score)
if __name__ == '__main__':
    main()
