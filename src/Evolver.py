from deep_nn import DeepNN
from demo_deep_nn import demo
import test
import numpy as np

class Evolver:
    def __init__(self, size, test_data, n_test):
        DNN= DeepNN(size)
        self.score = DNN.loss(test_data) / n_test
        

def firstGeneration(population, size, test_data, n_test):
    array_nn = []
    for i in range(population):
        array_nn.append (Evolver(size, test_data, n_test))
    return array_nn

def nextGeneration(population, survival_rate, population_array, size, test_data, n_test):
    population_array.sort(key=lambda x: x.score)
    new_population = population_array[0:int(survival_rate*population)]
    for i in range (int(population - survival_rate*100)):
        new_population.append(DeepNN(size))
    return new_population
        

if __name__ == '__main__':
    main()
