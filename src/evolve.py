from deep_nn import DeepNN

def f(x):
    return x*x

class Evolver:
    def __init__(self, nn=DeepNN, training_data=f, test_data=f, population=20, keep=.5):
        """Evolver is a class which takes a network implementation,
        training and test data, and some additional parameters
        in order to generate a population from which to select
        top performers and mutate them
        
        nn -- NN implementation
        training_data -- function to pull training data from
        test_data -- function to pull test data from
        population -- population of a single generation
        keep -- percentage of the population kept at each selection
        """
        self._nn = nn
        self._data_train = training_data
        self._data_test = test_data
        self._population = population
        self._keep = keep

    @staticmethod
    def mutate(model, num):
        """Mutate should take a model, and return $num instances of it
        with one being the exact model, and the rest having small mutations

        model -- network data
        num -- number of models to create
        """
        return [model for i in range(num)]

    def first_generation():
        """Generates first generation of models
        """
        pass

    def next_generation():
        """Generates a new generation by testing and selecting from
        last generation, and then mutating to fill the gap
        """
        pass

def main():
    Evolver()

if __name__ == '__main__':
    main()
