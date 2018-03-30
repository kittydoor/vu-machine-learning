from deep_nn import *
import numpy as np
from bla import *
training_zip, validation_zip, test_zip = load_data_wrapper()
training_data = list(training_zip)
validation_data = list(validation_zip)
test_data = list(test_zip)

dnn = DeepNN([784,49,10])
dnn.sgd(training_data, 20, 10, 2.0, test_data)
