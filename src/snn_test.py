#!/usr/bin/env python3

import unittest
from snn import SNN, GradientDescent
import numpy as np

class TestSNN(unittest.TestCase):

    def test_sigmoid(self):
        nn = SNN(3, 4)
        results = np.array(
                [nn.sigmoid(i) for i in range(-5, 6)]
                ).round(3)
        expected = np.array(
                [.007, .018, .047, .119, .269, .5
                    ,.731, .881, .953, .982, .993]
                )
        self.assertTrue((results==expected).all())

if __name__ == '__main__':
    unittest.main()
