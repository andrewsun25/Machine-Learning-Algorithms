import unittest as ut
from neural_network import NeuralNetwork


class Test(ut.TestCase):

    nn = NeuralNetwork()

    def test_random_weights_forShape(self):
        expected_weights_shape = (3,)
        expected_weights0_shape = (25, 401)
        expected_weights1_shape = (25, 26)
        expected_weights2_shape = (10, 26)
        self.assertEqual(Test.nn.weights.shape, expected_weights_shape)
        self.assertEqual(Test.nn.weights[0].shape, expected_weights0_shape)
        self.assertEqual(Test.nn.weights[1].shape, expected_weights1_shape)
        self.assertEqual(Test.nn.weights[2].shape, expected_weights2_shape)

    def test_prop_forward_forActivationShape(self):
        a = Test.nn.prop_forward(1)
        expected_a0_shape = (401, 1)
        expected_a1_shape = (26, 1)
        expected_a2_shape = (26, 1)
        expected_a3_shape = (10, 1)
        self.assertEqual(a[0].shape, expected_a0_shape)
        self.assertEqual(a[1].shape, expected_a1_shape)
        self.assertEqual(a[2].shape, expected_a2_shape)
        self.assertEqual(a[3].shape, expected_a3_shape)

    def test_prop_backward_forSmallDeltaShape(self):
        d = Test.nn.prop_backward(1)
        a = Test.nn.prop_forward(1)
        expected_d1_shape = a[1].shape
        expected_d2_shape = a[2].shape
        expected_d3_shape = a[3].shape
        self.assertEqual(d[1].shape, expected_d1_shape)
        self.assertEqual(d[2].shape, expected_d2_shape)
        self.assertEqual(d[3].shape, expected_d3_shape)

if __name__ == '__main__':
    ut.main()