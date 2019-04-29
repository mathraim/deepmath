import numpy as np
from deepmath.base_classes.optimizer import Optimizer


class SGD(Optimizer):
    """
    Child class of Optimizer
    represents Stochastic Gradient Descent optimization method
    """

    def get_steps(self, grad_weights, grad_biases, learning_rate):
        """
        Gets the current weights and biases gradients and returns one step for
        the update

        Parameters
        ----------
        grad_weights : numpy array(matrix)
            gradient of the corresponding layer weights (d loss / d weights)
        grad_biases : numpy array(matrix)
            gradient of the corresponding layer biases (d loss / d biasess)
        learning_rate : double
            learning rate
        """

        return learning_rate * grad_weights, learning_rate * grad_biases
