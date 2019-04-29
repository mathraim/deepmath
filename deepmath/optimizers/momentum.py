import numpy as np
from deepmath.base_classes.optimizer import Optimizer


class Momentum(Optimizer):
    """
    Child class of Optimizer - represents Stochastic gradient decent with
    momentum

    Attributes
    ----------
    m_weights, m_biases : numpy array(matrix)
        helper variables that helps to identify the weight and biases update
    """
    momentum = 0.9

    def help_init(self, input_dims, nodes):
        """
        Initializes m_weights, m_biases
        according to weights and biases dimentions
        and sets the current iteration to 0

        Parameters
        ----------
        input_dims : tuple of ints
            input_dims of the layer that this optimizer is attached to
            used for help_init method
        nodes : int
            number of nodes of the layer that this optimizer is attached to
            used for help_init method
        """
        self.m_weights = np.zeros((input_dims, nodes))
        self.m_biases = np.zeros(nodes)

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
        self.m_weights = Momentum.momentum * self.m_weights + grad_weights
        self.m_biases = Momentum.momentum * self.m_biases + grad_biases

        weights_step = learning_rate * self.m_weights
        biases_step = learning_rate * self.m_biases

        return weights_step, biases_step
