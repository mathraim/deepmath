import numpy as np
from deepmath.base_classes.optimizer import Optimizer


class RMSprop(Optimizer):
    """
    Child class of Optimizer - represents RMSprop optimization method

    Attributes
    ----------
    v_weights, v_biases : numpy array(matrix)
        helper variables that helps to identify the weight and biases update
    """
    momentum = 0.99

    def help_init(self, input_dims, nodes):
        """
        helper mutator and for different optimizers sets different attributes
        as different optimizers have different attributes
        needs the dimentions of the weights and biases for the child attributes

        Parameters
        ----------
        input_dims : tuple of ints
            input_dims of the layer that this optimizer is attached to
            used for help_init method
        nodes : int
            number of nodes of the layer that this optimizer is attached to
            used for help_init method
        """
        self.v_weights = np.zeros((input_dims, nodes))
        self.v_biases = np.zeros(nodes)

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
        eps = 1e-8
        self.v_weights = RMSprop.momentum * self.v_weights + (1 - RMSprop.momentum) * np.power(grad_weights, 2)
        self.v_biases = RMSprop.momentum * self.v_biases + (1 - RMSprop.momentum) * np.power(grad_biases, 2)

        weights_step = np.multiply((learning_rate / np.sqrt(self.v_weights + eps)), grad_weights)
        biases_step = np.multiply((learning_rate / np.sqrt(self.v_biases + eps)), grad_biases)

        return weights_step, biases_step
