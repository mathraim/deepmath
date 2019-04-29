import numpy as np
from deepmath.base_classes.optimizer import Optimizer


class Adam(Optimizer):
    """
    Child class of Optimizer - represents Adam optimization method

    Attributes
    ----------
    m_weights, v_weights, m_biases, v_biases : numpy array(matrix)
        helper variables that helps to identify the weight and biases update
    curr_iter : int
        current iteration of the algorithm
    """
    momentum1 = 0.9
    momentum2 = 0.99

    def help_init(self, input_dims, nodes):
        """
        Initializes m_weights, v_weights, m_biases, v_biases
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
        self.v_weights = np.zeros((input_dims, nodes))
        self.m_biases = np.zeros(nodes)
        self.v_biases = np.zeros(nodes)
        self.curr_iter = 0

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

        self.m_weights = Adam.momentum1 * self.m_weights + (1 - Adam.momentum1) * grad_weights
        self.v_weights = Adam.momentum2 * self.v_weights + (1 - Adam.momentum2) * np.power(grad_weights, 2)
        self.m_biases = Adam.momentum1 * self.m_biases + (1 - Adam.momentum1) * grad_biases
        self.v_biases = Adam.momentum2 * self.v_biases + (1 - Adam.momentum2) * np.power(grad_biases, 2)

        self.curr_iter += 1
        self.m_weights = self.m_weights / (1 - np.power(Adam.momentum1, self.curr_iter))
        self.v_weights = self.v_weights / (1 - np.power(Adam.momentum2, self.curr_iter))
        self.m_biases = self.m_biases / (1 - np.power(Adam.momentum1, self.curr_iter))
        self.v_biases = self.v_biases / (1 - np.power(Adam.momentum2, self.curr_iter))

        weights_step = np.multiply((learning_rate / (np.sqrt(self.v_weights) + eps)), self.m_weights)
        biases_step = np.multiply((learning_rate / (np.sqrt(self.v_biases) + eps)), self.m_biases)

        return weights_step, biases_step