import numpy as np
from deepmath.base_classes.layer import Layer
from deepmath.base_classes.optimizer import Optimizer


class Dense(Layer):
    """
    child class  of Layer that represents the Dense layer in the network

    ...

    Attributes
    ----------
    previous : Layer object
        previous layer in the network's queue of layers
    next : Layer object
        next layer in the queue of layers in the network
    network : Network object
        network that contains this layer
    nodes : int
        number of nodes
    optimizer : Optimizer object
        optimizer that is going to be used for the neural network
    input_dims: tuple of int
        input dimentions
    weights: numpy array(matrix)
        Weights that is used to perform the forward propagation for dense layer
    biases: numpy array(matrix)
        Biases that is used to perform the forward propagation for dense layer

    Methods
    -------
    help_init(self)
        helper mutator method for the weights and biases
    set_optimizer(self, optimizer)
        mutator method that sets the optimizer
    forward(self, input)
        Performes forward propagation thought the layer
    backward(self, input, grad_output)
        Performs a backpropagation step through the layer
    """

    def __init__(self, nodes, input_dims=None):
        """
        Parameters
        ----------
        nodes : int
            number of nodes
        input_dims: tuple of int
            input dimentions
        """

        self.input_dims = input_dims
        self.nodes = nodes
        super().__init__()

    def help_init(self):
        """
        Mutator method for the weights and biases
        They are dependent from input dimentions and number nodes
        """

        self.weights = np.random.randn(self.input_dims, self.nodes) * 0.01
        self.biases = np.zeros(self.nodes)

    def set_optimizer(self, optimizer):
        """
        Mutator method that sets the optimizer

        Parameters
        ----------
        optimizer : Optimizer object
            optimizer that is going to be used for the neural network
        """

        self.optimizer = Optimizer(self.input_dims, self.nodes, optimizer)

    def forward(self, input):
        """
        Takes input to the layer and calculates the grad_output
        Uses weights and biases martices to calculate it
        output = forward(input)

        Parameters
        ----------
        input : numpy array(matrix)
            Value for the input to the layers

        Returns
        -------
        numpy array(matrix)
            output of the layer
        """
        output = input @ self.weights + self.biases
        return output

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer
        It uses chain rule to calaculate gradients:

        (d loss / d input) = (d loss / d output) * (d output / d input)

        (d loss / d weights) = (d loss / d output) * (d output / d weights)

        (d loss / d biases) = (d loss / d output) * (d output / d biases)

        By the logic of the network (d loss / d output) is given

        It updates weghts and biases using the optimizer attached to layer

        Parameters
        ----------
        grad_output : numpy array(matrix)
            (d loss / d output) to use it in the chain rule
        input : numpy array(matrix)
            input used to calaculate the output - nessesary for the gradients

        Returns
        -------
        numpy array(matrix)
            return the gradient of the input variable with respect to loss
            (d loss / d input)
            We need it because it is grar_output for previous layer
        """
        grad_input = grad_output @ self.weights.T

        grad_weights = input.T @ grad_output
        grad_biases = np.sum(grad_output, axis=0)

        (weights_step, biases_step) = \
            self.optimizer.get_steps(grad_weights,
                                     grad_biases,
                                     self.network.learning_rate)

        self.weights -= weights_step
        self.biases -= biases_step

        return grad_input
