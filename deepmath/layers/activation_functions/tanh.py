import numpy as np
from deepmath.layers.activation import Activation


class Tanh(Activation):
    """
    Child class of a Activation layer
    represents Tanh Layer
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

    Methods
    -------
    help_init(self)
        helper mutator method for the weights and biases
    set_optimizer(self, optimizer)
        mutator method that sets the optimizer
    forward(self, input)
        Performes forward propagation thought the layer
    get_func_grad(self, input)
        Helper method to calculate (d output / d input)
    backward(self, input, grad_output)
        Performs a backpropagation step through the layer
    """

    def forward(self, input):
        """
        Takes input to the layer and calculates the grad_output
        output = tanh(input)

        Parameters
        ----------
        input : numpy array(matrix)
            Value for the input to the layers

        Returns
        -------
        numpy array(matrix)
            output of the layer
        """

        output = 1 - (2 / (np.exp(2 * input) + 1))
        return output

    def get_func_grad(self, input):
        """
        Helper method to calculate
        (d output / d input) that is dependent from input value and the type of
        activation

        Parameters
        ----------
        input : numpy array(matrix)
        """

        tanh = Activation.tanh_for(input)
        return 1 - np.power(tanh, 2)
