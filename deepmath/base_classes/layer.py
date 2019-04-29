import numpy as np


class Layer(object):
    """
    An "abstract" class that represents the layer in the network

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
        helper mutator method for the child classes - Dense and Activation
    set_optimizer(self, optimizer)
        mutator method that sets the optimizer
    forward(self, input)
        Performes forward propagation thought the layer
    backward(self, input, grad_output)
        Performs a backpropagation step through the layer
    """

    def __init__(self):
        """
        Initializes the initial members for the Layer
        """

        self.previous = None
        self.network = None
        self.next = None
        self.node = None
        self.optimizer = None


    def help_init(self):
        """
        This mutator method for the child classes - Dense and Activation
        Dense - set weights and height
        Activation - set number of nodes
        """

        pass

    def set_optimizer(self, optimizer):
        """
        Mutator method that sets the optimizer

        Parameters
        ----------
        optimizer : Optimizer object
            optimizer that is going to be used for the neural network
        """

        pass

    def forward(self, input):
        """
        Takes input to the layer and calculates the grad_output
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

        return input

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer
        It uses chain rule to calaculate gradients

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
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input)  # chain rule
