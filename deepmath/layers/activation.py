import numpy as np
from deepmath.base_classes.layer import Layer


class Activation(Layer):
    """
    Child "abstarct" class  of Layer that represents the Activation layer
    Also parent class for the classes - Sigmoid, Tanh, Relu, LRelu
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

    def __init__(self, activ_func='sigmoid'):
        """
        It initilaizes Activation layer first and then switches the object
        to corresponding child class - Sigmoid, Tanh, Relu, LRelu

        Parameters
        ----------
        activ_func : str
            type of the activation function.
        """

        from activation_functions import sigmoid,tanh,relu,lrelu
        switcher_for_activ = {
            'sigmoid': sigmoid.Sigmoid,
            'tanh': tanh.Tanh,
            'relu': relu.Relu,
            'lrelu': lrelu.LRelu,
        }
        super().__init__()
        self.__class__ = switcher_for_activ.get(activ_func)

    @staticmethod
    def softmax_for(input):
        """
        Softmax fucntion for the network class
        Softmax layer was very unstable because of the vanishing gradients
        """

        z = input - np.array([np.max(input, axis=1)]).T
        exponent = np.exp(z)
        sums = np.array([np.sum(exponent, axis=1)]).T
        return (exponent) / (sums)

    @staticmethod
    def softmax_back(input, grad_output):
        """
        gives (d loss / d input) for the softmax layer but for now it is here

        Parameters
        ----------
        grad_output : numpy array(matrix)
            (d loss / d output) to use it in the chain rule
        input : numpy array(matrix)
            input used to calaculate the output - nessesary for the gradients
        """

        soft = Activation.softmax_for(input)
        help_array = np.array([np.diag(soft @ grad_output.T)]).T
        return np.multiply(grad_output - help_array, soft)

    def help_init(self):
        """
        Mutator method for the number of nodes that is needed for other methods
        """

        self.nodes = self.previous.nodes

    def get_func_grad(self, input):
        """
        Helper method to calculate
        (d output / d input) that is dependent from input value

        Parameters
        ----------
        input : numpy array(matrix)
        """

        pass

    def backward(self, input, grad_output):
        """
        Performs a backpropagation step through the layer
        It uses chain rule to calaculate gradients:

        (d loss / d input) = (d loss / d output) * (d output / d input)

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

        return grad_output * self.get_func_grad(input)
