class Optimizer():
    """
    Helper class that represents optimization method for the each dense layer
    that updates the weight and biases. Each of the layer will have their
    own optimizer that gives opportunity to train different layers
    by different optimization methods. It is a "abstract" parent class
    for each of the different optimization techiniques -
    - Adam, RMSprop, Mometum, and basic SGD

    Metohds
    -------
    help_init(self, input_dims, nodes)
        helper mutator and for different optimizers sets different attributes
        as different optimizers have different attributes
    get_steps(grad_weights, grad_biases, learning_rate)
        Gets the current weights and biases gradients and returns one step for
        the update
    """

    def __init__(self, input_dims, nodes, optimizer):
        """
        First initializes the parent Optimizer class and then depending from
        the parameters it switches to the proper child class

        Parameters
        ----------
        input_dims : tuple of ints
            input_dims of the layer that this optimizer is attached to
            used for help_init method
        nodes : int
            number of nodes of the layer that this optimizer is attached to
            used for help_init method
        optimizer : str
            name of the optimizer child to switch it to this class
        """

        from optimizers import adam, rmsprop, momentum, sgd
        dict_optimizers = {
            'adam': adam.Adam,
            'rmsprop': rmsprop.RMSprop,
            'momentum': momentum.Momentum,
            'sgd': sgd.SGD,
        }
        self.__class__ = dict_optimizers.get(optimizer)
        self.help_init(input_dims, nodes)

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

        pass

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

        pass