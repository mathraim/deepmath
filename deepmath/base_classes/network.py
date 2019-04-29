import numpy as np
from deepmath.layers.activation import Activation

class Network():
    """
    The main class that represents Network as a queue of layers
    we can train the model from here

    ...

    Attributes
    ----------
    input_layer : Layer object
        the first layer in the queue. Only this layer should specify the
    curr_layer : Layer object
        Helper method to go back and forth over the queue
    learning_rate : int
        learning rate
    optimizer : str
        name of optimization technique

    Methods
    -------
    hot_encode(y, num_classes)
        hot encodes y vector with given number of classes
    add(self, layer)
        Adds a new layer to the network
    forward(self, X)
        Performs the forward propagation over whole network
        returns the output of the a
    compile_model(self, learning_rate=0.01, optimizer='sgd')
        Sets the learning rate the optimization method
    get_logits(self, X)
        Does forward pass and returns the finak output of the network
    predict(self, X)
        Predicts the value according to the curent network
    cost_func_grad(self, logits, y)
        calculates the loss function and the gradient of the final output
        (d loss / d final_output)
    train_ones(self, X, y)
        takes the batch and trains the network once on that batch_size
        one iteration of the optimizers
    fit(self, X, y, batch_size=32, epoch=10)
        specifies the batch size and number of epochs to train the model
        on the dataset X and lables y
        trains the model and prints the accuracy with each epoch
    compute_accuracy(self, X, y)
        computes the accuracy for the dataset
    """

    @staticmethod
    def hot_encode(y, num_classes):
        """
        hot encodes y vector with given number of classes

        Parameters
        ----------
        """

        help_array = np.ones((y.shape[0], 1))
        y = (np.dot(help_array, np.array([range(num_classes)])) == np.array([y]).T)
        return y.astype(int)

    def __init__(self):
        """
        Initializes the network and the attributes
        """

        self.input_layer = None
        self.curr_layer = None
        self.learing_rate = 0
        self.optimizer = None

    def add(self, layer):
        """
        Adds a new layer to the Network

        Parameters
        ----------
        layer : Layer object
            layer to add
            only thhe first one should specify dimentions
        """

        layer.network = self
        if (self.input_layer is None):
            if (layer.input_dims is None):
                print( "Here will be an exeption in future" )
            self.input_layer = layer
            self.curr_layer = layer
            layer.previous = None
            layer.help_init()
        else:
            layer.previous = self.curr_layer
            self.curr_layer.next = layer
            self.curr_layer = layer
            layer.input_dims = layer.previous.nodes
            layer.next = None
            layer.help_init()

    def forward(self, X):
        """
        Performs the forward propagation over whole network
        returns the output of the whole network of layers

        Parameters
        ----------
        X : numpy array(matrix)
            input to the network

        Returns
        -------
        list of numpy array(matrix)
            List containing all of the outpus of all of the layers
        """

        activations = []
        input = X
        current = self.input_layer
        while (current is not None):
            input = current.forward(input)
            current = current.next
            activations.append(input)
        return activations

    def compile_model(self, learning_rate=0.01, optimizer='sgd'):
        """
        Sets the learning rate the optimization method

        Parameters
        ----------
        learning_rate : int
            learning rate
        optimizer : str
            name of optimization technique
        """

        self.learning_rate = learning_rate
        current = self.input_layer
        while (current is not None):
            current.set_optimizer(optimizer)
            current = current.next

    def get_logits(self, X):
        """
        Does forward pass and returns the finak output of the network

        Parameters
        ----------
        X : numpy array(matrix)
            input to the network

        Returns
        -------
        numpy array(matrix)
            The output of whole network
        """

        input = X
        current = self.input_layer
        while current is not None:
            input = current.forward(input)
            current = current.next
        return input

    def predict(self, X):
        """
        Predicts the value according to the curent network

        Parameters
        ----------
        X : numpy array(matrix)
            input to the network

        Returns
        -------
        numpy array(matrix)
            predicted lables of the rows of X
        """

        logits = self.forward(X)[-1]
        return logits.argmax(axis=-1)

    def cost_func_grad(self, logits, y):
        """
        calculates the loss function and the gradient of the final output
        (d loss / d final_output)

        Parameters
        ----------
        logits : numpy array(matrix)
            the final output of the network
        y : numpy array(matrix)
            label array for each row of input

        Returns
        -------
        numpy array(matrix)
            (d loss / d final_output) to use it for backprop
        """

        y = Network.hot_encode(y, logits.shape[1])
        probs = Activation.softmax_for(logits)
        cost_grad = probs - y
        return cost_grad / y.shape[0]

    def train_ones(self, X, y):
        """
        takes the batch and trains the network once on that batch_size
        one iteration of the optimizers

        Parameters
        ----------
        X : numpy array(matrix)
            batch for the network
        y : numpy array(matrix)
            label matricx for this array
        """
        layer_activations = self.forward(X)
        layer_inputs = [X] + layer_activations
        logits = layer_activations[-1]

        cost_grad = self.cost_func_grad(logits, y)

        curr_grad = cost_grad
        current_layer = self.curr_layer
        curr_input_index = -2
        while (current_layer is not None):
            curr_grad = current_layer.backward(layer_inputs[curr_input_index], curr_grad)
            current_layer = current_layer.previous
            curr_input_index -= 1

    def fit(self, X, y, batch_size=32, epoch=10):
        """
        specifies the batch size and number of epochs to train the model
        on the dataset X and lables y
        trains the model and prints the accuracy with each epoch

        Parameters
        ----------
        X : numpy array(matrix)
            whole input dataset
        y : numpy array(matrix)
            labels of the input
        batc_size : int
            size of the batch
        epoch : int
            number of epochs
        """

        m = X.shape[0]
        for i in range(epoch):
            print("epoch", i)
            shuffled_range = np.random.permutation(np.arange(m))
            start_idx = 0
            for start_idx in range(0, len(X) - batch_size + 1, batch_size):
                excerpt = shuffled_range[start_idx:start_idx + batch_size]
                self.train_ones(X[excerpt], y[excerpt])
            print('accuracy', self.compute_accuracy(X, y))
            # print('loss',self.compute_loss(X,y))
            print("---------------------------------")

    def compute_accuracy(self, X, y):
        """
        Calculates the accuracy of the current network
        """

        y_pred = self.predict(X)
        a = (y_pred == y)
        return np.sum(a) / y.shape[0]