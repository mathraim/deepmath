Machine learnnning library build only on numpy for educational purposes

It will be my collection of machine learning algorithms implemented only numpy and set up in a package

The examples of using deepmath is represented in **test** folder.

Project structure:
* **base_classes** - folder for general parent classes for different classes that represent package
  * layer.py - Parent class that represents Layer class. Children's example - Convloutional, Dense, Activation
  * network.py - Class that represents a neural network. User can add layers to the network and then train it on a data
  * optimizer.py - Parent class that represents gradient decent variations such as Adam and RMSprop. Attachhed to evry layer class
* **layers** - folder containing children classes of a Layer class
  * Dense.py - Represent dense layer when every node of the current layer is connected to every node of a previous layer
  * activation.py - Represent activation function layer that applies some function to every node of a previsous class
  * **activation_functions** - folder containing children of the Activation class.
    * lrelu.py - class that represents Leaky Relu activation Layer
    * rely.py -  class that represents Relu activation Layer
    * sigmoid.py -  class that represents Sigmoid activation Layer
    * tanh.py - class that represents Tanh activation function
* **optimizers** - folder containing cihldren classes of a Optimizer class 
  * adam.py - class that represents ADAM optimization method
  * rmsprop.py - class that represents RMSprop optimization method
  * momentum.py - class that represents Gradient Decent with momentum
  * sgd.py - just usual Stochastic gradient decent
