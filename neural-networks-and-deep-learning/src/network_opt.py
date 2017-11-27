"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.

Author: Wei Song
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

"""
Keep in mind that this class is only used for computing gradident by
only one sample, and forward propagation is the same, only for one sample.
Because this is the start of the lecture, so it is easy to start with only
one sample.

But if you want to improve the performance and leverage the modern matrix library
then you need to change the code to vectorized it.
"""

class NetworkOpt(object):
    """
    This class is the same functional as class Network in network.py, but this class
    will use the advantages of modern matrix library numpy to vectorize the SGD.

    SGD Vectorized to speedup.

    weight that connect layer l and layer l + 1 will have a different representation from
    the tutorial, in tutorail the weight is (l + 1) * l, then for example, input vector is V * 1,
    and first hidden layer size is H, then weight connect input layer and first hidden layer will be
    H * V, so we use the compute: H * v * (V * 1) = H * 1 to get hidden value.

    But in this class, in order to vectorize it, weight connect l and l + 1 layer will be l * (l + 1)
    this is becasue the input will be a batch of training data, assume we have n training data, then the 
    input vector is n * V, and the weight connect first layer and first hidden layer is V * H,
    then we compute n * V * (V * H) = v * H to compute the hidden vectors of all the input training data 
    together.
    """

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    # TODO: update
    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        """ a should be a vector """
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        """
        Although this is called SGD, and it did compute as SGD way, but
        the speed is very slow because it compute the gradient of samples
        one by one, and then sum up the gradient of each sample as total 
        gradient of mini batch, this way could not leverage the advantage of 
        modern Matrix library.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            # delta_nabla_w & delta_nabla_b record the gradient change for 
            # each weight matrix of one sample
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        # gradient descent update
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""

        """
        ref link: http://www.jianshu.com/p/c69cd43c537a
        you could visit this link to get a better understand of how the
        network update its parameters.
        I will give a tutorial on network parameter update in future,
        you could pay attention to my blog: http://blog.csdn.net/watkinsong
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations - y)


#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    """
    d sigmoid(x)
    ------------ = sigmoid(x) * (1 - sigmoid(x))
    d x
    """
    return sigmoid(z) * (1-sigmoid(z))


if __name__ == '__main__':
    sizes = [784, 100, 10]
    learning_rate = 0.1
    epochs = 30
    mini_batch_size = 64

    import mnist_loader
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # define a network
    net = Network(sizes)

    # learn the network by SGD
    net.SGD(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data)