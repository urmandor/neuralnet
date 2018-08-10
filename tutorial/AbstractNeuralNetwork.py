import numpy as np
import matplotlib.pyplot as plt
import h5py
import copy
# from testCases_v4 import *
# from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

from abc import ABC, abstractmethod

class AbstractNeuralNetwork(ABC):

    def __init__(self):
        super().__init__()
        self.trainX, self.trainY = None, None 
        self.testX, self.testY = None, None
        self.classes = None
        self.activations = None
        self.layerdims = None

    def initializeParams(self, layerdims):
        """
            Arguments: 
            layerdims -- list containing the dimensions of each layer of the neural network. 

            returns: 
            parameters - dictionary containing all the 'Weights' & 'biases' for each layer.
        """

        parameters = {}
        L = len(layerdims)
        for l in range(1, L):
            parameters["W" + str(l)] = np.random.randn(layerdims[l], layerdims[l-1]) * 0.01
            parameters["b" + str(l)] = np.zeros((layerdims[l], 1))

        return parameters

    # FORWARD PROPAGATION START ...
    def forwardLinear(self, A, W, b):
        """
            Equation: Z = WA+b

            Arguments:
            A -- previous layer's activation function output or input
            W -- Weights of current layer
            b -- bias of current layer

            returns:
            Z -- Linear part of each layer in forward propagation.
        """

        Z = np.dot(W, A) + b
        return Z
    
    def forwardActivation(self, A_prev, W, b, activation_type):
        """
            Implements the forward propagation for neural networks.

            Arguments:
            A_prev -- previous layer's activation function output or input
            W -- Weights of current layer
            b -- bias of current layer
            activation_type -- (String) mentioning the type of activation function to be used

            returns:
            cache -- A cache which caches the important params such as A_prev, A, Z, W and b to be used again in back propagation
        """

        Z = self.forwardLinear(A_prev, W, b) 
        activation = self.findActivationImplementation(activation_type)
        A = activation(Z)

        cache = (A_prev, A, Z, W, b)

        return A, cache

    @abstractmethod
    def findActivationImplementation(self, activation_type):
        """
            Arguments:
            activation_type -- (String) mentioning the type of activation function to be used

            returns:
            function -- A function containing the implementation details of the activation function
        """
        pass

    def forwardPropagation(self, X, parameters, activations):
        """
            Implements the whole forward propagation algorithm.

            Arguments:
            X -- input to the neural network.
            parameters -- parameters dictionary containing all the weights and biases.
            activations -- array containing activation functions to be used in each layer. 
                           only the starting activation function should be listed.
                           e.g. for a setup like, input => N x RelUs => 1 x sigmoid => output
                           the array should be like this, [(N, 'relu'), (N+1, 'sigmoid')]

            returns:
            yHat -- final output value  
            caches -- caches array containing all the intermediate Z's, A's, and W's
        """

        cashes = []
        
        A = X
        L = len(parameters) // 2
        max_activation_index, activation_name = activations.pop(0)

        for l in range(1, L+1):
            
            if l > max_activation_index and len(activations) > 0:
                max_activation_index, activation_name = activations.pop(0)

            W = parameters["W" + str(l)]
            b = parameters["b" + str(l)]
            A, cache = self.forwardActivation(A, W, b, activation_name)
            cashes.append(cache)

        return A, cashes

    # FORWARD PROPAGATION END ...

    def calculateCost(self, yHat, Y):
        """
            Calculates the cost/error 

            Arguments:
            yHat -- predicted output from forward propagation
            Y -- desired output

            returns:
            cost -- cross-entropy cost
        """

        m = Y.shape[1]
        cost = np.divide(np.dot(Y, np.log(yHat).T) + np.dot((1-Y), np.log(1-yHat).T), -m)
        cost = np.squeeze(cost)
        return cost

    # BACK PROPAGATION START ...

    def backwardLinear(self, dA, Z, activation_type):
        """
            Implements the dZ part of backward propagation

            Arguments:
            dA -- Derivative of the activation function / cost function
            cache -- cache stored in the forward propagation part
            activation_type -- activation_type -- (String) mentioning the type of activation function to be used

            returns:
            dZ -- Derivation of the linear part of the neural network
        """
    
        activation_function = self.findDerivativeOfActivationImplementation(activation_type)
        dZ = activation_function(dA, Z)

        return dZ

    @abstractmethod
    def findDerivativeOfActivationImplementation(self, activation_type):
        """
            Arguments:
            activation_type -- (String) mentioning the type of activation function to be used

            returns:
            function -- A function containing the implementation details of the derivative of the activation function
        """
        pass
    
    def backwardActivation(self, dZ, cache):
        """
            Arguments:
            dZ -- derivation of the Z (linear part)
            cache -- contains (A_prev, A, Z, W, b)

            returns:
            dA_prev -- derivative of A of previous layer
            dW -- derivative of W of current layer
            db -- derivative of b of current layer 
        """
        A_prev, A, Z, W, b = cache
        m = A_prev.shape[1]
        dW = np.divide(np.dot(dZ, A_prev.T), m)
        db = np.divide(np.sum(dZ, keepdims=True, axis=1), m)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def backwardPropagation(self, yHat, Y, caches, activations):
        """
            Implements back propagation

            Arguments:
            yHat -- predicted output from forward propagation
            Y -- desired output
            caches -- array containing (A_prev, A, Z, W, b) from forward propagation

            returns:
            gradients -- A dictionary containing all the gradient => dA, dW, db
        """

        gradients = {}
        L = len(caches)
        Y = Y.reshape(yHat.shape)

        dAL = - (np.divide(Y, yHat) - np.divide(1-Y, 1-yHat))
        activations.reverse()
        max_activation_index, activation_name = activations.pop(0)
        dA = dAL
        
        for l in reversed(range(1, L+1)):
            
            if l < max_activation_index and len(activations) > 0:
                max_activation_index, activation_name = activations.pop(0)

            cache = caches[l-1]
            A_prev, A, Z, W, b = cache
            
            dZ = self.backwardLinear(dA, Z, activation_name)
            dA, dW, db = self.backwardActivation(dZ, cache)

            gradients["dA" + str(l - 1)] = dA
            gradients["dW" + str(l)] = dW
            gradients["db" + str(l)] = db

        return gradients

    # BACK PROPAGATION END ...

    def updateParams(self, parameters, gradients, learningRate):
        """
            Updates the parameters, Ws and bs

            Arguments:
            parameters -- dictionary of parameters containing weights and biases
            gradients -- dictionary of gradients calculated from back propagation
            learningRate -- learning rate alpha for gradient descent

            returns:
            parameters -- updated parameters
        """

        L = len(parameters) // 2

        for l in range(1, L+1):
            parameters["W" + str(l)] = parameters["W" + str(l)] - np.multiply(learningRate, gradients["dW" + str(l)])
            parameters["b" + str(l)] = parameters["b" + str(l)] - np.multiply(learningRate, gradients["db" + str(l)])

        return parameters
    
    
    def startTraining(self, X, Y, layerdims=None, activations=None, learningRate=0.0075, numIterations=3000, printCost=False, printInterval=100):
        
        costs = []

        layerdims = layerdims or self.layerdims
        activations = activations or self.activations

        parameters = self.initializeParams(layerdims)

        for i in range(0, numIterations):
            yHat, caches = self.forwardPropagation(X, parameters, copy.deepcopy(activations))

            cost = self.calculateCost(yHat, Y)

            gradients = self.backwardPropagation(yHat, Y, caches, copy.deepcopy(activations))
            parameters = self.updateParams(parameters, gradients, learningRate)

            if printCost and i % printInterval == 0:
                costs.append(cost)
                print("Cost after " + str(i) + " iterations : " + str(cost))
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations')
        plt.title('learning rate = ' + str(learningRate))
        plt.show()

        return parameters
    
    def loadData(self, trainingSrc, testSrc=None):

        classes, test_set_x_orig, test_set_y_orig = None, None, None

        train_dataset = h5py.File(trainingSrc, "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

        if testSrc:
            test_dataset = h5py.File(testSrc, "r")
            test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
            test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

            classes = np.array(test_dataset["list_classes"][:]) # the list of classes
            test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
        
        self.trainX = train_set_x_orig
        self.trainY = train_set_y_orig
        self.testX = test_set_x_orig
        self.testY = test_set_y_orig
        self.classes = classes

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

    def setLayers(self, layerdims):
        self.layerdims = layerdims

    def exploreData(self):
        # Explore your dataset 
        m_train = self.trainX.shape[0]
        num_px = self.trainX.shape[1]
        m_test = self.testX and self.testX.shape[0]

        print ("Number of training examples: " + str(m_train))
        print ("Number of testing examples: " + str(m_test))
        print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
        print ("train_x_orig shape: " + str(self.trainX.shape))
        print ("train_y shape: " + str(self.trainY.shape))
        print ("test_x_orig shape: " + str(self.testX and self.testX.shape))
        print ("test_y shape: " + str(self.testX and self.testY.shape))

    def flattenData(self):
        self.trainX = self.trainX.reshape(self.trainX.shape[0], -1).T
        self.testX = self.testX and self.trainX.reshape(self.trainX.shape[0], -1).T

    def normalizeData(self, maxValue=255):
        self.trainX = self.trainX/maxValue
        self.testX = self.testX and self.testX/maxValue


    def testNetwork(self):
        np.random.seed(1)
        return self.initializeParams([3,2,1])