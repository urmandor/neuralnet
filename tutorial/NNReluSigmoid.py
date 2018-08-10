import numpy as np
from AbstractNeuralNetwork import AbstractNeuralNetwork as ANN

class NNReluSigmoid(ANN):

    def __init__(self):
        super().__init__()

    def findActivationImplementation(self, activation_type):    
        if activation_type == "sigmoid":
            return self.sigmoid
        elif activation_type == "relu":
            return self.relu
        else:
            return self.relu
    
    def findDerivativeOfActivationImplementation(self, activation_type):
        if activation_type == "sigmoid":
            return self.backwardSigmoid
        elif activation_type == "relu":
            return self.backwardRelu
        else:
            return self.backwardRelu

    def sigmoid(self, Z):
        A = 1 / (1+np.exp(-Z))
        return A

    def relu(self, Z):
        A = np.maximum(0, Z)
        return A

    def backwardSigmoid(self, dA, Z):
        s = self.sigmoid(Z)
        dZ = dA * s * (1-s)
        return dZ

    def backwardRelu(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z<0] = 0
        return dZ

    def setDefaultActivation(self, activations=None):
        if activations:
            self.activations = activations
        else:
            size = len(self.layerdims)
            self.activations = [(size-2, 'relu'), (size-1, 'sigmoid')]