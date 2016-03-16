__author__ = 'Admin'

import numpy as np
import random
from basic_functions import *

def step(x):
    return np.sign(x)
#----------------------------------------------Implementation of Perceptron---------------------------------------------
class Perceptron:
    def __init__(self,alpha=0.1,activation=lambda x:step(x),num_iters=10):
        self.alpha = alpha
        self.activation = activation
        self.num_iters = num_iters

    #-------------------------------------------Training a Classifier---------------------------------------------------

    def train(self,train_inps,train_opts):
        num_train_inputs = train_inps.shape[0]
        num_features = train_inps.shape[1]
        num_outputs = train_opts.shape[1]

        #Initializing the weights
        self.w = np.random.uniform(-0.5,0.5,(num_features + 1,num_outputs))
        bias_train = np.ones((num_train_inputs,num_features + 1))
        bias_train[:,1:] = train_inps
        for i in range(self.num_iters):
            init_error = 0
            for x,y in zip(bias_train,train_opts):
                #get output and error
                out = self.activation(np.dot(self.w.T,x))
                error = y -out
                if np.sum(error) != 0:
                    init_error += 1
                    #Modify the weights
                    self.w[:,np.argmax(out)] -= self.alpha*x
                    self.w[:,np.argmax(y)] += self.alpha*x
            if (init_error < 1):
                break

    def predict(self,valid_inp):
        #Predicting output for examples in Validation set
        num_features = valid_inp.size
        bias_valid_inp = np.ones(num_features + 1)
        bias_valid_inp[1:] = valid_inp
        return np.argmax(self.activation(np.dot(self.w.T,bias_valid_inp)))

#---------------------------------Implementation of sigmoid function and its differentiation----------------------------
def sigmoid_func(x):
    return 1.0 / (1.0 + np.exp(-x))
sigmoid = np.vectorize(sigmoid_func)
def sigmoid_diff_func(x):
    return sigmoid(x) * (1-sigmoid(x))
sigmoid_diff = np.vectorize(sigmoid_diff_func)
#-----------------------------------------------Implementation of Neural Networks---------------------------------------
class NeuralNetworks():
    #----------------------------------------------Initializing the parameters------------------------------------------
    def __init__(self,layers,alpha=0.01,epochs=10):
        self.activation = sigmoid
        self.activation_diff = sigmoid_diff
        self.alpha = alpha
        self.epochs = epochs
        self.weights = []

        #Initialize the Weights
        for i in range(1,len(layers) - 1):
            w = 2*np.random.random((layers[i-1] + 1,layers[i] + 1)) - 1
            self.weights.append(w)
        w = 2*np.random.random((layers[1] + 1,layers[i+1])) - 1
        self.weights.append(w)
    #--------------------------------------------Training the Classifier------------------------------------------------
    def train(self,train_inps,train_opts):
        ones = np.atleast_2d(np.ones(train_inps.shape[0]))
        train_inps = np.concatenate((ones.T,train_inps),axis = 1)
        for epoch in range(self.epochs):
            for i,train in enumerate(train_inps):
                #---------------------------Performing feed-forward activation of layers--------------------------------
                a = [train]
                for w in range(len(self.weights)):
                    value = np.dot(a[w],self.weights[w])
                    activation = self.activation(value)
                    a.append(activation)
                #------------------------------Iterating backward through Neural Network--------------------------------
                error = train_opts[i] - a[-1]
                deltas = [error*self.activation_diff(a[-1])]
                for l in range(len(a) - 2,0,-1):
                    deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_diff(a[l]))
                deltas.reverse()
                #------------------------------Update the weights accordingly-------------------------------------------
                for i in range(len(self.weights)):
                    layer = np.atleast_2d(a[i])
                    delta = np.atleast_2d(deltas[i])
                    self.weights[i] += self.alpha * layer.T.dot(delta)
            print "Finished epoch %d" %(epoch + 1)
    #-----------------------------------------Predict the test output---------------------------------------------------
    def predict(self,test_opts):
        a = np.concatenate((np.ones(1).T,np.array(test_opts)))
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return np.argmax(a)






