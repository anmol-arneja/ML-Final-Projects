__author__ = 'Admin'

import numpy as np
import csv
import random
from sklearn import decomposition
from sklearn import preprocessing
from array import *

def clean_data():
    #-----------------------------------Loading the Training Data-------------------------------------------------------
    print "Loading Training Input"
    train_input = np.load("train_inputs.npy")
    train_output = np.load("train_outputs.npy")
    print "Done with loading of Training Data Set"
    #----------------------------------Loading the Validation Data set--------------------------------------------------
    valid_input = np.load("valid_inputs.npy")
    valid_output = np.load("valid_outputs.npy")
    #----------------------------------Loading the Test Data Set--------------------------------------------------------
    test_input = np.load("test_inputs.npy")
    #----------------------------------------Standardizing the Training set---------------------------------------------
    print "Standardizing the features for Training Set"
    std = preprocessing.StandardScaler().fit(train_input)
    std_train_inputs = std.transform(train_input)
    np.save("standardized_train_inputs",std_train_inputs)
    print"Done with Standardizing the features of Training Set"
    #----------------------------------------Standardizing the Validation Set-------------------------------------------
    print "Standardizing the feature of Validation Set"
    std_valid_inputs = std.transform(valid_input)
    np.save("standardized_valid_inputs",std_valid_inputs)
    print "Done with Standardizing the features of Validation Set"
    #-----------------------------------------Standardizing the Test Set------------------------------------------------
    print "Standardizing the features of Test Set"
    std_test_inputs = std.transform(test_input)
    np.save("standardized_test_inputs",std_test_inputs)
    print "Done with standardizing features of Test Set"
    #-------------------------Getting PCA features for Training Inputs--------------------------------------------------
    print "Obtaining PCA features for Training Inputs"
    pca = decomposition.PCA()
    pca.fit(std_train_inputs)
    pca_train = pca.transform(std_train_inputs)
    np.save("pca_train_inputs",pca_train)
    print "Done with PCA on Training Data"
    #---------------------------Getting PCA features for Validation Inputs----------------------------------------------
    print"Obtaining PCA features for Validation Inputs"
    pca_validation = pca.transform(std_valid_inputs)
    np.save("pca_valid_inputs",pca_validation)
    print "Done with PCA on Validation Data"
    #-----------------------------Getting PCA features for Testing Inputs----------------------------------------------
    print "Obtaining PCA features for Testing Inputs"
    pca_test = pca.transform(std_test_inputs)
    np.save("pca_test_inputs",pca_test)
    print "Done with PCA on Test Data"
clean_data()





