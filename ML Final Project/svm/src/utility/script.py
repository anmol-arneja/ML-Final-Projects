import numpy as np
import pandas as pd
from numpy import loadtxt
from utility import *

# ---------------------------------------------- #
# EXRACTING PARKINSONS RESULT FROM RAW DATA 
# ---------------------------------------------- #

print("Obtaining Parkinsons Classification Results")
# Load raw dataset
trainName = "../../dataset/raw/train.txt"
testName = "../../dataset/raw/test.txt"
trainData = loadtxt(trainName, delimiter=",")
testData = loadtxt(testName, delimiter=",")

# Convert raw dataset to numpy
trainData = np.array(trainData)
testData = np.array(testData)

# Extracting Data if each patient has Parkinsons or Not
train_results = patient_type(trainData)
test_results = patient_type(testData)

# Calling method to convert data structures to pandas
data_to_pandas(train_results, test_results)

# -------------------------------------------- #
# CONVERTING SLOO DATA TO PANDAS CSV
# -------------------------------------------- # 

print("Converting SLOO DATA to PANDAS CSV")
# Filepath to SLOO Data
trainName = "../../dataset/sloo/train.csv"
testName = "../../dataset/sloo/test.csv"

# Conversion process
csv_to_pandas(trainName, testName)

