__author__ = 'Admin'

from skimage.filter import gabor_kernel
from scipy import ndimage
from numpy import linalg
import numpy as np
from sklearn.preprocessing import normalize

def get_gabor_features(image,gabor_kernels):
    features = list()
    for i in gabor_kernels:
        convoimg =  ndimage.convolve(image.reshape(48,48),i,mode='wrap')
        mean = convoimg.mean()
        variance = convoimg.var()
        norm = linalg.norm(convoimg)
        norm1 = linalg.norm(convoimg,1)
        features.append(mean)
        features.append(variance)
        features.append(norm)
        features.append(norm1)
    return features
def get_gabor_kernels(num_theta = 4,sigmas=[1,3],freq = [0.05,0.25]):
    gaborkernels =[]
    for theta in range(num_theta):
        theta = theta / float(num_theta) * np.pi
        for sigma in sigmas:
            for fr in freq:
                kernel = np.real(gabor_kernel(fr,theta,sigma_x=sigma,sigma_y=sigma))
                gaborkernels.append(kernel)
    return gaborkernels
def load_file(filename):
    data = np.load(filename)
    return data
def normalizing(TrainX):
    normTrainX = normalize(TrainX)
    return normTrainX
def vectorize(n):
    v = np.zeros(10)
    v[n]=1
    return v

def save_train_features():
    print "Loading the Training Data Set Outputs"
    train_output = load_file("train_outputs.")







