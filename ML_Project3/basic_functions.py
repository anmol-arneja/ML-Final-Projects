__author__ = 'Admin'
import numpy as np
import csv
import random


#------------------------------------------------Returning the Confusion Matrix-----------------------------------------

def get_confusion_matrix(actual, predicted):
    m = np.zeros((10,10))
    for a, b in zip(actual, predicted):
    	m[a,b] += 1

    class_totals = np.sum(m, axis=1)

    for i in xrange(10):
    	m[i] = m[i]*1. / class_totals[i]*1.

    return m

#-----------------------------------------Function for vectorizing the output-------------------------------------------
def vectorizer(n):
    v = np.zeros(10)
    v[n] = 1
    return v
#------------------------------Performing k-fold cross-validation over training data------------------------------------
class crossvalid():
    def __init__(self,train_inps,train_opts,k=5):
        assert len(train_inps) == len(train_opts)
        self.train_inps = train_inps
        self.train_opts =train_opts
        self.k = len(train_inps) // k
        self.i = 0


    def __iter__(self):
        return self

    def next(self):
        start,end = self.i * self.k,(self.i+1) * self.k
        if start >= len(self.train_inps):
            raise StopIteration
        self.i += 1

        train_data = np.concatenate((self.train_inps[:start,:],self.train_inps[end:,:]))
        valid_data = self.train_inps[start:end]
        train_result = np.concatenate((self.train_opts[:start],self.train_opts[end:]))
        valid_result = self.train_opts[start:end]

        return train_data,valid_data,train_result,valid_result

#------------------------------Creating a output CSV to store the output for validation set-----------------------------
def create_test_output(predictions,algo_name,s):
    with open("output_"+algo_name+"_"+s+".csv",'wb') as f:
        writer = csv.writer(f, quoting = csv.QUOTE_ALL)
        writer.writerow(['Id','Prediction'])
        for i,predict in enumerate(predictions):
            writer.writerow((str(i+1),predict))

def randomize_params(alphas, n_layers, nodes_layer):
    alpha = random.choice(alphas)
    n_lay = random.choice(n_layers)
    layers = [48*48]
    for _ in xrange(n_lay):
        layers.append(random.choice(nodes_layer))
    layers.append(10)

    return alpha, layers