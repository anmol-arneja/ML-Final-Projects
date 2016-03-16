__author__ = 'Admin'

import numpy as np
#from sklearn.svm import SVC
from classifiers import *
from basic_functions import *
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import *

#-----------------------------------------------------------------------------------------------------------------------
'''def vectorr(n):
	v = np.zeros(10)
	v[n] = 1
	return v
#-----------------------------------------Loading Taining input with PCA------------------------------------------------
print "Loading Training inputs with PCA"
train_inputs_pca = np.load("pca_train_inputs.npy")
print "Done loading Training Inputs"

#---------------------------------------Loading Training Outputs--------------------------------------------------------
train_outputs_pca = np.load("train_outputs.npy")

#----------------------------------------Loading Validation Inputs------------------------------------------------------
print "Loading the Validation inputs"
valid_inputs_pca = np.load("pca_valid_inputs.npy")
print "Done loading Validation Inputs"

#--------------------------------------Loading Validation Outputs-------------------------------------------------------
valid_outputs_pca = np.load("valid_outputs.npy")

#-------------------------------------------------Loading Test Inputs---------------------------------------------------
print "Loading the Test Inputs"
test_inputs_pca = np.load("pca_test_inputs.npy")
print "Done loading Test Inputs"

#----------------------------------------Building a classifier to predict Valid output----------------------------------------
algorithm = os.path.basename(__file__)
print "Running %s script to predict the output for Validation Set"%algorithm
algo_name = algorithm.replace(".py",'')
s = "valid"
classifier = Perceptron(alpha=0.0005,num_iters=25)
classifier.train(train_inputs_pca,np.asarray(map(vectorr,train_outputs_pca)))
valid_prediction = map(classifier.predict,valid_inputs_pca)
create_test_output(valid_prediction,algo_name,s)

#----------------------------------------Building a classifier to predict Test output----------------------------------------
algorithm = os.path.basename(__file__)
print "Running %s script to predict the output for Test Set"%algorithm
algo_name = algorithm.replace(".py",'')
s = "test"
test_prediction = map(classifier.predict,test_inputs_pca)
create_test_output(test_prediction,algo_name,s)
#------------------------------------Measuring the accuracy of predicted outputs----------------------------------------
correct_guesses = filter(lambda x : x[0] == x[1],zip(valid_prediction,valid_outputs_pca))
print "The number of correct guesses using %s algorithm is %d correct out of %d test inputs tested"%(algo_name,len(correct_guesses),len(valid_inputs_pca))
percent_correct_guesses = len(correct_guesses)* 1. / len(valid_inputs_pca) * 100
print "The percentage of correct guesses is %f"%percent_correct_guesses

#------------------------Performing Cross Validation to find the best values for hyperparameters------------------------
#hyperparametes: alpha and num_iters
results= open("results_crossvalid_perceptron.txt",'w')
alphas = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
num_iters = [10,15,25,30,40,50]
cross_val_results = []
cross_val_train_results=[]
cross_val_confusion_matrices = []

for iters in num_iters:
	for alpha in alphas:
		predictions =[]
		valid_success_rates =[]
		train_success_rates=[]
		for data in crossvalid(train_inputs_pca,train_outputs_pca,k=5):
			train_data,valid_data,train_result,valid_result = data
			cross_classifier = Perceptron(alpha=alpha,num_iters=iters)
			cross_classifier.train(train_data,np.asarray(map(vectorr,train_result)))
			training_guesses=map(cross_classifier.predict,train_data)
			train_correct_guesses = filter(lambda x: x[0] == x[1],zip(training_guesses,train_result))
			train_correct_ratio = len(train_correct_guesses)*1. / len(train_data)
			train_success_rates.append(train_correct_ratio)
			valid_guesses = map(cross_classifier.predict,valid_data)
			valid_correct_guesses = filter(lambda x : x[0] == x[1],zip(valid_guesses,valid_result))
			valid_correct_ratio = len(valid_correct_guesses)*1.  / len(valid_data)
			valid_success_rates.append(valid_correct_ratio)
			predictions.extend(valid_guesses)
		confusion_matrix = get_confusion_matrix(train_outputs_pca,valid_outputs_pca)
		train_success_percent = sum(train_success_rates)/len(train_success_rates) * 100
		valid_success_percent = sum(valid_success_rates)/len(valid_success_rates) * 100
		print "CrossValidation Accuracy for alpha = %f and number of iterations = %d is %f percent"%(alpha,iters,valid_success_percent)
		results.write("CrossValidation Accuracy for alpha = %f and number of iterations = %d is %f percent"%(alpha,iters,valid_success_percent) + "\n")
		cross_val_confusion_matrices.append(confusion_matrix)
		cross_val_results.append(valid_success_percent)
		cross_val_train_results.append(train_success_percent)
		np.save("perceptron_crossval_confmatrices",cross_val_confusion_matrices)
		np.save("perceptron_crossval_results",cross_val_results)
		np.save("perceptron_crossval_training_accuracy",cross_val_train_results)

a = np.load("perceptron_crossval_confmatrices.npy")
print a
print "Done"'''

class PerceptronPlotter(object):

	def __init__(self):
		# these were the parameters considered
		self.alphas = np.array([0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001])
		self.niters = np.array([10, 15, 20, 25, 30, 35])

		# load crossval results
		self.crossval_results = np.load('perceptron_crossval_results.npy')

		self.crossval_matrices = np.load('perceptron_crossval_confmatrices.npy')

		self.crossval_training = np.load('perceptron_crossval_training_accuracy.npy')

		self.argmax = np.argmax(self.crossval_results)
		self.best = np.max(self.crossval_results)
		self.best_alpha = self.alphas[self.argmax % 7]
		self.best_niters = self.niters[self.argmax / 7]

	def best_results(self):
		print 'Best PERCEPTRON: '
		print 'accuracy = %f' % self.best
		print 'alpha = %f' % self.best_alpha
		print 'n_iters = %d' % self.best_niters


	def gridsearch(self):
		accuracies = self.crossval_results.reshape((len(self.niters),len(self.alphas)))

		fig = plt.figure(figsize=(7,6))
		ax1 = fig.add_subplot(111)

		ax1.tick_params(direction='out', which='both')
		ax1.set_xlabel('Learning rate')
		ax1.set_ylabel('Number of iterations')
		ax1.set_xticks(self.alphas)
		ax1.set_yticks(self.niters)

		cax = ax1.contourf(self.alphas, self.niters, accuracies, np.arange(22, 27, 0.25), extend='both')
		# cs=ax1.contour(alphas, niters, accuracies, np.arange(20,27,0.25),colors='k')
		# ax1.clabel(cs, fmt = '%d', colors = 'k')

		ax1.set_xscale('log')

		cbar = fig.colorbar(cax)

		plt.savefig('perceptron_gridsearch.pdf')

	def conf_matrix(self):
		fig = plt.figure()

		ax = fig.add_subplot(111)

		mat = self.crossval_matrices[self.argmax].tolist()
		cax = ax.matshow(mat, cmap=cm.jet)
		fig.colorbar(cax)

		for x in xrange(10):
			for y in xrange(10):
				ax.annotate('%4.2f' % (mat[x][y]), xy=(y,x), horizontalalignment='center', verticalalignment='center', color='white')

		plt.xticks(np.arange(10))
		plt.yticks(np.arange(10))
		ax.set_title('Prediction', fontsize=16)
		ax.set_ylabel('True label', fontsize=16)

		plt.savefig('perceptron_confusion.pdf')

	def plt_learningrate(self):
		'''
		Perceptron's learning rate -- validation vs training set
		Uses the best n_iter, varies learning rate
		'''
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		crossval_results = self.crossval_results
		crossval_training = self.crossval_training
		argmax = self.argmax

		valid_acc = crossval_results[(argmax/7)*7:(argmax/7)*7 + 7]
		train_acc = crossval_training[(argmax/7)*7:(argmax/7)*7 + 7]

		v, = ax.plot(self.alphas, valid_acc, marker='D', color='green', label='Validation, %d iterations' % self.best_niters)
		t, = ax.plot(self.alphas, train_acc, marker='D', color='green', linestyle=':', label='Training, %d iterations' % self.best_niters)

		# legend
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc=0)

		# labels
		ax.set_xlabel('Learning rate')
		ax.set_ylabel('Mean accuracy')
		ax.set_xscale('log')

		plt.savefig("perceptron1_learningrate.pdf")


	def plt_niters(self):
		'''
		Perceptron's number of iterations -- validation vs training set
		Uses the best alpha, varies number of iterations.
		'''
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)

		crossval_results = self.crossval_results
		crossval_training = self.crossval_training
		argmax = self.argmax

		valid_acc = []
		for i in xrange(len(self.niters)):
			valid_acc.append(crossval_results[7*i + argmax%7])
		train_acc = []
		for i in xrange(len(self.niters)):
			train_acc.append(crossval_training[7*i + argmax%7])

		v, = ax.plot(self.niters, valid_acc, marker='D', color='green', label='Validation, alpha=%5.4f' % self.best_alpha)
		t, = ax.plot(self.niters, train_acc, marker='D', color='green', linestyle=':', label='Training, alpha=%5.4f' % self.best_alpha)

		# legend
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc=0)

		# labels
		ax.set_xlabel('Number of iterations')
		ax.set_ylabel('Mean accuracy')

		plt.savefig('perceptron1_iterations.pdf')



a = PerceptronPlotter()
#a.gridsearch()
#a.conf_matrix()
a.plt_learningrate()
a.plt_niters()


