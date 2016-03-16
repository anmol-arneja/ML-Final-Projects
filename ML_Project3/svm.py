__author__ = 'Admin'
import numpy as np
from sklearn import linear_model
from basic_functions import *
import os
import matplotlib.pyplot as plt
from matplotlib.pylab import *

#-------------------------------------------Loading Training Inputs-----------------------------------------------------
print "loading Training Inputs"
train_inputs = np.load("standardized_train_inputs.npy")
print "Done with loading Training inputs"

#------------------------------------------Loading Training Outputs-----------------------------------------------------
train_outputs = np.load("train_outputs.npy")

#------------------------------------------Loading Validation Set Inputs------------------------------------------------
print "Loading validation Inputs"
valid_inputs = np.load("standardized_valid_inputs.npy")
print "Done with loading Validation Inputs"

#-------------------------------------------Loading Validation set outputs----------------------------------------------
valid_outputs = np.load("valid_outputs.npy")

#--------------------------------------------Loading Testing Inputs-----------------------------------------------------
print "Loading Testing Inputs"
test_inputs = np.load("Standardized_test_inputs.npy")
print "Done with loading Testing Inputs"

#--------------------------------------------Training the Classifier----------------------------------------------------
'''algorithm = os.path.basename(__file__)
print "Running %s script to predict the output for Test Set"%algorithm
algo_name = algorithm.replace(".py",'')
print "Training the Classifier"
classifier = linear_model.SGDClassifier(alpha =0.0001 ,n_iter=5 )
classifier.fit(train_inputs,train_outputs)

#------------------------------------Predicting the output for validation set-------------------------------------------
s = "valid"
valid_predictions = classifier.predict(valid_inputs)
create_test_output(valid_predictions,algo_name,s)

#------------------------------------Predicting the output for Test set-------------------------------------------------
s = "test"
test_predictions = classifier.predict(test_inputs)
create_test_output(test_predictions,algo_name,s)

#------------------------Measuring the accuracy of prediction for validation set----------------------------------------
correct_guesses = filter(lambda x : x[0] == x[1],zip(valid_outputs,valid_predictions))
print "The number of correct guesses using %s algorithm is %d correct out of %d test inputs tested"%(algo_name,len(correct_guesses),len(valid_inputs))
percent_correct_guesses = len(correct_guesses)* 1. / len(valid_inputs) * 100
print "The percentage of correct guesses is %f"%percent_correct_guesses

#--------------------------------------CrossValidating to find the best values of hyperparameters-----------------------
#hyperparameters : learning rate(alpha) and number of iterations
results = open("results_crossvalid_SVM.txt",'w')
alphas = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001,0.00005,0.00001,0.000005,0.000001]
cross_val_results = []
cross_val_train_results = []
cross_val_confusion_matrices = []

for num_iters in range(1,30):
	for alpha in alphas:
		training_success_rates =[]
		valid_success_rates =[]
		predictions =[]
		for data in crossvalid(train_inputs,train_outputs,k=5):
			train_data,valid_data,train_result,valid_result = data
			cross_classifier =linear_model.SGDClassifier(alpha=alpha,n_iter=num_iters)
			cross_classifier.fit(train_data,train_result)
			training_guesses = cross_classifier.predict(train_data)
			correct_training_guesses = filter(lambda  x : x[0] == x[1],zip(training_guesses,train_result))
			train_correct_ratio = len(correct_training_guesses) * 1. / len(train_data)
			training_success_rates.append(train_correct_ratio)
			valid_guesses = cross_classifier.predict(valid_data)
			correct_valid_guesses = filter(lambda x : x[0] == x[1],zip(valid_guesses,valid_result))
			valid_correct_ratio = len(correct_valid_guesses) * 1. / len(valid_data)
			valid_success_rates.append(valid_correct_ratio)
			predictions.extend(valid_guesses)
		confusion_matrix = get_confusion_matrix(train_outputs,valid_outputs)
		train_success_percent = sum(training_success_rates)/len(training_success_rates)*100
		valid_success_rates = sum(training_success_rates) / len(training_success_rates)*100
		print "CrossValidation Accuracy for alpha = %f and number of iterations = %d is %f percent"%(alpha,num_iters,valid_success_rates)
		results.write( "CrossValidation Accuracy for alpha = %f and number of iterations = %d is %f percent"%(alpha,num_iters,valid_success_rates) + "\n")
		cross_val_confusion_matrices.append(confusion_matrix)
		cross_val_results.append(valid_success_rates)
		cross_val_train_results.append(train_success_percent)
		np.save("svm_crossval_confmatrices",cross_val_confusion_matrices)
		np.save("svm_crossval_results",cross_val_results)
		np.save("svm_crossval_training_accuracy",cross_val_train_results)'''


class PerceptronPlotter(object):

	def __init__(self):
		# these were the parameters considered
		self.alphas = np.array([0.1,0.01,0.001,0.0001,0.00001,0.000005,0.000001])
		self.niters = np.array([5,10,15,20,25,30])

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

		plt.savefig('svm_gridsearch.pdf')

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

		plt.savefig('svm_confusion_new.pdf')

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

		plt.savefig("svm_learningrate_new.pdf")


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

		v, = ax.plot(self.niters, valid_acc, marker='D', color='green')
		t, = ax.plot(self.niters, train_acc, marker='D', color='green', linestyle=':')

		# legend
		handles, labels = ax.get_legend_handles_labels()
		ax.legend(handles, labels, loc=0)

		# labels
		ax.set_xlabel('Number of iterations')
		ax.set_ylabel('Mean accuracy')

		plt.savefig('svm_iterations_new.pdf')



a = PerceptronPlotter()
#a.gridsearch()
a.conf_matrix()
a.plt_learningrate()
a.plt_niters()

