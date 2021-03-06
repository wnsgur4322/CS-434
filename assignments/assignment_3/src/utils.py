# CS 434 - Spring 2020
# implmentation assignment 3
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.metrics import f1_score

def load_data(rootdir='./'):	
	x_train = np.loadtxt(rootdir+'x_train.csv', delimiter=',').astype(int)
	y_train = np.loadtxt(rootdir+'y_train.csv', delimiter=',').astype(int)
	x_test = np.loadtxt(rootdir+'x_test.csv', delimiter=',').astype(int)
	y_test = np.loadtxt(rootdir+'y_test.csv', delimiter=',').astype(int)
	y_train[y_train == -1] = 0
	y_test[y_test == -1] = 0
	return x_train, y_train, x_test, y_test

def load_dictionary(rootdir='./'):
	county_dict = pd.read_csv(rootdir+'county_facts_dictionary.csv')
	return county_dict

def dictionary_info(county_dict):
	for i in range(county_dict.shape[0]):
		print('Feature: {} - Description: {}'.format(i, county_dict['description'].iloc[i]))

def accuracy_score(preds, y):
	accuracy = (preds == y).sum()/len(y)
	return accuracy

def f1(y, yhat):
	return f1_score(y, yhat)



###########################################################################
# you may add plotting or data processing functions (etc) in here if desired
###########################################################################

# this function is for drawing plot between tree depth and training/test performance
def draw_plot_1(train_accs, test_accs, algorithm):
	if algorithm == "DT":
		depth_list = list(range(1, 26))
		plt.plot(
			# training ASE drew with red solid line
			depth_list, train_accs, 'r-',
			depth_list, test_accs, 'b-'
		)
		plt.legend(['train_accuracy', 'test_accuracy'])

		plt.xlabel('Decision tree depth value (1 ~ 25)')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('tree_accuracy_vs_tree_depth.png')
		print("tree_accuracy_vs_tree_depth.png is created successfully!")
		plt.show()
	
	if algorithm == "RF_b":
		n_trees = list(range(10, 210, 10))
		plt.plot(
			# training ASE drew with red solid line
			n_trees, train_accs, 'r-',
			n_trees, test_accs, 'b-'
		)
		plt.legend(['train_accuracy', 'test_accuracy'])
		plt.xlabel('Random forest n_trees (10, 20, 30 .. 200)')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('random_forest_accuracy_vs_n_trees.png')
		print("random_forest_accuracy_vs_n_trees.png is created successfully!")
		plt.show()

	if algorithm == "RF_d":
		max_features = [1, 2, 5, 8, 10, 20, 25, 35, 50]
		plt.plot(
			# training ASE drew with red solid line
			max_features, train_accs, 'r-',
			max_features, test_accs, 'b-'
		)
		plt.legend(['train_accuracy', 'test_accuracy'])
		plt.xlabel('Random forest max_features [1, 2, 5, 8, 10, 20, 25, 35, 50]')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('random_forest_accuracy_vs_max_features.png')
		print("random_forest_accuracy_vs_max_features.png is created successfully!")
		plt.show()

	if algorithm == "ADA_f":
		n_trees = list(range(10, 210, 10))
		plt.plot(
			# training ASE drew with red solid line
			n_trees, train_accs, 'r-',
			n_trees, test_accs, 'b-'
		)
		plt.legend(['train_accuracy', 'test_accuracy'])
		plt.xlabel('Adaboost L (10, 20, 30 .. 200)')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('adaboost_accuracy_vs_L.png')
		print("adaboost_accuracy_vs_L.png is created successfully!")
		plt.show()

	return 0

# this function is for drawing plot between tree depth and F1 test performance
def draw_plot_2(f1_train_accs, f1_test_accs, algorithm):
	if algorithm =="DT":
		depth_list = list(range(1, 26))
		plt.plot(
			# training ASE drew with red solid line
			depth_list, f1_train_accs, 'r-',
			depth_list, f1_test_accs, 'b-'
		)
		plt.legend(['f1_train_accuracy', 'f1_test_accuracy'])
		plt.xlabel('decision tree depth value (1 ~ 25)')
		plt.ylabel('accuracy %')
		# save the plot as a png file
		plt.savefig('f1_tree_accuracy_vs_tree_depth.png')
		print("f1_tree_accuracy_vs_tree_depth.png is created successfully!")
		plt.show()

	if algorithm == "RF_b":
		n_trees = list(range(10, 210, 10))
		plt.plot(
			# training ASE drew with red solid line
			n_trees, f1_train_accs, 'r-',
			n_trees, f1_test_accs, 'b-'
		)
		plt.legend(['f1_train_accuracy', 'f1_test_accuracy'])
		plt.xlabel('Random forest n_trees (10, 20, 30 .. 200)')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('f1_random_forest_accuracy_vs_n_trees.png')
		print("f1_random_forest_accuracy_vs_n_trees.png is created successfully!")
		plt.show()

	if algorithm == "RF_d":
		max_features = [1, 2, 5, 8, 10, 20, 25, 35, 50]
		plt.plot(
			# training ASE drew with red solid line
			max_features, f1_train_accs, 'r-',
			max_features, f1_test_accs, 'b-'
		)
		plt.legend(['f1_train_accuracy', 'f1_test_accuracy'])
		plt.xlabel('Random forest max_features [1, 2, 5, 8, 10, 20, 25, 35, 50]')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('f1_random_forest_accuracy_vs_max_features.png')
		print("f1_random_forest_accuracy_vs_max_features.png is created successfully!")
		plt.show()


	if algorithm == "ADA_f":
		n_trees = list(range(10, 210, 10))
		plt.plot(
			# training ASE drew with red solid line
			n_trees, f1_train_accs, 'r-',
			n_trees, f1_test_accs, 'b-'
		)
		plt.legend(['f1_train_accuracy', 'f1_test_accuracy'])
		plt.xlabel('Adaboost L (10, 20, 30 .. 200)')
		plt.ylabel('accuracy %')
	# save the plot as a png file
		plt.savefig('f1_adaboost_accuracy_vs_L.png')
		print("f1_adaboost_accuracy_vs_L.png is created successfully!")
		plt.show()
	return 0


# this function is for chainging 0 label to -1 for AdaBoost
def zero_to_negone(y_train, y_test):
	for i in range(len(y_train)):
		if y_train[i] == 0:
			y_train[i] = -1
	
	for i in range(len(y_test)):
		if y_test[i] == 0:
			y_test[i] = -1
