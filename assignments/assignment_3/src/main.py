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

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, tree_draw_plot_1, tree_draw_plot_2
from tree import DecisionTreeClassifier, RandomForestClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test, max_depth):
	print('Decision Tree')
	print("depth : %d" % max_depth)
	
	clf = DecisionTreeClassifier(max_depth)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	preds_train =clf.predict(x_train)

	print('F1 Train {}'.format(f1(y_train, preds_train)))
	print('F1 Test {}\n'.format(f1(y_test, preds)))
	
	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_test, preds) 
	

def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=50)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))



###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	if args.county_dict == 1:
		county_info(args)
	
	#decision tree
	if args.decision_tree == 1:
		train_accs = []
		test_accs = []
		f1_train_accs = []
		f1_test_accs = []

		for i in range(25):
			train_acc, test_acc, f1_train_acc, f1_test_acc = decision_tree_testing(x_train, y_train, x_test, y_test, i + 1)
			train_accs.append(train_acc * 100)
			test_accs.append(test_acc * 100)
			f1_train_accs.append(f1_train_acc * 100)
			f1_test_accs.append(f1_test_acc * 100)
		
		# Q1 - D plot part
		tree_draw_plot_1(train_accs, test_accs)
		tree_draw_plot_2(f1_train_accs, f1_test_accs)
		
		# Q1 - E part
		print("The best accuracies (train, test, F1_train, F1_test): %f %f %f %f" % (max(train_accs), max(test_accs), max(f1_train_accs), max(f1_test_accs)))
		with open("accuracy_result.txt", "w") as res:
			res.write("train_acc\ttest_acc\tF1_train_acc\tF1_test_acc\n")
			for i in range(len(train_accs)):
				res.write("{0}\t{1}\t{2}\t{3}\n".format(train_accs[i], test_accs[i], f1_train_accs[i], f1_train_accs[i]))
			
			res.close()
		print("accuracy_result.txt created !\n")

	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)

	print('Done')
	
	





