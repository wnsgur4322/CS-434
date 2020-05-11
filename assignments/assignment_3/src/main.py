# CS 434 - Spring 2020
# implmentation assignment 3
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu
# Haewon Cho, choha@oregonstate.edu

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from statistics import mean
import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info, draw_plot_1, draw_plot_2, zero_to_negone
from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier

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
	

def random_forest_testing(x_train, y_train, x_test, y_test, n_trees, max_features):
	print('Random Forest')
	print("max_depth: %d, max_features: %d, n_trees: %d" % (7,max_features, n_trees))
	rclf = RandomForestClassifier(n_trees, max_features, max_depth=7)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	preds_train = rclf.predict(x_train)

	print('F1 Train {}'.format(f1(y_train, preds_train)))
	print('F1 Test {}\n'.format(f1(y_test, preds)))

	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_test, preds)

def ada_boost_testing(x_train, y_train, x_test, y_test, num_learner):
	print('Ada Boost and L(', num_learner, ')')
	aba = AdaBoostClassifier(num_learner)
	aba.fit(x_train, y_train)
	preds_train = aba.predict(x_train)
	preds_test = aba.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = aba.predict(x_test)
	preds_train = aba.predict(x_train)

	print('F1 Train {}'.format(f1(y_train, preds_train)))
	print('F1 Test {}\n'.format(f1(y_test, preds)))

	return train_accuracy, test_accuracy, f1(y_train, preds_train), f1(y_test, preds)

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
		draw_plot_1(train_accs, test_accs, "DT")
		draw_plot_2(f1_train_accs, f1_test_accs, "DT")
		
		# Q1 - E part
		print("The best accuracies (train, test, F1_train, F1_test): %f %f %f %f" % (max(train_accs), max(test_accs), max(f1_train_accs), max(f1_test_accs)))
		with open("accuracy_result.txt", "w") as res:
			res.write("train_acc\ttest_acc\tF1_train_acc\tF1_test_acc\n")
			for i in range(len(train_accs)):
				res.write("{0}\t{1}\t{2}\t{3}\n".format(train_accs[i], test_accs[i], f1_train_accs[i], f1_train_accs[i]))
			
			res.close()
		print("accuracy_result.txt created !\n")

	# Part 2 - Random Forest
	if args.random_forest == 1:
		forest_train_accs = []
		forest_test_accs = []
		forest_f1_train_accs = []
		forest_f1_test_accs = []

		
		#Q2 - B part
		# looping n_trees
		for i in range(10, 210, 10):
			forest_train_acc, forest_test_acc, forest_f1_train_acc, forest_f1_test_acc = random_forest_testing(x_train, y_train, x_test, y_test, i, 11)
			forest_train_accs.append(forest_train_acc * 100)
			forest_test_accs.append(forest_test_acc * 100)
			forest_f1_train_accs.append(forest_f1_train_acc * 100)
			forest_f1_test_accs.append(forest_f1_test_acc * 100)
		
		print("check:", forest_test_accs)
		#Q2 - B plot
		draw_plot_1(forest_train_accs, forest_test_accs, "RF_b")
		draw_plot_2(forest_f1_train_accs, forest_f1_test_accs, "RF_b")

		#Q2 - D part
		forest_train_accs = []
		forest_test_accs = []
		forest_f1_train_accs = []
		forest_f1_test_accs = []

		#Q2 - B part
		# looping max_features
		max_features = [1, 2, 5, 8, 10, 20, 25, 35, 50]
		for i in max_features:
			forest_train_acc, forest_test_acc, forest_f1_train_acc, forest_f1_test_acc = random_forest_testing(x_train, y_train, x_test, y_test, 50, i)
			forest_train_accs.append(forest_train_acc * 100)
			forest_test_accs.append(forest_test_acc * 100)
			forest_f1_train_accs.append(forest_f1_train_acc * 100)
			forest_f1_test_accs.append(forest_f1_test_acc * 100)
		
		#Q2 - D plot
		draw_plot_1(forest_train_accs, forest_test_accs, "RF_d")
		draw_plot_2(forest_f1_train_accs, forest_f1_test_accs, "RF_d")

		
		#Q2 - F chart
		for i in range(10):
			forest_train_acc, forest_test_acc, forest_f1_train_acc, forest_f1_test_acc = random_forest_testing(x_train, y_train, x_test, y_test, 152, 25)
			forest_train_accs.append(forest_train_acc * 100)
			forest_test_accs.append(forest_test_acc * 100)
			forest_f1_train_accs.append(forest_f1_train_acc * 100)
			forest_f1_test_accs.append(forest_f1_test_acc * 100)

		print("best train accuracy")
		print(forest_train_accs)
		print("best test accuracy")
		print(forest_test_accs)
		print("best f1 train accuracy")
		print(forest_f1_train_accs)
		print("best f1 test accuracy")
		print(forest_f1_test_accs)

		print("mean of best train accuracy")
		print(mean(forest_train_accs))
		print("mean of best test accuracy")
		print(mean(forest_test_accs))
		print("mean of best f1 train accuracy")
		print(mean(forest_f1_train_accs))
		print("mean of best f1 test accuracy")
		print(mean(forest_f1_test_accs))



		

	# Part 3 - Ada Boosting
	# part 3 - A
	if args.ada_boost == 1:
		adaboost_train_accs = []
		adaboost_test_accs = []
		adaboost_f1_train_accs = []
		adaboost_f1_test_accs = []
		
		zero_to_negone(y_train, y_test)

		#Q3 - E
		for i in range(10, 210, 10):
			adaboost_train_acc, adaboost_test_acc, adaboost_f1_train_acc, adaboost_f1_test_acc = ada_boost_testing(x_train, y_train, x_test, y_test, i)
			adaboost_train_accs.append(adaboost_train_acc * 100)
			adaboost_test_accs.append(adaboost_test_acc * 100)
			adaboost_f1_train_accs.append(adaboost_f1_train_acc * 100)
			adaboost_f1_test_accs.append(adaboost_f1_test_acc * 100)
		#Q3 - F
		draw_plot_1(adaboost_train_accs, adaboost_test_accs, "ADA_f")
		draw_plot_2(adaboost_f1_train_accs, adaboost_f1_test_accs, "ADA_f")
	print('Done')
	
	





