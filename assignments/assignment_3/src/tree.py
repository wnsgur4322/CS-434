# CS 434 - Spring 2020
# implmentation assignment 3
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import numpy as np
import random
import pandas as pd
import math

class Node():
	"""
	Node of decision tree

	Parameters:
	-----------
	prediction: int
		Class prediction at this node
	feature: int
		Index of feature used for splitting on
	split: int
		Categorical value for the threshold to split on for the feature
	left_tree: Node
		Left subtree
	right_tree: Node
		Right subtree
	"""
	def __init__(self, prediction, feature, split, left_tree, right_tree):
		self.prediction = prediction
		self.feature = feature
		self.split = split
		self.left_tree = left_tree
		self.right_tree = right_tree


class DecisionTreeClassifier():
	"""
	Decision Tree Classifier. Class for building the decision tree and making predictions

	Parameters:
	------------
	max_depth: int
		The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
	"""

	def __init__(self, max_depth=None):
		self.max_depth = max_depth

	# take in features X and labels y
	# build a tree
	def fit(self, X, y, max_feat = None):
		self.num_classes = len(set(y))
		self.root = self.build_tree(X, y, depth=1, max_feat = max_feat)
	


	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy

	# function to build a decision tree
	def build_tree(self, X, y, depth, max_feat = None):
		num_samples, num_features = X.shape
		# which features we are considering for splitting on
		if max_feat == None:
			self.features_idx = np.arange(0, X.shape[1])
		else:
			self.features_idx = random.sample(range(51), max_feat)			

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_gain = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None

		# what we would predict at this node if we had to
		# majority class
		num_samples_per_class = [np.sum(y == i) for i in range(self.num_classes)]
		prediction = np.argmax(num_samples_per_class)

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					gain, left_X, right_X, left_y, right_y = self.check_split(X, y, feature, split)
					# if we have a better gain, use this split and keep track of data
					if gain > best_gain:
						best_gain = gain
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
		
		# if we haven't hit a leaf node
		# add subtrees recursively
		if best_gain > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

		# if we did hit a leaf node
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		gain = self.calculate_gini_gain(y, left_y, right_y)
		return gain, left_X, right_X, left_y, right_y

	def calculate_gini_gain(self, y, left_y, right_y):
		# not a leaf node
		# calculate gini impurity and gain
		gain = 0
		if len(left_y) > 0 and len(right_y) > 0:
			########################################
			#       YOUR CODE GOES HERE            #
			########################################

			# For U(AL) part
			l_pos = np.count_nonzero(left_y == 1)
			l_neg = len(left_y) - l_pos

			# For U(AR) part
			r_pos = np.count_nonzero(right_y == 1)
			r_neg = len(right_y) - r_pos

			# For U(A) part
			t_pos = np.count_nonzero(y == 1)
			t_neg = len(y) - t_pos

			# U(A) part
			u_top = 1 - (t_pos/len(y))**2 - (t_neg/len(y))**2
			# U(AL) part = 1 - (p_p)^2 - (p_n)^2
			u_left = 1 - (l_pos/len(left_y))**2 - (l_neg/len(left_y))**2
			# U(AR) part = 1 - (p_p)^2 - (p_n)^2
			u_right = 1 - (r_pos/len(right_y))**2 - (r_neg/len(right_y))**2
			
			# B = U(A) − plU(AL) − prU(AR)
			gain = u_top - ((len(left_y)/len(y))*u_left) - ((len(right_y)/len(y))*u_right)
			
			return gain
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0

class RandomForestClassifier():
	"""
	Random Forest Classifier. Build a forest of decision trees.
	Use this forest for ensemble predictions

	YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

	Parameters:
	-----------
	n_trees: int
		Number of trees in forest/ensemble
	max_features: int
		Maximum number of features to consider for a split when feature bagging
	max_depth: int
		Maximum depth of any decision tree in forest/ensemble
	"""
	def __init__(self, n_trees, max_features, max_depth):
		self.n_trees = n_trees
		self.max_features = max_features
		self.max_depth = max_depth

		##################
		# YOUR CODE HERE #
		##################

	# fit all trees
	def fit(self, X, y):
		bagged_X, bagged_y = self.bag_data(X, y)
		self.forest = []
		print('Fitting Random Forest...\n')
		for i in range(self.n_trees):
			tr = DecisionTreeClassifier(max_depth = self.max_depth)
			tr.fit(bagged_X[i], bagged_y[i], max_feat = self.max_features)
			self.forest.append(tr)
			
		
		

	def bag_data(self, X, y, proportion=1.0):
		bagged_X = []
		bagged_y = []
		
		for i in range(self.n_trees):
			randList = random.choices(range(0,2097), k=2098)
			bagged_X.append(X[randList])
			bagged_y.append(y[randList])

		# ensure data is still numpy arrays
		return np.array(bagged_X), np.array(bagged_y)


	def predict(self, X):
		preds = []
		preds = np.zeros(len(X)).astype(int)

		for i in range(self.n_trees):
			preds = preds + self.forest[i].predict(X)
            
		vote = self.n_trees/2
		preds = np.where(preds>=vote,1,0)
		
		return preds



################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################

class DecisionTreeAdaBoost():
	def __init__(self):
		self.max_depth = 1 # The depth is 1 = Decision Stump.

	# take in features X and labels y
	# build a tree
	def fit(self, X, y, feat_idx = None, weights = None):
		self.num_classes = list(set(y))
		self.root = self.build_tree(X, y, depth=1, feat_idx = feat_idx, weights = weights)



	# make prediction for each example of features X
	def predict(self, X):
		preds = [self._predict(example) for example in X]

		return preds

	# prediction for a given example
	# traverse tree by following splits at nodes
	def _predict(self, example):
		node = self.root
		while node.left_tree:
			if example[node.feature] < node.split:
				node = node.left_tree
			else:
				node = node.right_tree
		return node.prediction

	# accuracy
	def accuracy_score(self, X, y):
		preds = self.predict(X)
		accuracy = (preds == y).sum()/len(y)
		return accuracy


	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
	# Need to make one node tree
	# Working!!!!!!!!!!!!!!!!!!!!!!!!!!!
	# function to build a decision stump
	def build_tree(self, X, y, depth, feat_idx = None, weights = None):
		num_samples, num_features = X.shape
		
		# weight values for adaboost
		if weights is None:
			weights = np.ones(X.shape[0])
		weights = np.asarray(weights)
		best_left_w = None
		best_right_w = None

		# which features we are considering for splitting on
		if feat_idx == None:
			self.features_idx = np.arange(0, X.shape[1])
		else:
			self.features_idx = feat_idx			

		# store data and information about best split
		# used when building subtrees recursively
		best_feature = None
		best_split = None
		best_err = 0.0
		best_left_X = None
		best_left_y = None
		best_right_X = None
		best_right_y = None
		

		# what we would predict at this node if we had to
		# majority class
		'''
		num_samples_per_class = [np.sum(y == i) for i in range(-1, self.num_classes, 2)]
		prediction = np.argmax(num_samples_per_class)
		'''
		num_samples_per_class = [np.sum(y == i) for i in self.num_classes]
		prediction = self.num_classes[np.argmax(num_samples_per_class)]

		# if we haven't hit the maximum depth, keep building
		if depth <= self.max_depth:
			# consider each feature
			for feature in self.features_idx:
				# consider the set of all values for that feature to split on
				possible_splits = np.unique(X[:, feature])
				for split in possible_splits:
					# get the gain and the data on each side of the split
					# >= split goes on right, < goes on left
					err, left_X, right_X, left_y, right_y, left_w, right_w = self.check_split(X, y, feature, weights, split)
					# if we have a better gain, use this split and keep track of data
					if err > best_err:
						best_err = err
						best_feature = feature
						best_split = split
						best_left_X = left_X
						best_right_X = right_X
						best_left_y = left_y
						best_right_y = right_y
						best_left_w = left_w
						best_right_w = right_w

		# if we did hit a leaf node
		if best_err > 0.0:
			left_tree = self.build_tree(best_left_X, best_left_y, depth=depth+1, weights =best_left_w)
			right_tree = self.build_tree(best_right_X, best_right_y, depth=depth+1, weights = best_right_w)
			return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)
		
		return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)


	# gets data corresponding to a split by using numpy indexing
	def check_split(self, X, y, feature, weights, split):
		left_idx = np.where(X[:, feature] < split)
		right_idx = np.where(X[:, feature] >= split)
		left_X = X[left_idx]
		right_X = X[right_idx]
		left_y = y[left_idx]
		right_y = y[right_idx]
		left_w = weights[left_idx]
		right_w = weights[right_idx]

		# calculate gini impurity and gain for y, left_y, right_y
		err = self.calculate_weighted_err(y, left_y, right_y, left_w, right_w, weights)
		return err, left_X, right_X, left_y, right_y, left_w, right_w

	
	def calculate_weighted_err(self, y, left_y, right_y, left_w, right_w, w):
		# not a leaf node
		# calculate gini impurity and gain

		err = 0
		l_product = left_y * left_w
		r_product = right_y * right_w
		t_product = y*w

		if len(left_y) > 0 and len(right_y) > 0:
			# For U(AL) part
			l_pos = sum([a for a in l_product if a >= 0])
			l_neg = abs(sum([a for a in l_product if a < 0]))
			l_pos_p = l_pos / (l_pos+l_neg)
			l_neg_p = l_neg / (l_pos+l_neg)

			# For U(AR) part
			r_pos = sum([a for a in r_product if a >= 0])
			r_neg = abs(sum([a for a in r_product if a < 0]))
			r_pos_p = r_pos / (r_pos+r_neg)
			r_neg_p = r_neg / (r_pos+r_neg)

			# For U(A) part
			t_pos = sum([a for a in t_product if a >= 0])
			t_neg = abs(sum([a for a in t_product if a < 0]))
			t_pos_p = t_pos / (t_pos+t_neg)
			t_neg_p = t_neg / (t_pos+t_neg)

			# U(A) part
			u_top = 1 - (t_pos_p)**2 - (t_neg_p)**2
			# U(AL) part = 1 - (p_p)^2 - (p_n)^2
			u_left = 1 - (l_pos_p)**2 - (l_neg_p)**2
			# U(AR) part = 1 - (p_p)^2 - (p_n)^2
			u_right = 1 - (r_pos_p)**2 - (r_neg_p)**2
			
			# B = U(A) − plU(AL) − prU(AR)
			err = u_top - ((l_pos+l_neg)*u_left) - ((r_pos+r_neg)*u_right)

			return err
		# we hit leaf node
		# don't have any gain, and don't want to divide by 0
		else:
			return 0
			
class AdaBoostClassifier():
	def __init__(self, num_learner):
		self.num_learner = num_learner
		self.alphas = None
		self.stumps = None

	
	def fit(self, X, y):
		stumps = []

		#initialize weights
		# set all weights as 1/n (initial weights)
		weights = []
		alphas = []
		weights = (np.ones(X.shape[0]) / X.shape[0])

		for i in range(0, self.num_learner):
			stump = DecisionTreeAdaBoost()
			stump.fit(X, y, weights = weights)
			stumps.append(stump)
			prediction = stump.predict(X)
			
			classified = np.where(prediction == y, 1, 0)
			missclassified = np.where(prediction != y, 1, 0)

			accuracy = sum(classified)/len(classified)
			miss = sum(missclassified)/len(missclassified)

			err = np.sum(weights*missclassified)/np.sum(weights)

			alpha = (1/2)*np.log((1-err)/err)
			alphas.append(alpha)
			weights *= np.exp(alpha*missclassified)
			sum_weights = sum(weights)
			weights = weights/sum_weights

		self.alphas = alphas
		self.stumps = stumps

	def predict(self, X):
		preds = []
		predictions = []
		for i in range(len(self.stumps)):
			pred = np.asarray(self.stumps[i].predict(X)) * self.alphas[i]
			preds.append(pred)
		predictions = np.sign(np.sum(np.array(preds), axis=0))
		return predictions