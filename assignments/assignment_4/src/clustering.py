# CS 434 - Spring 2020
# implmentation assignment 4
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu
# Haewon Cho, choha@oregonstate.edu

import numpy as np
import random

class KMeans():
    """
    KMeans. Class for building an unsupervised clustering model
    """

    def __init__(self, k, max_iter=20):

        """
        :param k: the number of clusters
        :param max_iter: maximum number of iterations
        """

        self.k = k
        self.max_iter = max_iter

    def init_center(self, x):
        """
        initializes the center of the clusters using the given input
        :param x: input of shape (n, m)
        :return: updates the self.centers
        """

        self.centers = np.zeros((self.k, x.shape[1]))


        ################################
        #      YOUR CODE GOES HERE     #
        ################################
        

        random_list = random.sample(range(x.shape[0]), self.k)
        for i in range(self.k):
            self.centers[i,:] = x[random_list[i]]
            
            
            
        
        

    def revise_centers(self, x, labels):
        """
        it updates the centers based on the labels
        :param x: the input data of (n, m)
        :param labels: the labels of (n, ). Each labels[i] is the cluster index of sample x[i]
        :return: updates the self.centers
        """

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            self.centers[i, :] = x[wherei, :].mean(0)



    def predict(self, x):
        """
        returns the labels of the input x based on the current self.centers
        :param x: input of (n, m)
        :return: labels of (n,). Each labels[i] is the cluster index for sample x[i]
        """
        labels = np.zeros((x.shape[0]), dtype=int)
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################

        comp_distance = np.zeros((x.shape[0], self.k))
        for i in range(self.k):
            distance = 0
            distance = x - self.centers[i]
            distance = np.square(distance)
            distance = np.sum(distance, axis = 1)
            comp_distance[:,i] = distance

        labels = np.argmin(comp_distance, axis=1)
       
        return labels

    def get_sse(self, x, labels):
        """
        for a given input x and its cluster labels, it computes the sse with respect to self.centers
        :param x:  input of (n, m)
        :param labels: label of (n,)
        :return: float scalar of sse
        """

        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################


        sse = 0.

        for i in range(self.k):
            wherei = np.squeeze(np.argwhere(labels == i), axis=1)
            sse += sum(sum(list((x[wherei, :] - self.centers[i, :])**2)))


        print("sse: ", sse)

        
        return sse


    def get_purity(self, x, y):
        """
        computes the purity of the labels (predictions) given on x by the model
        :param x: the input of (n, m)
        :param y: the ground truth class labels
        :return:
        """
        labels = self.predict(x)
        purity = 0
        ##################################
        #      YOUR CODE GOES HERE       #
        ##################################
        ground_truth = 0
        for i in range(self.k):
            b = [x==i for x in labels]
            print(b)
            res = np.array(range(len(b)))
            print(i)
            k_predict = res[b]
            y_preds = y[k_predict]
            y_list = y_preds.tolist()
            max_class = max(set(y_list), key=y_list.count)
            ground_truth += y_list.count(max_class)

        purity = ground_truth / y.shape[0]

        return purity

    def fit(self, x):
        """
        this function iteratively fits data x into k-means model. The result of the iteration is the cluster centers.
        :param x: input data of (n, m)
        :return: computes self.centers. It also returns sse_veersus_iterations for x.
        """

        # intialize self.centers
        self.init_center(x)

        sse_vs_iter = []
        for iter in range(self.max_iter):
            # finds the cluster index for each x[i] based on the current centers
            labels = self.predict(x)

            # revises the values of self.centers based on the x and current labels
            self.revise_centers(x, labels)

            # computes the sse based on the current labels and centers.
            sse = self.get_sse(x, labels)

            sse_vs_iter.append(sse)

        return sse_vs_iter
