# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import numpy as np
import pandas as pd
import random
import pylab
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse as arg
import math
import mpmath

# command line prompt with argparse library
parser = arg.ArgumentParser()
parser.add_argument('usps_train')
parser.add_argument('usps_test')
parser.add_argument('learningrate')
args = parser.parse_args()

# create matrix with usps dataset
def create_matrix(data):
    # convert the pandas dataframe into numpy array matrix
    data_matrix = data.to_numpy()
    data_matrix = data_matrix.tolist()

    X = []
    Y = []
    for i in range(len(data_matrix)):

        # X is for attributes
        X.append(data_matrix[i][:-1])
        # Y is for MEDV values
        Y.append([data_matrix[i][-1]])

    X = np.matrix(X)
    Y = np.array(Y)

    return X,Y

#Logistic regression part
def batch_learning(X_train, Y_train, X_test, Y_test, w, learning_rate, acc_train, acc_test):
    #set epsilon for stopping condition
    epsilon = 10000
    norm_delta = epsilon + 1

    print("The learning rate is {0} (recommend to set 0.0001) \nThe epilon is {1}".format(learning_rate, epsilon))

    while(epsilon < norm_delta):
        print("the norm is {0}".format(norm_delta))
        #set up a gradient
        gradient = np.array([0] * X_train.shape[1])

        #for looping i = 1, ... , n (the number of training data rows)
        for i in range(X_train.shape[0]):
            # set xi for formula (a row)
            x_i = np.array(X_train[i,:])[0]
            # from the lecture slide, the y hat (prediction) is 1 / (1 + e^(-w^T xi))
            # use try and except condition because OverflowError will be happened if learning rate value is too huge
            try:
                y_hat = 1 / (1 + math.exp(-1 * np.dot(w, x_i)))
            
            except OverflowError:
                y_hat = float(1 / (1 + mpmath.exp(-1 * np.dot(w, x_i))))
                        
            # 𝛻 ← 𝛻 + (yi hat - yi) xi
            gradient = gradient + (y_hat - Y_train[i]) * x_i

        # 𝒘 ← 𝒘 − 𝜂𝛻 
        w = w - (learning_rate * gradient)

        # Until |𝛻| ≤ epsilon
        norm_delta = np.linalg.norm(gradient)


        #accuracy calculation part
        correct_num = 0
        
        # calculate the accuracy of training data
        for i in range(X_train.shape[0]):
            # set xi for formula (a row)
            x_i = np.array(X_train[i,:])[0]
            # from the lecture slide, the y hat (prediction) is 1 / (1 + e^(-w^T xi))
            # use try and except condition because OverflowError will be happened if learning rate value is too huge
            try:
                y_hat = int(1 / (1 + math.exp(-1 * np.dot(w, x_i))))
            
            except OverflowError:
                y_hat = int(1 / (1 + mpmath.exp(-1 * np.dot(w, x_i))))
            
            if (y_hat == Y_train[i]):
                correct_num += 1
        
        acc_train.append(correct_num / Y_train.shape[0])

        # calculate the accuracy of test data

        correct_num = 0

        for i in range(X_test.shape[0]):
            # set xi for formula (a row)
            x_i = np.array(X_test[i,:])[0]
            # from the lecture slide, the y hat (prediction) is 1 / (1 + e^(-w^T xi))
            # use try and except condition because OverflowError will be happened if learning rate value is too huge
            try:
                y_hat = int(1 / (1 + math.exp(-1 * np.dot(w, x_i))))
            
            except OverflowError:
                y_hat = int(1 / (1 + mpmath.exp(-1 * np.dot(w, x_i))))
            
            if (y_hat == Y_test[i]):
                correct_num += 1
        
        acc_test.append(correct_num / Y_test.shape[0])

    print("the weight vector is {}".format(w))

def draw_plot(acc_train, acc_test):

    # set up x-axis as the number of training accuracies
    x_axis = range(1, len(acc_train)+1)

    plt.plot(
        x_axis,
        # training accuracy list drew with red solid line
        [acc * 100 for acc in acc_train],'r-',
    )
    plt.legend(['Training Accuracy'])
    plt.xlabel('iterations')
    plt.ylabel('Accuracy of %')
    # save the plot as a png file
    plt.savefig('train_Accuracy.png')
    print("train_Accuracy.png is created successfully!")
    plt.show()

    plt.plot(
        x_axis,
        # testing accuracy list drew with blue solid line
        [acc * 100 for acc in acc_test],'b-'
    )
    plt.legend(['Testing Accuracy'])
    plt.xlabel('iterations')
    plt.ylabel('Accuracy of %')
    # save the plot as a png file
    plt.savefig('test_Accuracy.png')
    print("test_Accuracy.png is created successfully!")
    plt.show()
    
    plt.plot(
        x_axis,
        # training accuracy list drew with red solid line
        [acc * 100 for acc in acc_train],'r-',
        x_axis,
        # testing accuracy list drew with blue solid line
        [acc * 100 for acc in acc_test],'b-'
    )
    plt.legend(['Training Accuracy','Testing Accuracy'])
    plt.xlabel('iterations')
    plt.ylabel('Accuracy of %')
    # save the plot as a png file
    plt.savefig('train_vs_test_Accuracy.png')
    print("train_vs_test_Accuracy.png is created successfully!")
    plt.show()

    return 0


if __name__ == '__main__':

    # read file first
    training_data = pd.read_csv(args.usps_train, header=None)
    test_data = pd.read_csv(args.usps_test, header=None)
    learning_rate = float(args.learningrate)

    # chceck csv is loaded correctly
    print(training_data)
    print(test_data)
    # print the statistical detials of train dataset (trainset: 1400 x 257 , testset: 400 x 257)
    print(training_data.describe())
    print(test_data.describe())
    
    #create train and test matrices
    # X is attribute, Y is result
    X_train, Y_train = create_matrix(training_data)
    X_test, Y_test = create_matrix(test_data)

    #initialize weight vector and accuracy lists
    w = np.array([float(0)] * X_train.shape[1])
    acc_train = []
    acc_test = []

    #batch learning for logistic regression
    batch_learning(X_train, Y_train, X_test, Y_test, w, learning_rate, acc_train, acc_test)

    #drawing accuracy plot with pyplot
    draw_plot(acc_train, acc_test)