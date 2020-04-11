# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import numpy as np
import pandas as pd
import random
import pylab
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse as arg
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import math

# command line prompt with argparse library
parser = arg.ArgumentParser()
parser.add_argument('usps_train')
parser.add_argument('usps_test')
parser.add_argument('learningrate')
args = parser.parse_args()


# sigmoid function (L(w) - for q2)
def logistic_function(w, x):
    ptive_inf = math.inf
    ntive_inf = -math.inf
    result = 1/(1 + np.exp(-np.dot(x, w.T)))
    #print(result)
    '''
    size = len(result)
    print(size)
    for i in range(0, size-1):
        if result[i] == [ntive_inf]:
            print("test")
            result[i] = 1
            '''
    return result
    #return 1/(1 + np.exp(-np.dot(x, w.T))) #(700,1) or (400, 1)

def logistic_gradient(w, x, y):
    #print(y.reshape(x.shape[0], -1))
    temp = logistic_function(w, x) - y.reshape(x.shape[0], -1)
    result = np.dot(temp.T, x)
    return result

def learning_w(w, x, y, lamb):
    logistic_fun = logistic_function(w, x)
    #print(logistic_fun)
    y = np.squeeze(y)
    #print(np.log(logistic_fun))
    size = len(x) # number of samples for train
    ptive_inf = math.inf
    ntive_inf = -math.inf
    print(1*ptive_inf, 0*ptive_inf, 1*ntive_inf, 0*ntive_inf)
    print(np.log(0.5), np.log(1), np.log(0))

    result = -(y*np.log(logistic_fun)) - ((1-y)*np.log(1-logistic_fun)) + (1/2)*(lamb)*(w*w.T)*size
    return np.sum(result)

def batch_grad_desc(x, y, w, learning_r, eps, x_type, y_type, lamb):

    iteration = 1
    index = []
    accuracy = []

    condition = True
    cost = learning_w(w, x, y, lamb)
    check = 1
    #row, col = w.shape

    #gradient = np.zeros((row,col))

    while condition:
        #row, col = w.shape
        #gradient = np.zeros((row,col))
        temp = cost
        #gradient = logistic_gradient(w, x, y)
        #w = w - (0.01*gradient)
        w = w - (0.01*logistic_gradient(w, x, y))
        cost = learning_w(w, x, y, lamb)
        check = temp - learning_w(w, x, y, lamb)
        temp_accuracy = prediction(w, x_type, y_type)
        index.append(iteration)
        accuracy.append(temp_accuracy)
        if check > eps:
        #if (math.sqrt(gradient*gradient.T)) <= eps:
            condition = False
        elif iteration >= 500:
            condition = False
        elif np.isnan(check) == True:
            condition = False
        iteration += 1
    
    print(temp, cost, check)


    return w, iteration, index, accuracy




def prediction(w, x_type, y_type):
    pred_prob = logistic_function(w, x_type)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    result = np.matrix(np.squeeze(pred_value)).T
    return np.sum(y_type == result)/np.size(y_type) * 100


if __name__ == '__main__':

    # read file first
    training_data = pd.read_csv(args.usps_train, header=None)
    test_data = pd.read_csv(args.usps_test, header=None)
    learningrate = args.learningrate
    
    x_train = np.matrix(training_data.iloc[:, :256])
    y_train = np.matrix(training_data.iloc[:, 256]).T #labels

    train_x_train, train_x_test, train_y_train, train_y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)


    x_test = np.matrix(test_data.iloc[:, :256])
    y_test = np.matrix(test_data.iloc[:, 256]).T #labels

    test_x_train, test_x_test, test_y_train, test_y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    #initial w value
    w_train = np.matrix(np.zeros(train_x_train.shape[1]))
    w_test = np.matrix(np.zeros(test_x_train.shape[1]))

    w, iteration, index, accuracy = batch_grad_desc(train_x_train, train_y_train, w_train, learningrate, 0.001, train_x_test, train_y_test, 0.001)

    m = len(accuracy)
    print("Accuracy of training(%): ", accuracy[m-1], " / Size of iteration: ", m)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Training accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()

    w_test, iteration, index, accuracy = batch_grad_desc(test_x_train, test_y_train, w_test, learningrate, 0.001, test_x_test, test_y_test, 0.001)
    
    m = len(accuracy)
    print("Accuracy of test(%): ", accuracy[m-1], " / Size of iteration: ", m)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Test accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()