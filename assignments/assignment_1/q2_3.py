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
parser.add_argument('lamb_t')
args = parser.parse_args()

#nomalization
def normalize(x): 
    mins = np.min(x, axis = 0) 
    maxs = np.max(x, axis = 0) 
    rng = maxs - mins 
    norm = 1 - ((maxs - x)/rng) 
    return norm 

# sigmoid function (L(w) - for q2)
# result of 0 or 1 == (o(w^T*x)) 
def logistic_function(w, x):
    return 1.0/(1 + np.exp(-np.dot(x, w.T)))

# gradietn (derivate) of Loss function l() == inverted triangel with l()
def logistic_gradient(w, x, y):
    temp = logistic_function(w, x) - y #compare with y (label)
    result = np.dot(temp.T, x) #reverse traingle of l (page on Gradient of L(w))
    return result

#loss function == l()
#cost : needs to work with hypthosis
def learning_w(w, x, y, lamb):
    logistic_fun = logistic_function(w, x)
    y = np.squeeze(y)

    # Learning w for logistic regression (L(w))
    result = -(y*np.log(logistic_fun)) - ((1-y)*np.log(1-logistic_fun))
    return result + 1/2*(lamb)*(w*w.T)

#batch learning of logistic regression
def batch_grad_desc(x, y, w, learning_r, eps, x_type, y_type, lamb):

    iteration = 1
    index = []
    accuracy = []
    cost_all = []

    condition = True
    cost = 0
    check = 1

    while condition:
        temp = cost
        w = w - (learning_r*logistic_gradient(w, x, y))
        cost = learning_w(w, x, y, lamb)
        check = temp - cost
        cost_all.append(check)
        temp_accuracy = prediction(w, x_type, y_type, lamb)
        index.append(iteration)
        accuracy.append(temp_accuracy)
        if np.sqrt((logistic_gradient(w, x, y)*logistic_gradient(w, x, y).T)) <= eps:
            condition = False
        elif iteration >= 500:
            condition = False
        iteration += 1
    return w, iteration, index, accuracy, cost_all


def prediction(w, x_type, y_type, lamb):
    pred_prob = logistic_function(w, x_type)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    result = np.matrix(np.squeeze(pred_value)).T
    return np.sum(y_type == result)/np.size(y_type) * 100


if __name__ == '__main__':

    # read file first
    training_data = pd.read_csv(args.usps_train, header=None)
    test_data = pd.read_csv(args.usps_test, header=None)
    lamb = float(args.lamb_t)
    
    x_train = np.matrix(training_data.iloc[:, :256])
    x_train = normalize(x_train)
    y_train = np.matrix(training_data.iloc[:, 256]).T #labels

    train_x_train, train_x_test, train_y_train, train_y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)


    x_test = np.matrix(test_data.iloc[:, :256])
    x_test = normalize(x_test)
    y_test = np.matrix(test_data.iloc[:, 256]).T #labels

    test_x_train, test_x_test, test_y_train, test_y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    #initial w value
    w_train = np.matrix(np.zeros(train_x_train.shape[1]))
    w_test = np.matrix(np.zeros(test_x_train.shape[1]))

    w, iteration, index, accuracy, cost = batch_grad_desc(train_x_train, train_y_train, w_train, 0.001, 0.001, train_x_test, train_y_test, lamb)

    m = len(accuracy)
    mean_acc = np.sum(accuracy)/m
    print("Accuracy of test(%): ", accuracy[m-1], " / Mean accuracy: ", mean_acc, " / Size of iteration: ", m)
    
    #print plot for training
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Training accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.savefig('train_Accuracy_lamb.png')
    print("train_Accuracy.png is created successfully!")
    plt.show()
    

    w_test, iteration, index, accuracy, cost = batch_grad_desc(test_x_train, test_y_train, w_test, 0.001, 0.001, test_x_test, test_y_test, lamb)
    
    m = len(accuracy)
    mean_acc = np.sum(accuracy)/m
    print("Accuracy of test(%): ", accuracy[m-1], " / Mean accuracy: ", mean_acc, " / Size of iteration: ", m)

    #print plot for test
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Test accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.savefig('test_Accuracy_lamb.png')
    print("test_Accuracy.png is created successfully!")
    plt.show()