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
parser.add_argument('learning_rate')
args = parser.parse_args()

# result of 0 or 1 == (o(w^T*x)) 
# hypothesis
# pass
def logistic_function(w, x):
    return 1/(1 + np.exp(-np.dot(x, w.T))) # x -> (700 or 400 ,256) / w -> (1,256) / w.T (256,1)
    # result => (700,1) -> each column has 1 or 0

#loss function == l()
#cost : needs to work with hypthosis
# likelihood
def learning_w(w, x, y, lamb):
    #logistic_fun = logistic_function(w, x)
    #y = np.squeeze(y) # Transpose?
    result = np.zeros((len(y), 1)) #700,1
    # Learning w for logistic regression (L(w))
    for i in range(0, len(y)): 
        result[i] = -(y[i]*np.log(logistic_function(w, x[i]))) - ((1-(y[i]))*np.log(1-logistic_function(w, x[i])))

    return np.mean(result)

# gradietn (derivate) of Loss function l() == inverted triangel with l()
def logistic_gradient(w, x, y):
    #print(y.reshape(x.shape[0], -1))
    temp = np.zeros((len(y),1))
    result = np.zeros((len(y),256))
    gradient = np.zeros((1, 256))
    result = np.dot((logistic_function(w, x) - y).T, x) #reverse traingle of l (page on Gradient of L(w))
        #print(logistic_function(w, x[i]).shape, y[i].shape, x[i].shape)
        #print(np.dot((logistic_function(w, x[i]) - y[i]), x[i]).shape)
        #result[i] = (1, 256)  /// (logistic_function(w, x[i]) - y[i]) = 1,256
    return result
    #reulst = 700,256

#batch learning of logistic regression
def batch_grad_desc(x, y, w, learning_r, eps, x_type, y_type, lamb):

    iteration = 1
    index = []
    accuracy = []
    condition = True
    
    #initialize w
    w = np.zeros((1, 256))

    while condition:
        gradient = np.zeros((1,256))
        gradient = logistic_gradient(w,x,y)
        w = w - (learning_r*gradient) # w - learning_rate*gradient - update

        temp_accuracy = prediction(w, x_type, y_type)
        index.append(iteration)
        accuracy.append(temp_accuracy)

        if (math.sqrt(np.dot(gradient,gradient.T)[0][0])) <= 0.0001:
            condition = False
            iteration -= 1
        elif iteration >= 500:
            condition = False
            iteration -= 1
        iteration += 1

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
    learningrate = float(args.learning_rate)
    
    x_train = np.matrix(training_data.iloc[:, :256])
    y_train = np.matrix(training_data.iloc[:, 256]).T #labels

    train_x_train, train_x_test, train_y_train, train_y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)


    x_test = np.matrix(test_data.iloc[:, :256])
    y_test = np.matrix(test_data.iloc[:, 256]).T #labels

    test_x_train, test_x_test, test_y_train, test_y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    #initial w value
    w_train = np.matrix(np.zeros(train_x_train.shape[1]))
    w_test = np.matrix(np.zeros(test_x_train.shape[1]))

    w, iteration, index, accuracy = batch_grad_desc(train_x_train, train_y_train, w_train, learningrate, 0.0001, train_x_test, train_y_test, 0.001)

    m = len(accuracy)
    print("Accuracy of training(%): ", accuracy[m-1], " / Size of iteration: ", iteration)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Training accuracy)")
    #plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()

    w2, iteration, index, accuracy = batch_grad_desc(test_x_train, test_y_train, w_test, learningrate, 0.0001, test_x_test, test_y_test, 0.001)
    
    m = len(accuracy)
    print("Accuracy of test(%): ", accuracy[m-1], " / Size of iteration: ", iteration)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Test accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()