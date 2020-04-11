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

'''
# sigmoid function (L(w) - for q2)
def logistic_function(w, x):
    return 1.0/(1 + np.exp(-np.dot(x, w.T))) #(700,1) or (400, 1)

def logistic_gradient(w, x, y):
    #print(y.reshape(x.shape[0], -1))
    temp = logistic_function(w, x) - y
    result = np.dot(temp.T, x)
    return result


# sigmoid function (L(w) - for q2)
def logistic_function(w, x):
    
    row, col = x.shape
    result = np.zeros((row, col))
    #print(result.shape)
    for i in range(0,row-1):
        result[i] = 1.0/(1 + np.exp(-np.dot(x[i], w.T)))

    print(result)
    return result

def logistic_gradient(w, x, y):
    row, col = x.shape
    yhat = np.zeros((row,1))
    gradient = np.zeros((row,1))
    for i in range(0, row-1):
        yhat = 1.0/(1 + np.exp(-np.dot(x[i], w.T)))
        gradient = gradient + np.dot((yhat - y[i]),x[i])
    print(gradient)
    return gradient


def learning_w(w, x, y):
    logistic_fun = logistic_function(w, x)
    y = np.squeeze(y)

    result = -(y*np.log(logistic_fun)) - ((1-y)*np.log(1-logistic_fun))
    return result

def batch_grad_desc(x, y, w, learning_r, eps, x_type, y_type):

    iteration = 1
    index = []
    accuracy = []

    condition = True
    #cost = learning_w(w, x, y)
    check = 1
    row, col = w.shape

    gradient = np.zeros((row,col))

    while condition:
        row, col = w.shape
        gradient = np.zeros((row,col))
        #temp = cost
        gradient = logistic_gradient(w, x, y)
        w = w - (0.01*gradient)
        #cost = learning_w(w, x, y)
        #check = temp - cost
        temp_accuracy = prediction(w, x_type, y_type)
        index.append(iteration)
        accuracy.append(temp_accuracy)
        #print(gradient*gradient.T, math.sqrt(gradient*gradient.T))
        #if(math.sqrt(np.sum(gradient**2, axis=1))) <= eps:
        if (math.sqrt(gradient*gradient.T)) <= eps:
            condition = False
        elif iteration >= 500:
            condition = False
        iteration += 1
    

    while condition:
        gradient = 0
        temp = cost
        gradient = (0.01*logistic_gradient(w, x, y))
        w = w - (0.01*logistic_gradient(w, x, y))
        cost = learning_w(w, x, y)
        check = temp - cost
        temp_accuracy = prediction(w, x_type, y_type)
        index.append(iteration)
        accuracy.append(temp_accuracy)
        if abs(check) > eps:
            condition = False
        elif iteration >= 500:
            condition = False
        iteration += 1

    return w, iteration, index, accuracy




def prediction(w, x_type, y_type):
    pred_prob = logistic_function(w, x_type)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    result = np.matrix(np.squeeze(pred_value)).T
    return np.sum(y_type == result)/np.size(y_type) * 100
'''

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
    
    # added by JH
    #create each version train and test matrices depending on d value
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

    # fin
    '''
    x_train = np.matrix(training_data.iloc[:, :256])
    y_train = np.matrix(training_data.iloc[:, 256]).T #labels

    train_x_train, train_x_test, train_y_train, train_y_test = train_test_split(x_train, y_train, test_size=0.5, random_state=42)


    x_test = np.matrix(test_data.iloc[:, :256])
    y_test = np.matrix(test_data.iloc[:, 256]).T #labels

    test_x_train, test_x_test, test_y_train, test_y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    
    #initial w value
    w_train = np.matrix(np.zeros(train_x_train.shape[1]))
    w_test = np.matrix(np.zeros(test_x_train.shape[1]))

    w, iteration, index, accuracy = batch_grad_desc(train_x_train, train_y_train, w_train, learningrate, 0.001, train_x_test, train_y_test)

    m = len(accuracy)
    print("Accuracy of training(%): ", accuracy[m-1], " / Size of iteration: ", m)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Training accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()

    w_test, iteration, index, accuracy = batch_grad_desc(test_x_train, test_y_train, w_test, learningrate, 0.001, test_x_test, test_y_test)
    
    m = len(accuracy)
    print("Accuracy of test(%): ", accuracy[m-1], " / Size of iteration: ", m)
    plt.plot(index,accuracy)
    plt.title("Plot of the learning curve (Test accuracy)")
    plt.xticks(np.arange(0, 500, step = 100), rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Accuracy (%)")
    plt.show()
    '''