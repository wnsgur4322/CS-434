# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

'''
(20 pts) Modify the data by adding additional random features. You will do this to both training
and testing data. In particular, generate 20 random features by sampling from a standard normal
distribution. Incrementally add the generated random features to your data, 2 at a time. So we will
create 20 new train/test datasets, each with d of random features, where d = 2, 4, ..., 20. For each
version, learn the optimal linear regression model (i.e., the optimal weight vector) and compute its
resulting training and testing ASEs. Plot the training and testing ASEs as a function of d. What
trends do you observe for training and testing ASEs respectively? In general, how do you expect
adding more features to influence the training ASE? How about testing ASE? Why?
'''

import math
import sys
import numpy as np
import pandas as pd
import argparse as arg
from matplotlib import pyplot as plt

# command line prompt with argparse library
parser = arg.ArgumentParser()
parser.add_argument('housing_train')
parser.add_argument('housing_test')
args = parser.parse_args()

# to caculate the optimal weight vector based on formula w = (((X^T)*X)^-1)*((X^T)*Y) 
def weight_vector(X,Y):
    return ((X.T * X).I) * (X.T * Y)

# to calculate the average squared error(ASE) which is the sum of squared error normalized by n
def ASE(attributes, MEDVs, w_vector):
    SSE = 0
    y_hat = attributes * w_vector    

    for i in range(len(attributes)):
        SSE = SSE + ((MEDVs[i] - y_hat[i]) ** 2)
    
    return SSE/len(attributes)

def create_matrix(data, d):
    # convert the pandas dataframe into numpy array matrix
    data_matrix = data.to_numpy()
    data_matrix = data_matrix.tolist()

    X = []
    Y = []
    for i in range(len(data_matrix)):

        # X is for attributes
        X.append(data_matrix[i][:-1])
        # add d random features by sampling from a standard normal distribution with numpy function (numpy.random.standard_normal())
        # numpy API reference: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.standard_normal.html
        for j in range(d):
            X[i].append(np.random.standard_normal())
        # Y is for MEDV values
        Y.append([data_matrix[i][-1]])

    X = np.matrix(X)
    Y = np.matrix(Y)

    return X,Y


# this function is for drawing plot between training and testing ASE depending on the number of randomfeatures
def draw_plot(train_ASEs, test_ASEs, added_features):
    plt.plot(
        # training ASE drew with red solid line
        added_features,train_ASEs,'r-',
        # testing ASE drew with blue solid line
        added_features,test_ASEs,'b-'
    )
    plt.legend(['Training ASE','Testing ASE'])
    plt.xlabel('The number of random features')
    plt.ylabel('Average Squared Error')
    # save the plot as a png file
    plt.savefig('train_vs_test_ASE.png')
    print("train_vs_test_ASE.png is created successfully!")
    plt.show()

    return 0

if __name__ == "__main__":
        # read csv file with pandas library (put file path exactly)
        housing_test = pd.read_csv(args.housing_test, header=None) 
        housing_train = pd.read_csv(args.housing_train, header=None)

        # add column name with features
        housing_train.rename(columns={0:'CRIM', 1:'ZN', 2:'INDUS', 3:'CHAS', 4:'NOX', 5:'RM', 6:'AGE',
         7:'DIS', 8:'RAD', 9:'TAX', 10:'PTRATIO', 11:'B', 12:'LSTAT', 13:'MEDV'}, inplace=True)
        
        housing_test.rename(columns={0:'CRIM', 1:'ZN', 2:'INDUS', 3:'CHAS', 4:'NOX', 5:'RM', 6:'AGE',
         7:'DIS', 8:'RAD', 9:'TAX', 10:'PTRATIO', 11:'B', 12:'LSTAT', 13:'MEDV'}, inplace=True)

        # chceck csv is loaded correctly
        print(housing_train)
        print(housing_test)
        # print the statistical detials of train dataset 
        print(housing_train.describe())
        print(housing_test.describe())

        # looping til d = 2, 4, ... 20 and appending each version's ASE
        train_ASEs = []
        test_ASEs = []
        # d is from 2 to 20 with step 2
        for d in range(2, 22, 2):
            #create each version train and test matrices depending on d value
            X_train, Y_train = create_matrix(housing_train, d)
            X_test, Y_test = create_matrix(housing_test, d)
            print("Created train and test matrices with %d of random features" % d)

            #calculate and print out weight vector
            w = weight_vector(X_train, Y_train)
            print("The learned weight vector with dummy variable: {0}".format([i[0,0] for i in w]))

            #ASE part
            print("The ASE of training data: %f" % ASE(X_train, Y_train, w))
            print("THe ASE of testing data: %f " % ASE(X_test, Y_test, w))

            #append each version's ASE value in the list
            train_ASEs.append(float(ASE(X_train, Y_train, w)))
            test_ASEs.append(float(ASE(X_test, Y_test, w)))
        
        print("\n-- Answer part --")
        print("\n the training ASE's list: {0}".format(train_ASEs))
        print("\n the testing ASE's list: {0}".format(test_ASEs))
        
        draw_plot(train_ASEs, test_ASEs, range(2, 22, 2))

