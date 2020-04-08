# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

'''
1-3. (10 pts) Remove the dummy variable (the column of ones) from X, repeat 1 and 2. How does this
change influence the ASE on the training and testing data? Provide an explanation for this influence.
'''

import math
import sys
import numpy as np
import pandas as pd
import argparse as arg

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

        # convert the pandas dataframe into numpy array matrix
        train_matrix = housing_train.to_numpy()
        train_matrix = train_matrix.tolist()

        test_matrix = housing_test.to_numpy()
        test_matrix = test_matrix.tolist()

        X = []
        Y = []
        for i in range(len(train_matrix)):
            # without dummy column for question 1-3 
            # train_matrix[i].insert(0,1)
            # X is for attributes
            X.append(train_matrix[i][:-1])
            # Y is for MEDV values
            Y.append([train_matrix[i][-1]])
        X = np.matrix(X)
        Y = np.matrix(Y)

        print("\n-- Answer part --")
        #print out weight vector
        w = weight_vector(X,Y)
        print("The learned weight vector without dummy variable: {0}".format([i[0,0] for i in w]))

        X_test = []
        Y_test = []
        for i in range(len(test_matrix)):
            # without dummy column for question 1-3  
            #test_matrix[i].insert(0,1)
            # X is for attributes
            X_test.append(test_matrix[i][:-1])
            # Y is for MEDV values
            Y_test.append([test_matrix[i][-1]])
        X_test = np.matrix(X_test)
        Y_test = np.matrix(Y_test)

        #ASE part
        print("The ASE of training data: %f" % ASE(X, Y, w))
        print("THe ASE of testing data: %f " % ASE(X_test, Y_test, w))
