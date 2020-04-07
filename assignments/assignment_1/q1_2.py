# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

'''
4. Relevant Information:

   Concerns housing values in suburbs of Boston.

5. Number of Instances: 506

6. Number of Attributes: 13 continuous attributes (including "class"
                         attribute "MEDV"), 1 binary-valued attribute.

7. Attribute Information:

    1. CRIM      per capita crime rate by town
    2. ZN        proportion of residential land zoned for lots over
                 25,000 sq.ft.
    3. INDUS     proportion of non-retail business acres per town
    4. CHAS      Charles River dummy variable (= 1 if tract bounds
                 river; 0 otherwise)
    5. NOX       nitric oxides concentration (parts per 10 million)
    6. RM        average number of rooms per dwelling
    7. AGE       proportion of owner-occupied units built prior to 1940
    8. DIS       weighted distances to five Boston employment centres
    9. RAD       index of accessibility to radial highways
    10. TAX      full-value property-tax rate per $10,000
    11. PTRATIO  pupil-teacher ratio by town
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
                 by town
    13. LSTAT    % lower status of the population
    14. MEDV     Median value of owner-occupied homes in $1000's
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
            # put dummy variable 1 at index 0 
            train_matrix[i].insert(0,1)
            # X is for attributes
            X.append(train_matrix[i][:-1])
            # Y is for MEDV values
            Y.append([train_matrix[i][-1]])
        X = np.matrix(X)
        Y = np.matrix(Y)

        print("\n-- Answer part --")
        #print out weight vector
        w = weight_vector(X,Y)
        print("The learned weight vector with dummy variable: {0}".format([i[0,0] for i in w]))

        X_test = []
        Y_test = []
        for i in range(len(test_matrix)):
            # put dummy variable 1 at index 0 
            test_matrix[i].insert(0,1)
            # X is for attributes
            X_test.append(test_matrix[i][:-1])
            # Y is for MEDV values
            Y_test.append([test_matrix[i][-1]])
        X_test = np.matrix(X_test)
        Y_test = np.matrix(Y_test)

        #ASE part
        print("The ASE of training data: %f" % ASE(X, Y, w))
        print("THe ASE of testing data: %f " % ASE(X_test, Y_test, w))





#drawing code with matplot (maybe use later)
'''
        #train dataset 2D graph plot
        X = housing_train['CRIM'].values.reshape(-1,1)
        Y = housing_train['MEDV'].values.reshape(-1,1)
        
        reg = LinearRegression()
        reg.fit(X, Y)
        print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))

        predictions = reg.predict(X)

        plt.figure(figsize=(16, 8))
        plt.scatter(
            housing_train['CRIM'],
            housing_train['MEDV'],
            c='black'
        )
        plt.plot(
            housing_train['CRIM'],
            predictions,
            c='blue',
            linewidth=2        
        )
        plt.title('median value of housing (attr 14.) vs Crime rate ')
        plt.xlabel('CRIM')
        plt.ylabel('MEDV')
        plt.savefig('CRIM vs MEDV.png')
        plt.show()
'''
