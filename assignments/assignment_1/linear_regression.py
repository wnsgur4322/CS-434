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
from matplotlib import pyplot as plt

if __name__ == "__main__":
        # read csv file with pandas library (put file path exactly)
        housing_test = pd.read_csv('housing_test.csv') 
        housing_train = pd.read_csv('housing_train.csv', header=None)

        # add column name with features
        housing_train.rename(columns={0:'CRIM', 1:'ZN', 2:'INDUS', 3:'CHAS', 4:'NOX', 5:'RM', 6:'AGE',
         7:'DIS', 8:'RAD', 9:'TAX', 10:'PTRATIO', 11:'B', 12:'LSTAT', 13:'MEDV'}, inplace=True)
        #housing_train.to_csv('housing_train.csv', index=False)

        print(housing_train)
        # print the statistical detials of train dataset 
        print(housing_train.describe())

        #train dataset 2D graph plot
        housing_train.plot(x='CRIM', y='MEDV', style='o')
        plt.title('median value of housing (attr 14.) vs Crime rate ')
        plt.xlabel('CRIM')
        plt.ylabel('MEDV')
        plt.savefig('CRIM vs MEDV.png')
        plt.show()

