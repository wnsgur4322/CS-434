# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import math
import sys
import numpy as np
import pandas as pd

if __name__ == "__main__":
        # read csv file with pandas library (put file path exactly)
        housing_test = pd.read_csv('housing_test.csv') 
        housing_train = pd.read_csv('housing_train.csv')

        print(housing_train)
        print(housing_test)
