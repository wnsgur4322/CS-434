CS 434 - Spring 2020
implmentation assignment 1
Team members
Junhyeok Jeong, jeongju@oregonstate.edu
Youngjoo Lee, leey3@oregonstate.edu

1. To compile the python files in this repository, you should set up virtual environment first with below commands

cd ~/CS-434
bash
virtualenv venv -p $(which python3)
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install numpy pandas matplotlib sklearn mpmath

2. Check library versions with library_check.py
python3 library_check.py

- then you should see this prompts on the terminal
numpy version: 1.18.2 (or other version)
pandas version: 1.0.3 (or other version)
matplotlib version: 3.2.1 (or other version)


3. How to compile question 1-1 ~ 1-2 part file
python3 q1_2.py housing_train.csv housing_test.csv

# make sure to place the data files on the same directory with python file

4. How to compile question 1-3 part file
python3 q1_3.py housing_train.csv housing_test.csv

# make sure to place the data files on the same directory with python file

5. How to compile question 1-4 part file
python3 q1_4.py housing_train.csv housing_test.csv

# make sure to place the data files on the same directory with python file 