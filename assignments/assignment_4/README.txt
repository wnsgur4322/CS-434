CS 434 - Spring 2020
implmentation assignment 4
Team members
Junhyeok Jeong, jeongju@oregonstate.edu
Youngjoo Lee, leey3@oregonstate.edu
Haewon Cho, choha@oregonstate.edu

1. To compile the python files in this repository, you should set up virtual environment first with below commands

cd ~/CS-434
bash
virtualenv venv -p $(which python3)
source ./venv/bin/activate
pip3 install --upgrade pip
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install seaborn

2. Check library versions with library_check.py
python3 library_check.py

- then you should see this prompts on the terminal
numpy version: 1.18.2 (or other version)
pandas version: 1.0.3 (or other version)
matplotlib version: 3.2.1 (or other version)
seaobrn version: 0.10.1 (or other version)

3. How to compile k-means with PCA

python3 main.py --pca 1 --kmeans 1

4. How to compile k-means without PCA

python3 main.py --pca 0 --kmeans 1

5. How to change PCA retain ratio

python3 main.py --pca 1 --kmeans 1 --pca_retain_ratio (your rate)

