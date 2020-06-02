# CS 434 - Spring 2020
# Competition - Twitter Sentiment Extaction
# Team members
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu
# Haewon Cho, choha@oregonstate.edu

import pandas as pd
import numpy as np
import re
import string


# https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


