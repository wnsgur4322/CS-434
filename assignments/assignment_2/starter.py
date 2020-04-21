# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re

def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def create_bow(input):
    '''
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        max_features=2000
    )
    '''
    vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = 2000) 
    bag_of_words = vectorizer.fit_transform(input)
    #bag_of_words = bag_of_words.toarray()

    #print(bag_of_words[0][0], bag_of_words[0][1], bag_of_words[1][0])
    print(bag_of_words)
    name = vectorizer.get_feature_names()
    print(np.shape(name))
    print(np.shape(bag_of_words))

    return bag_of_words, name

if __name__ == "__main__":
    # Importing the dataset
    imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

    # divide the dataset into train, test, and validation
    imdb_data = imdb_data.to_numpy()
    imdb_data = imdb_data.tolist()

    print(imdb_data[0][0])
    
    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data[0])
    print(len(imdb_data[0]))


    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]
    print(vocabulary)
'''
    bow, name = create_bow(vocabulary)
    bow = np.sum(bow, axis=0)
    print(bow.shape)
    #print(name)
    model = pd.DataFrame( 
        (count, word) for word, count in
        zip(bow.T, name))
    model.columns = ['Word', 'Count']
    #model.sort_values('Count', ascending=False, inplace=True)
    model.head()
    print(model)
    #model['voca'] = name
    #model['count'] = bow[1]

    #model.sort_values(by=['count'], ascending=False, inplace=True)
    #print(model.head(10))

'''