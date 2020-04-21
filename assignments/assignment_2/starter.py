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

def create_bow(input, vocabulary):
    vectorizer = CountVectorizer(
        analyzer = "word",
        tokenizer=None,
        preprocessor=None,
        stop_words=None,
        #vocabulary=vocabulary,
        max_features=1000
    )
    bag_of_words = vectorizer.fit_transform(input)
    #np.asarray(bag_of_words)
    bag_of_words = bag_of_words.toarray()
    
    name = vectorizer.get_feature_names()

    return bag_of_words, name

if __name__ == "__main__":
    # Importing the dataset
    imdb_data = pd.read_csv('IMDB.csv', delimiter=',')

    # divide the dataset into train, test, and validation
    #imdb_data = imdb_data.to_numpy()
    #imdb_data = imdb_data.tolist()

    vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]


    cleaned_text = []
    num_texts = imdb_data['review'].size
    for i in range(0, num_texts):
        cleaned_text.append(clean_text(imdb_data['review'][i]))

    

    bow, name = create_bow(cleaned_text, vocabulary)
    #bow = np.sum(bow, axis=0)



    model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow, name))
    model.columns = ['Word', 'Count']
    #model.sort_values('Count', ascending=False, inplace=True)
    model.head(10)
    print(model, model.head(10))

'''
    print("test")
    for i in range(0, len(imdb_data)):
        cleaned_text += clean_text(imdb_data[i][0])
        #print(cleaned_text)
    print("test2")
    bow, name = create_bow(cleaned_text)
    print("test3")
    model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow, name))
    model.columns = ['Word', 'Count']
    model.sort_values('Count', ascending=False, inplace=True)
    model.head()

'''
    

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