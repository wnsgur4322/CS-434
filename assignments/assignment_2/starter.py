# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter

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
        max_features=2000
    )
    bag_of_words = vectorizer.fit_transform(input)
    #np.asarray(bag_of_words)
    bag_of_words = bag_of_words.toarray()
    
    name = vectorizer.get_feature_names()

    return bag_of_words, name

# get the probability of a word from positive or negative reviews
# ex) 'abc' occured 100 times totally. if x is the occurance in positive reviews,
# then x/100 is the positive probability of 'abc' word
def word_prob(model_type, reviews, word, index):
    count = 0

    for i in range(len(reviews)):
        review = list(map(str, clean_text(reviews[i]).split(" ")))
        for j in range(len(review)):
            if review[j] == word:
                count += 1
    return float(count/model_type['Count'][index])

# To calculate the total number of words in reviews (positive or negative)
def total_num(reviews, pos_neg):
    res = 0
    for i in range(len(reviews)):
        review = list(map(str, clean_text(reviews[i]).split(" ")))
        review = list(filter(('').__ne__, review))
        res += len(review)

    print("The total number of words in all %s reviews: %d" % (pos_neg,res))
    return res

def conditional_probability(model_type, total_num, index, pos_neg):
    # formula: (the number of words in class(pos or neg) + Laplace smooth (1)) / (total number of words in class + bag of words size (2000))
    res = float((model_type['Count'][index] + 1) / (total_num + 2000))
    print("P(%dth word | %s) : %f" % (index + 1, pos_neg, res))
    return res

if __name__ == "__main__":
    # Importing the dataset
    imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
    label_data = pd.read_csv('IMDB_labels.csv', delimiter=',')

    vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text
    )

    # fit the vectorizer on the text
    vectorizer.fit(imdb_data['review'])
    # get the vocabulary
    inv_vocab = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [inv_vocab[i] for i in range(len(inv_vocab))]

    # 1. generate BOW for all 50k reviews
    print("generating BOW for 50k reviews ...")
    cleaned_text = []
    num_texts = imdb_data['review'].size
    for i in range(0, num_texts):
        cleaned_text.append(clean_text(imdb_data['review'][i]))

    bow, name = create_bow(cleaned_text, vocabulary)
    bow = np.sum(bow, axis=0)
    print(bow[0])

    model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow, name))
    model.columns = ['Word', 'Count']
    #model.sort_values('Count', ascending=False, inplace=True)
    print(model)
    print("done ... !")

    print("generating BOW for 30k (training set) reviews ...")
    bow_train, name_train = create_bow(cleaned_text[:30000], vocabulary)
    bow_train = np.sum(bow_train, axis=0)

    train_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow_train, name_train))
    train_model.columns = ['Word', 'Count']
    print(train_model)
    print(len(train_model))
    print("done ... !")

    # divide the dataset into train, test, and validation
    train_set = list(imdb_data['review'][:30000])
    #train_set = imdb_data.iloc[:30000]
    valid_set = list(imdb_data['review'][30000:40000])
    #valid_set = imdb_data.iloc[30000:40000]
    test_set = list(imdb_data['review'][40000:])
    
    print("the lengths of three data set: train_set = %d, valid_set = %d, test_set = %d" % (len(train_set), len(valid_set), len(test_set)))

    # apply reveiw label on training and validation sets
    train_label = list(label_data["sentiment"][:30000])
    valid_label = list(label_data["sentiment"][30000:])

    # separate reviews based on label data
    train_pos = [train_set[i] for i in range(len(train_label)) if train_label[i] == "positive" ]
    train_neg = [train_set[i] for i in range(len(train_label)) if train_label[i] == "negative" ]

    # 2. Train a multi-nomial Naive Bayes classifier with Laplace smooth with a = 1 on the training set

    # Priors: training set's positive and negative probabilities
    train_pos_prob = len(train_pos) / len(train_set)
    train_neg_prob = len(train_neg) / len(train_set)

    print("probabilities of training set\npositive: %f\nnegative: %f" % (train_pos_prob, train_neg_prob))

    print("the positive probability of the word '%s' is %f " % (model['Word'][0], word_prob(train_model, train_pos, train_model['Word'][0], 0)))
    print("the negative probability of the word '%s' is %f " % (model['Word'][0], word_prob(train_model, train_neg, train_model['Word'][0], 0)))

    # https://www.youtube.com/watch?v=km2LoOpdB3A
    # Conditional Probabilities: P(Wi...2000|Positive) & P(Wi...2000|Negative)
    # formula: (the number of words in class(pos or neg) + Laplace smooth (1)) / (total number of words in class + bag of words size (2000))
    # P(Wi...2000|Positive) part
    pos_CP = []
    total_pos = total_num(train_pos, "Positive")
    for i in range(2000):
        pos_CP.append(conditional_probability(train_model, total_pos, i, "Positive"))
    
    #P(Wi...2000|Negative) Part
    Neg_CP = []
    total_neg = total_num(train_neg, "Negative")
    for i in range(2000):
        Neg_CP.append(conditional_probability(train_model, total_neg, i, "Negative"))




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