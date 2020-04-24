# CS 434 - Spring 2020
# Team members - Group 33
# Junhyeok Jeong, jeongju@oregonstate.edu
# Youngjoo Lee, leey3@oregonstate.edu

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import math

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

# apply the multi-nomial Naive Bayes classifier with Laplace smooth (a = 1)
def conditional_probability(model_type, total_num, index, pos_neg):
    # formula: (the number of words in class(pos or neg) + Laplace smooth (1)) / (total number of words in class + bag of words size (2000))
    res = float((model_type['Count'][index] + 1) / (total_num + 2000))
    print("P(%dth word | %s) : %f" % (index + 1, pos_neg, res))
    return res

from functools import reduce
# Question 3 part
def multiply(arr):
    return reduce(lambda x, y: x * y, arr)

def MNB_valid(sentence, pos_CP, neg_CP, train_pos_prob, train_neg_prob):
        #res = []
        word_count = []
        
        for j in range(len(valid_model['Word'])):
            #count = 0
            
            '''
            for h in range(len(sentence)):
                if sentence[h] == valid_model['Word'][j]:
                    count += 1

            '''
            #word_count.append(count)
            word_count.append(sentence.count(valid_model['Word'][j]))

        print(len(word_count))


        pos_pow_list = [math.log(wi) * n for wi, n in zip(pos_CP, word_count)]
        pos_pow_list = list(filter((0.0).__ne__, pos_pow_list))

        neg_pow_list = [math.log(wi) * n for wi, n in zip(neg_CP, word_count)]
        neg_pow_list = list(filter((0.0).__ne__, neg_pow_list))
        # P(Positive | Validation reviews) = train_pos_prob * pos_CP[word_1]^n * pos_CP[word_2] ....
        
        #pos_res = math.log(train_pos_prob) + reduce(lambda x, y: x + y, pos_pow_list)
        pos_res = train_pos_prob * multiply(pos_pow_list)
        print(pos_res)
        # P(Negative | Validation reviews) = train_neg_prob * neg_CP[word_1]^n * neg_CP[word_2] ....
        #neg_res = math.log(train_neg_prob) + reduce(lambda x, y: x + y, neg_pow_list)
        neg_res = train_neg_prob * multiply(neg_pow_list)
        print(neg_res)
        

        if pos_res > neg_res:
            return("positive")
            #res.append("positive")
        else:
            return("negative")
            #res.append("negative")

if __name__ == "__main__":
    # Importing the dataset
    imdb_data = pd.read_csv('IMDB.csv', delimiter=',')
    label_data = pd.read_csv('IMDB_labels.csv', delimiter=',')

    vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000
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
    neg_CP = []
    total_neg = total_num(train_neg, "Negative")
    for i in range(2000):
        neg_CP.append(conditional_probability(train_model, total_neg, i, "Negative"))

    # 3-1. Apply the learned Naive Bayes model to the validation set
    # P(Positive | Validation review 1) = train_pos_prob * pos_CP[word_1]^n * pos_CP[word_2] ....

    print("generating BOW for 10k (validation set) reviews ...")
    bow_valid, name_valid = create_bow(cleaned_text[30000:40000], vocabulary)
    bow_valid = np.sum(bow_valid, axis=0)

    valid_model = pd.DataFrame( 
    (count, word) for word, count in
    zip(bow_valid, name_valid))
    valid_model.columns = ['Word', 'Count']
    print(valid_model)
    print(len(valid_model))
    print("done ... !")

    sentences = []
    for i in range(len(valid_set)):
        valid_set[i] = clean_text(valid_set[i])
        sentence = valid_set[i].split(" ")
        sentence = list(filter(('').__ne__, sentence))
        sentences.append(sentence)
        print(i)
    
    print("done .. !")        
    
    #print(valid_label)

    #sys.end(1)
    acc = 0
    validation_res = []
    for i in range(len(valid_set)):
        validation_res.append(MNB_valid(sentences[i], pos_CP, neg_CP, train_pos_prob, train_neg_prob))
        print(i)
    #print(validation_res)

    for j in range(len(validation_res)):
        if validation_res[j] == valid_label[j]:
            acc += 1
    print("validation accuracy(%): %f" % float(acc/100))
                