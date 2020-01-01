import datetime
import os
import re
import time
from itertools import islice
from operator import itemgetter
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
import nltk
from nltk.corpus import stopwords

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
             "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

vectorizer = CountVectorizer(analyzer="word",
                                tokenizer=None,
                                preprocessor=None,
                                stop_words=None,
                                max_features=10)

def print_words_frequency(train_data_features):
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print ("Words in vocabulary:", vocab)

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)

    # For each, print the vocabulary word and the number of times it
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print (count, tag)

def bagOfWords(filename):
    # Reading the Data
    clean_train_reviews = pd.read_csv(filename, nrows=50)

    # ignore all 3* reviews
    clean_train_reviews = clean_train_reviews[clean_train_reviews["level"] != 3]
    # positive sentiment = 4* or 5* reviews
    clean_train_reviews["sentiment"] = clean_train_reviews["level"] >= 4

    train, test = train_test_split(clean_train_reviews, test_size=0.2)
    train_text = train["text"].values.astype('U')
    test_text = test["text"].values.astype('U')
    
    # convert data-set to term-document matrix
    X_train = vectorizer.fit_transform(train_text).toarray()
    y_train = train["sentiment"]
    
    X_test = vectorizer.fit_transform(test_text).toarray()
    y_test = test["sentiment"]
    print_words_frequency(X_train)


            ###TRAINING###

def train(X_train, y_train, X_test, y_test):
    # iterate over classifiers
    results = {}
    for name, clf in zip(names, classifiers):
        print ("Training " + name + " classifier...")
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[name] = score

    print ("---------------------------")
    print ("Evaluation results")
    print ("---------------------------")

    # sorting results and print out
    sorted(results.items(), key=itemgetter(1))
    for name in results:
        print (name + " accuracy: %0.3f" % results[name])


def test():
    clean_train_reviews = pd.read_csv("Data_train.csv", nrows=30)

    # ignore all 3* reviews
    print(clean_train_reviews[clean_train_reviews["level"] != 4])
    
if __name__ =='__main__':
    bagOfWords("Data_train.csv")
    #test()