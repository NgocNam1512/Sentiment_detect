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

negative_emoticons = {':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{'}

positive_emoticons = {'=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)'}

un_letter = {'!', '.', ',', '?'}

def clean_sentence(sentence):
    # Remove non-letters
    
    #letters_only = re.sub("[^a-zA-Z0-9àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬĐÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ]"," ", review_text)
    for i in sentence:
        if i in un_letter:
            sentence = sentence.replace(i, "")
    return sentence

def preprocess_file(name):
        sentences = open(name, encoding='utf8').read().strip().split("\n")
        train_file = open("Data_train.csv", "w", encoding='utf8')
        train_file.write("label,level,text\n")
    
        for i in range(0, len(sentences), 2):
            text = clean_sentence(sentences[i]) 
            part2 = sentences[i+1]
            label = part2.split("\t")[0]
            level = part2.split("\t")[1]
            #print("text: {}\nlabel: {}\n".format(text, labels))
            train_file.write(label + "," + level + "," + text + "\n")          


def get_reviews_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        return pd.read_csv(file_name)

def countEmoticons(s):
    positive_count = 0
    negative_count = 0
    for emoticon in positive_emoticons:
        positive_count += s.count(emoticon)
    for emoticon in negative_emoticons:
        negative_count += s.count(emoticon)
    return positive_count, negative_count

def cleaning_data(dataset, file_name):
    # Get the number of reviews based on the dataframe column size
    num_reviews = dataset["text"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review
    for i in range(1, num_reviews):
        # clean reviews
        from review_to_word import review_to_words
        label = str(dataset["label"][i])
        level = str(dataset["level"][i])
        text = review_to_words(str(dataset["text"][i]))

        clean_train_reviews.append(label + "," + level + "," + text + "\n")

    with open(file_name, "w", encoding='utf8') as f:
        f.write("label,level,text\n")
        for review in clean_train_reviews:
            f.write("%s\n" % review)


if __name__ =='__main__':
    #preprocess_file("student_manual.train.txt")
    
    
        
        
    
    