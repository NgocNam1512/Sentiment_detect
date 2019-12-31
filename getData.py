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

negative_emoticons = [':(', '☹', '❌', '👎', '👹', '💀', '🔥', '🤔', '😏', '😐', '😑', '😒', '😓', '😔', '😕', '😖',
                      '😞', '😟', '😠', '😡', '😢', '😣', '😤', '😥', '😧', '😨', '😩', '😪', '😫', '😭', '😰', '😱',
                      '😳', '😵', '😶', '😾', '🙁', '🙏', '🚫', '>:[', ':-(', ':(', ':-c', ':c', ':-<', ':っC', ':<',
                      ':-[', ':[', ':{']

positive_emoticons = ['=))', 'v', ';)', '^^', '<3', '☀', '☺', '♡', '♥', '✌', '✨', '❣', '❤', '🌝', '🌷', '🌸',
                      '🌺', '🌼', '🍓', '🎈', '🐅', '🐶', '🐾', '👉', '👌', '👍', '👏', '👻', '💃', '💄', '💋',
                      '💌', '💎', '💐', '💓', '💕', '💖', '💗', '💙', '💚', '💛', '💜', '💞', ':-)', ':)', ':D', ':o)',
                      ':]', ':3', ':c)', ':>', '=]', '8)']

neg, pos = "", ""
for i in negative_emoticons:
    neg += i + " "
for i in positive_emoticons:
    pos += i + " "

def clean_sentence(sentence):
    # Remove HTML
    review_text = BeautifulSoup(sentence).text

    # Remove non-letters
    
    letters_only = re.sub("[^a-zA-Z0-9àảãáạăằẳẵắặâầẩẫấậđèẻẽéẹêềểễếệìỉĩíịòỏõóọôồổỗốộơờởỡớợùủũúụưừửữứựỳỷỹýỵÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬĐÈẺẼÉẸÊỀỂỄẾỆÌỈĨÍỊÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢÙỦŨÚỤƯỪỬỮỨỰỲỶỸÝỴ]"," ", review_text)
    
    return letters_only

def preprocess_file(name):
        j = 0
        sentences = open(name, encoding='utf8').read().strip().split("\n")
        train_file = open("Data_train.csv", "w", encoding='utf8')
        train_file.write("label,level,text\n")
        output_sentences = []
        for i in range(0, len(sentences), 2):
            text = clean_sentence(sentences[i])
            part2 = sentences[i+1]
            label = part2.split("\t")[0]
            level = part2.split("\t")[1]
            output_sentence = f"{label} {text}"
            output_sentences.append(output_sentence)
            j += 1
            #print("text: {}\nlabel: {}\n".format(text, labels))
            train_file.write(label + "," + level + "," + text + "\n")          
            if j == 60:
                return 0


def get_reviews_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        return pd.read_csv(file_name)

    


if __name__ =='__main__':
    preprocess_file("student_manual.train.txt")
    # Reading the Data
    #train = get_reviews_data("Data_train.csv")
    #print ("Data dimensions:", train.shape)
    #print ("List features:", train.columns.values)
    #print ("First review:", train["label"][0], "|", train["text"][0])
        
    
    