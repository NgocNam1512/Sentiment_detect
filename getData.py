import re
import io
import os
import pandas as pd
from collections import Counter



def preprocess_file(name):
        sentences = open(name, encoding='utf8').read().strip().split("\n")
        train_file = open("Data_train.csv", "w", encoding='utf8')
        train_file.write("label,level,text\n")
        output_sentences = []
        for i in range(0, len(sentences), 2):
            text = sentences[i]
            part2 = sentences[i+1]
            label = part2.split("\t")[0]
            level = part2.split("\t")[1]
            output_sentence = f"{label} {text}"
            output_sentences.append(output_sentence)
            #print("text: {}\nlabel: {}\n".format(text, labels))
            train_file.write(label + "," + level + "," + text + "\n")          


def get_reviews_data(file_name):
    """Get reviews data, from local csv."""
    if os.path.exists(file_name):
        print("-- " + file_name + " found locally")
        return pd.read_csv(file_name)

    


if __name__ =='__main__':
    #preprocess_file("student_manual.train.txt")
    # Reading the Data
    train = get_reviews_data("Data_train.csv")
    print ("Data dimensions:", train.shape)
    print ("List features:", train.columns.values)
    #print ("First review:", train["label"][0], "|", train["text"][0])
        
    