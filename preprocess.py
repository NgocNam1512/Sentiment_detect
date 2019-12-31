import time
import datetime
import re
from itertools import islice
from calculate_time import time_diff_str
from bs4 import BeautifulSoup

def clean_sentence(sentence):
    # Remove non-letter
    letters_only = re.sub("[^a-zA-Z]"+, " ", sentence)
    return letters_only


def convert_plain_to_csv(plain_name, csv_name):
    t0 = time.time()
    with open(plain_name, "r", encoding="utf8") as f1, open(csv_name, "w", encoding="utf8") as f2:
        # process next_2_lines: get score,label,text info
        # remove special characters from summary and text
        i = 0
        f2.write("score,label,text\n")
        while True:
            next_n_lines = list(islice(f1, 2))
            if not next_n_lines:
                break

            output_line=""
            for j, line in enumerate(f1):
                if j % 2 == 0:
                    #text = clean_sentences(line.strip())
                    text = line.strip()
                else:
                    label, score = line.strip().split("\t")
                    output_line = score + "," + label + "," + text +'\n'
                    f2.write(output_line)
                    
            # print status
            i += 1
            if i % 10000 == 0:
                print("%d reviews converted..." % i)

    print(" %s - Converting completed %s" % (datetime.datetime.now(), time_diff_str(t0, time.time())))


if __name__ == "__main__":
    convert_plain_to_csv("test_data.txt", "test_data.csv")