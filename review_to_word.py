import string

def review_to_words(review):
    """
    Function to convert a raw review to a string of words
    :param review
    :return: meaningful_words
    """
    # 1. Convert to lower case, split into individual words
    words = review.lower().split()
    #
    # 2. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set()
    f = open("vie_stopwords.txt", "r", encoding="utf8")
    for line in f:
        word = line.strip()
        stops.add(word)
    #
    # 3. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 4. Join the words back into one string separated by space,
    # and return the result.
    return " ".join(meaningful_words)


if __name__ == "__main__":
    review_to_words("Xin chào")