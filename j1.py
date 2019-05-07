import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity

if __name__== "__main__":
    print('enter resume txt file\n')
    res_f = input()
    with open(res_f) as file:
        resume = file.read()
    print('enter job posting file \n')
    job_f = input()
    with open(job_f) as file:
        job = file.read()


    st = stopwords.words('english')
    tf  = TfidfVectorizer(stop_words = st)
    m1 = tf.fit([job])
    j_vect = m1.transform([job])
    r_vect = m1.transform([resume])
    ##right now tfidf === countvectoriser
    print("tf idf vectors comparision cosine similarity" cosine_similarity(j_vect, r_vect))
