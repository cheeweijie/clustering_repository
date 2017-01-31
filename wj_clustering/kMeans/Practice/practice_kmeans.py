import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import time
from sklearn import feature_extraction
from sklearn.cluster import KMeans
import mpld3
import fileinput
import pickle

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")	

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]			# stemmer.stem(t) stems each word in filtered_tokens
    return stems

documents = (
"The sky is blue",
"The sun is bright",
"The sun in the sky is bright",
"We can see the shining sun, the bright sun"
)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
print "The tfidf_matrix is"
print tfidf_matrix
print "Its shape is {}\n\n".format(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()
print "These are the keyword/terms."
print terms
print "The length of terms is {}\n".format(len(terms))

from sklearn.metrics.pairwise import cosine_similarity
similarityMatrix = cosine_similarity(tfidf_matrix)
print "Below is the similarity matrix"
print similarityMatrix
print "Its shape is {}\n".format(similarityMatrix.shape)

num_clusters = 2
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print "The indices for both the document and cluster list are the same."
print "The value at each index below gives the clsuter that the sentences belong to."
print clusters
