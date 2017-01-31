import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
import time
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import mpld3
import fileinput
import pickle
import sys

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


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

def load_parameters():
    with open("author_ids.p", "rb") as f:     # Save the vectorizer in a pickle file
        author_IDs = pickle.load(f)
        f.close()

    with open("abstracts.p", "rb") as f:     # Save the vectorizer in a pickle file
        abstracts = pickle.load(f)
        f.close()

    with open("vocab_frame.p", "rb") as f:     # Save the vectorizer in a pickle file
        vocab_frame = pickle.load(f)
        f.close()

    with open("tfidf_vectorizer.p", "rb") as g:		# Save the matrix in a pickle file
    	tfidf_vectorizer = pickle.load(g)
    	g.close()

    with open("tfidf_matrix.p", "rb") as h:
    	tfidf_matrix = pickle.load(h)
    	h.close()

    with open("frame.p", "rb") as h:
        frame = pickle.load(h)
        h.close()

    return (author_IDs, abstracts, vocab_frame, tfidf_vectorizer, tfidf_matrix, frame)


author_IDs, abstracts, vocab_frame, tfidf_vectorizer, tfidf_matrix, frame = load_parameters()

terms = tfidf_vectorizer.get_feature_names()

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print "This is the length of the cluster list {}.".format(len(clusters))
print "The authors in each index are assigned to the clusters according to this list.\n{}\n".format(clusters)

faculty = { 'author_IDs': author_IDs, 'abstracts': abstracts, 'cluster': clusters}
frame = pd.DataFrame(faculty, index = [clusters], columns = ['author_IDs','abstracts', 'cluster'])
print frame['cluster'].value_counts()
print frame
print

print("Top terms per cluster:")
print
clusterCenters = km.cluster_centers_			# Returns the quantitative significance of the clusters
print clusterCenters
print
# argsort() returns the indices that would sort an array
# hence order_centroids give the indices in terms (it's a list of keywords) corresponding to the keywords from least to most significance
order_centroids = clusterCenters.argsort()		
print order_centroids
print
# reverse the order so that you have the indices in terms corresponding to the keywords from most to least significance
order_centroids = order_centroids[:, ::-1]
print order_centroids

for i in range(num_clusters):
    print("Cluster %d words:" % i)
    for ind in order_centroids[i, :6]:
        print('%s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
    print
    print
    print("Cluster %d titles:" % i)
    for author_id in frame.ix[i]['author_IDs'].values.tolist():
        print(' %s,' % author_id)
    print
    print

