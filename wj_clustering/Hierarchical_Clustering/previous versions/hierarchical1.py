
# coding: utf-8

# In[ ]:

'''
Notes:
Tokenization: In lexical analysis, tokenization is the process of breaking a stream of text up into words, phrases, 
              symbols, or other meaningful elements called tokens. Typically, tokenization occurs at the word level. 
              However, it is sometimes difficult to define what is meant by a "word". 

Stemming: In linguistic morphology and information retrieval, stemming is the process for reducing inflected (or sometimes 
          derived) words to their stem, base or root form—generally a written word form. 
'''


# In[2]:

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
    stems = [stemmer.stem(t) for t in filtered_tokens]          # stemmer.stem(t) stems each word in filtered_tokens
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

def load_parameters(fileNames):
    parameters = []
    for fileName in fileNames:
        fileName += ".p"        # Add the extension
        with open(fileName, "rb") as f:
            parameters.append(pickle.load(f))
            f.close()

    return parameters

fileNames = ['author_IDs', 'abstracts', 'vocab_frame', 'tfidf_vectorizer', 'tfidf_matrix', 'frame', 'dist']

parameters = load_parameters(fileNames)
author_IDs, abstracts, vocab_frame, tfidf_vectorizer, tfidf_matrix, frame, dist = parameters

print tfidf_matrix
print tfidf_matrix.shape
print len(dist)
print len(dist[:,0])

# Hierarachical Clustering
# In this case, we use Ward's method, https://en.wikipedia.org/wiki/Ward%27s_method

from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

linkage_matrix = ward(tfidf_matrix) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(15, 20)) # set size
ax = dendrogram(linkage_matrix, orientation="left", labels=author_IDs);

plt.tick_params(    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters


# In[ ]:

'''
Todo: 
different cluster size
d_limit
find out which words excluded 

dendrogram -> manual labelling
'''


