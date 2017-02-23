
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
from scipy.cluster.hierarchy import dendrogram, linkage, ward, cophenet
from scipy.spatial.distance import pdist, squareform
from matplotlib import pyplot as plt
import mpld3
import fileinput
import pickle
import sys

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation

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

def print_parameters():
    print "vocab_frame"
    print vocab_frame

    print "\ntfidf_vectorizer"
    print tfidf_vectorizer

    print "\ntfidf_matrix"
    print tfidf_matrix

    print "\nframe"
    print frame

    print "\ndist"
    print dist

fileNames = ['author_IDs', 'abstracts', 'vocab_frame', 'tfidf_vectorizer', 'tfidf_matrix', 'frame', 'dist']

parameters = load_parameters(fileNames)
author_IDs, abstracts, vocab_frame, tfidf_vectorizer, tfidf_matrix, frame, dist = parameters

#print_parameters()

"""
Extract the information about authors by creating a dictionary of author IDs to names
"""
g_cluster = pd.read_csv('golden_cluster.csv')
g_author_id = list(g_cluster[g_cluster.columns[0]])
g_first_name = g_cluster[g_cluster.columns[1]]
g_last_name = g_cluster[g_cluster.columns[2]]

g_name = list(np.array(g_first_name) + np.array(g_last_name))
g_author_id_to_name = dict(zip(g_author_id, g_name))

# This is so that the indices in nameList and author_IDs correspond to the same person
nameList = [g_author_id_to_name[authorid] for authorid in author_IDs]



# generate the linkage matrix Z
# Each element in matrix Z has the form [idx1, idx2, dist, sample_count].
Z = linkage(y=dist, method='ward')

print Z

# c:cophenetic coefficient, coph_dists: coph_distances
c, coph_dists = cophenet(Z, pdist(dist))            

def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    fig, ddata = plt.subplots(figsize=(10, 8))     #set size
    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.ylabel('sample index or (cluster size)')
        plt.xlabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            y = 0.5 * sum(i[1:3])
            x = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % x, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axvline(x=max_d, c='k')
    
        plt.tick_params(    axis= 'x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            labelbottom='off')
        plt.tight_layout() #show plot with tight layout
    return ddata


# set cut-off to 50
max_d = 10  # max_d as in max_distance

fancy_dendrogram(
    Z,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
    orientation="left",
    labels=nameList
)


from scipy.cluster.hierarchy import fcluster

max_d=10
clusters = fcluster(Z, max_d, criterion='distance')

hierarachical_name_list = {}

for cluster in clusters:
    current_cluster = []
    for index in range(len(clusters)):
        if clusters[index] == cluster:
            name = nameList[index]
            current_cluster.append(name)

    hierarachical_name_list[cluster] = current_cluster

print hierarachical_name_list

directoryPath = os.path.join(os.path.dirname(__file__), 'ward_clustering_clusters\\')
for key, value in hierarachical_name_list.items():
    fileName = 'cluster' + str(key) + '.p'
    pathWay = directoryPath + fileName

    with open(pathWay, 'wb') as f:
        pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    fileName = 'cluster' + str(key) + '.txt'
    pathWay = directoryPath + fileName

    with open(pathWay, 'w') as f:
        f.write(str(value))
        f.close()



max_d=4
clusters = fcluster(Z, max_d, criterion='distance')

hierarachical_name_list = {}

for cluster in clusters:
    current_cluster = []
    for index in range(len(clusters)):
        if clusters[index] == cluster:
            name = nameList[index]
            current_cluster.append(name)

    hierarachical_name_list[cluster] = current_cluster

print hierarachical_name_list

directoryPath = os.path.join(os.path.dirname(__file__), 'ward_clustering_clusters2\\')
for key, value in hierarachical_name_list.items():
    fileName = 'cluster' + str(key) + '.p'
    pathWay = directoryPath + fileName

    with open(pathWay, 'wb') as f:
        pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    fileName = 'cluster' + str(key) + '.txt'
    pathWay = directoryPath + fileName

    with open(pathWay, 'w') as f:
        f.write(str(value))
        f.close()



"""
g_cluster.drop(['Group'],1,inplace=True)
aid_cluster_dict = g_cluster.set_index('Author ID')['Cluster'].to_dict()
"""
plt.savefig("hierarchical.png", dpi=2000)
plt.show()





# Hierarachical Clustering
# In this case, we use Ward's method, https://en.wikipedia.org/wiki/Ward%27s_method

"""
linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(10, 8))     #set size
ax = dendrogram(linkage_matrix, orientation="left", labels=author_IDs)


plt.tick_params(    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',        # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout

#uncomment below to save figure
plt.savefig('ward_clusters.png', dpi=200) #save figure as ward_clusters
plt.show()
"""