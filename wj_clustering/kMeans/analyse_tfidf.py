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

with open("vocab_frame.p", "rb") as f:     # Save the vectorizer in a pickle file
    vocab_frame = pickle.load(f)
    f.close()

with open("tfidf_vectorizer.p", "rb") as g:		# Save the matrix in a pickle file
	tfidf_vectorizer = pickle.load(g)
	g.close()

with open("tfidf_matrix.p", "rb") as h:
	tfidf_matrix = pickle.load(h)
	h.close()

print "In this program, we are going to analyse the properties of the tfidf_matrix."
print "tfidf stands for term frequency-inverse document frequency\n"
print tfidf_matrix
print "This is the data type of the tfidf_matrix {}".format(type(tfidf_matrix))
raw_input("\nThe above is the tfidf_matrix.\nPlease press any key to continue.\n")

print "The tfidf_matrix's shape has data type {}".format(type(tfidf_matrix.shape))
print "This is the shape of the matrix {}".format(tfidf_matrix.shape)
print "There are {} authors/author_IDs".format(tfidf_matrix.shape[0])
print "There are {} keywords to be analysed\n".format(tfidf_matrix.shape[1])

raw_input("\nTake a look at the shape of the tfidf_matrix produced.\nPress any key to continue.\n")


print "Now we view the entries in the tfidf_matrix\n"

print "tfidf_matrix\n"
print tfidf_matrix

print "\ntfidf_matrix[0]\nThis is the "
print tfidf_matrix[0]
print "\ntfidf_matrix[123]\n"
print tfidf_matrix[123]

print "Below is the list of keywords within the abstracts."
print "These keywords are used to analyse the abstracts written by each author with a unique author_ID\n"
terms = tfidf_vectorizer.get_feature_names()
print terms[:100]
print "The length of terms is {}\n".format(len(terms))

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
print "This is the distance matrix:\n{}\n".format(dist)
print type(dist)
print dist.shape

distList = dist.tolist()
print distList[0]
print "\n\n"

print "The following is a consistency check of the distance matrix."
print "The distance matrix shows how different are each author's set of abstracts from one another"
print "Thus, one key property of this distance matrix is that"
print "we expect the distance from each author's set of abstracts to itself to be zero."
print "Hence, for every document, its list of distances from the other documents of which contains itself should contain at least one zero.\n"

diagonalZero = True
zeroFound = True		# Assume zeroes found in every list within the distList
for k in range(len(distList)):
	print "This is index {} in distList".format(k)
	zeroInList = False
	for index, value in enumerate(distList[k]):
		if value <= 10**-5:
			print "Zero found at index {} with value {}".format(index, value)
			zeroInList = True	# Indicate that a zero has been found
		
		else:
			if index == k:			# If non-zero, check if this is a diagonal entry
				print "The diagonal entry at {} has value {}".format(index, value)
				raw_input("")
				diagonalZero = False		# Non-zero diagonal entry found so not all diagonal entries are zero

	else:
		if zeroInList:
			print "Zeroes found\n"
		else:
			zeroFound = False
			print "No zeroes found\n"

else:
	if zeroFound:
		print "\nAll lists within distList contain at least one zero"
	else:
		print "\nThere is one list within distList where no zeroes are found"

	if diagonalZero:
		print "The diagonal entries of the distance matrix is zero"
	else:
		print "At least one diagonal entry of the distance matrix is non-zero"


