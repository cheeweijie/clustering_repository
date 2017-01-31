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

# Import Author IDs List
author_IDs = []
for file in os.listdir("C:\WJ\Imperial\Physics\Year_3\Third_Year_Project\Data_Sets\PhysicsStaff151113-2010-15-All"):
    if file.endswith(".txt"):
        author_IDs.append(file.rstrip(".txt"))
        
print 'First 5 author IDs are: ', author_IDs[:5]
print '\n'

with open("author_ids.p", "wb") as f:     # Save the vectorizer in a pickle file
    pickle.dump(author_IDs, f, pickle.HIGHEST_PROTOCOL)
    f.close()

# Import Abstract List
# Each item in abstracts contain all the abstracts for a particular author_ID
# The author ID corresponding to the abstract has the same index in both the author_IDs and abstracts list
abstracts = []
path = 'C:\WJ\Imperial\Physics\Year_3\Third_Year_Project\Data_Sets\PhysicsStaff151113-2010-15-All\\'
for i in author_IDs:
    abstracts.append(open(path + i + '.txt').read())

with open("abstracts.p", "wb") as f:     # Save the vectorizer in a pickle file
    pickle.dump(abstracts, f, pickle.HIGHEST_PROTOCOL)
    f.close()
    
print 'compiled abstract for 1st a_ID: ', abstracts[0][:200] + '...'
print '\n'
print 'compiled abstract for 2nd a_ID: ', abstracts[1][:200] + '...'
print '\n'


# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
print 'English Stopwords: ', stopwords
print '\n'

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")			# Create an instance of the object SnowballStemmer

# Tokenizer and stemmer which returns the set of stems in the text that it is passed

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

print 'Tokenizing + stemming of 1st A_id example:', tokenize_and_stem(abstracts[0][:200])
print '\n'

def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens

print 'Tokenizing only of 1st A_id example:', tokenize_only(abstracts[0][:200])
print '\n'

totalvocab_stemmed = []
totalvocab_tokenized = []
for i in abstracts:
	allwords_stemmed = tokenize_and_stem(i)
	totalvocab_stemmed.extend(allwords_stemmed)

	allwords_tokenized = tokenize_only(i)
	totalvocab_tokenized.extend(allwords_tokenized)


# DataFrame with the stemmed vocabulary as the index and the tokenized words as the column
vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print vocab_frame
print

with open("vocab_frame.p", "wb") as f:     # Save the vectorizer in a pickle file
    pickle.dump(vocab_frame, f, pickle.HIGHEST_PROTOCOL)
    f.close()

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts) # Fit the vectorizer to abstracts

with open("tfidf_vectorizer.p", "wb") as g:		# Save the vectorizer in a pickle file
	pickle.dump(tfidf_vectorizer, g, pickle.HIGHEST_PROTOCOL)
	g.close()

with open("tfidf_matrix.p", "wb") as h:		# Save the matrix in a pickle file
	pickle.dump(tfidf_matrix, h, pickle.HIGHEST_PROTOCOL)
	h.close()

terms = tfidf_vectorizer.get_feature_names()
print terms[:100]
print "The length of terms is {}".format(len(terms))

raw_input("")

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

faculty = { 'author_IDs': author_IDs, 'abstracts': abstracts, 'cluster': clusters}
frame = pd.DataFrame(faculty, index = [clusters], columns = ['author_IDs','abstracts', 'cluster'])
print frame['cluster'].value_counts()

with open("frame.p", "wb") as f:     # Save the vectorizer in a pickle file
    pickle.dump(frame, f, pickle.HIGHEST_PROTOCOL)
    f.close()


print("Top terms per cluster:")
print
clusterCenters = km.cluster_centers_			# Returns the coordinates of the clusters
order_centroids = clusterCenters.argsort()		# Returns the indices that would sort an array.
order_centroids = order_centroids[:, ::-1]

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
