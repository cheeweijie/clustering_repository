
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



# Import Author IDs List
author_IDs = []
for file in os.listdir("C:\WJ\Imperial\Physics\Year_3\Third_Year_Project\Data_Sets\PhysicsStaff151113-2010-15-All"):
    if file.endswith(".txt"):
        author_IDs.append(file.rstrip(".txt"))
        
print 'First 5 author IDs are: ' ,author_IDs[:5]
print '\n'

# Import Abstract List
abstracts = []
path = 'C:\WJ\Imperial\Physics\Year_3\Third_Year_Project\Data_Sets\PhysicsStaff151113-2010-15-All\\'
for i in author_IDs:
    abstracts.append(open(path + i + '.txt').read())
    
print 'compiled abstract for 1st a_ID: ',abstracts[0][:200] + '...'
print '\n'
print 'compiled abstract for 2nd a_ID: ',abstracts[1][:200] + '...'
print '\n'


# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
print 'English Stopwords: ', stopwords
print '\n'

# load nltk's SnowballStemmer as variabled 'stemmer'
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# Tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
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


# In[3]:

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(abstracts) #fit the vectorizer to synopses
print tfidf_matrix.shape

terms = tfidf_vectorizer.get_feature_names()
print terms[:20]

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)


# K means clustering

num_clusters = 5
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()
print clusters


faculty = { 'author_IDs': author_IDs, 'abstracts': abstracts, 'cluster': clusters}
frame = pd.DataFrame(faculty, index = [clusters] , columns = ['author_IDs','abstracts', 'cluster'])
print frame['cluster'].value_counts()




# In[10]:


print("Top terms per cluster:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
for i in range(num_clusters):
    print("Cluster %d words:" % i)
    for ind in order_centroids[i, :6]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore'))
    print()
    print()
    print("Cluster %d titles:" % i)
    for title in frame.ix[i]['author_IDs'].values.tolist():
        print(' %s,' % title)
    print()
    print()


# In[15]:

from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

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


