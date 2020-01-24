#!/usr/bin/env python
# coding: utf-8

# # Lab: Word embedding and semantic change

# In[1]:


import pandas as pd
import scipy.spatial.distance
import scipy.stats
import pickle
import nltk
from gensim.models import KeyedVectors


# ## Load data

# In[2]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[3]:


rg65 = pd.read_csv('rg65.csv')


# In[4]:


rg65.head(5)


# In[5]:


brown_words = nltk.corpus.brown.words()
fdist = nltk.FreqDist(w.lower() for w in brown_words)
brown_words


# In[6]:


# Deserialize LSA vectors from exercise
with open('lsa-exercise.pkl', 'rb') as f:
  pickle_data = pickle.load(f)
  WORDS = pickle_data['WORDS']
  M2 = pickle_data['M2']
  
# Mapping between word and integer
word_to_int = {}
int_to_word = {}
for ix, w in enumerate(sorted(list(WORDS))):
  word_to_int[w] = ix
  int_to_word[ix] = w


# ## Cosine distance between embeddings

# In[7]:


def get_cosine_similarity(row):
  return 1 - scipy.spatial.distance.cosine(model[row['word1']], model[row['word2']])

rg65['w2v_similarity'] = rg65.apply(get_cosine_similarity, axis=1)


# In[8]:


# Pearson correlation
print('Pearson correlation:', scipy.stats.pearsonr(rg65.similarity, rg65.w2v_similarity)[0])


# ## Analogy test

# In[ ]:




