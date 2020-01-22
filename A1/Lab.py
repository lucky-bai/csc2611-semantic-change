#!/usr/bin/env python
# coding: utf-8

# # Lab: Word embedding and semantic change

# In[1]:


import pandas as pd
import scipy.spatial.distance
import scipy.stats

from gensim.models import KeyedVectors


# ## Load data

# In[2]:


model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[3]:


rg65 = pd.read_csv('rg65.csv')


# In[4]:


rg65.head(5)


# ## Cosine distance between embeddings

# In[5]:


def get_cosine_distance(row):
  return scipy.spatial.distance.cosine(model[row['word1']], model[row['word2']])

rg65['w2v_cosine_distance'] = rg65.apply(get_cosine_distance, axis=1)


# In[6]:


# Pearson correlation
print('Pearson correlation:', scipy.stats.pearsonr(rg65.similarity, rg65.w2v_cosine_distance)[0])


# In[7]:


# Need to do exercise to get LSA...


# In[ ]:




