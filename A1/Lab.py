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
import tqdm
import multiprocessing as mp


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


# ## Part 3: cosine distance between embeddings

# In[7]:


def get_cosine_similarity(row):
  return 1 - scipy.spatial.distance.cosine(model[row['word1']], model[row['word2']])

rg65['w2v_similarity'] = rg65.apply(get_cosine_similarity, axis=1)


# In[8]:


# Pearson correlation
print('Pearson correlation:', scipy.stats.pearsonr(rg65.similarity, rg65.w2v_similarity)[0])


# ## Part 4: analogy test

# In[9]:


# Read words
with open('word-test.v1.txt') as f:
  lines = f.read().split('\n')


# In[10]:


analogy_words = []
for line in lines:
  line = line.lower()
  line_words = line.split()
  if len(line_words) == 4:
    w1, w2, w3, w4 = line_words
    if w1 in WORDS and w2 in WORDS and w3 in WORDS and w4 in WORDS:
      analogy_words.append((w1, w2, w3, w4))


# ### Evaluate word2vec

# In[11]:


def w2v_get_nearest(v):
  best_cosine_distance = 1000
  best_word = None
  for w in WORDS:
    if w not in model.vocab:
      continue
    d = scipy.spatial.distance.cosine(v, model[w])
    if d < best_cosine_distance:
      best_cosine_distance = d
      best_word = w
  return best_word

def w2v_process_analogy(inp):
  w1, w2, w3, w4 = inp
  return w2v_get_nearest(model[w3] + model[w2] - model[w1])

# multi-threading to speed it up
with mp.Pool() as pool:
  analogy_results = pool.map(w2v_process_analogy, analogy_words)
  
w2v_correct = 0
for (w1, w2, w3, w4), w_guess in zip(analogy_words, analogy_results):
  if w4 == w_guess:
    w2v_correct += 1
print('Correct:', w2v_correct, '/', len(analogy_words))


# ## Evaluate LSA

# In[ ]:


def lsa_get_nearest(v):
  best_cosine_distance = 1000
  best_word = None
  for w in WORDS:
    d = scipy.spatial.distance.cosine(v, M2[word_to_int[w]])
    if d < best_cosine_distance:
      best_cosine_distance = d
      best_word = w
  return best_word

def lsa_process_analogy(inp):
  w1, w2, w3, w4 = inp
  return lsa_get_nearest(model[w3] + model[w2] - model[w1])

# multi-threading to speed it up
with mp.Pool() as pool:
  analogy_results = pool.map(lsa_process_analogy, analogy_words)
  
lsa_correct = 0
for (w1, w2, w3, w4), w_guess in zip(analogy_words, analogy_results):
  if w4 == w_guess:
    lsa_correct += 1
print('Correct:', lsa_correct, '/', len(analogy_words))


# In[ ]:




