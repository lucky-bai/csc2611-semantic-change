#!/usr/bin/env python
# coding: utf-8

# # Exercise: Meaning construction from text

# In[1]:


import nltk
import pandas as pd
import scipy.sparse
import sklearn.decomposition
import math
import pickle
from collections import defaultdict


# In[2]:


rg65 = pd.read_csv('rg65.csv')


# In[3]:


rg65.head(5)


# ## Step 2: construct W

# In[4]:


brown_words = nltk.corpus.brown.words()
fdist = nltk.FreqDist(w.lower() for w in brown_words)


# In[5]:


# W = most common 5000 in Brown corpus + Table 1 of RG65
WORDS = set([t[0] for t in fdist.most_common(5000)])
WORDS.update(set(rg65.word1))
WORDS.update(set(rg65.word2))


# In[6]:


# Mapping between word and integer
word_to_int = {}
int_to_word = {}
for ix, w in enumerate(sorted(list(WORDS))):
  word_to_int[w] = ix
  int_to_word[ix] = w


# ## How frequent are the RG65 words?
for w in set(rg65.word1.tolist() + rg65.word2.tolist()):
  print(w, fdist[w])
# ## Step 3: construct word-context matrix

# In[7]:


M1 = scipy.sparse.lil_matrix((len(WORDS), len(WORDS)))
for w1, w2 in nltk.bigrams(brown_words):
  w1 = w1.lower()
  w2 = w2.lower()
  if w1 in WORDS and w2 in WORDS:
    M1[word_to_int[w1], word_to_int[w2]] += 1


# ## Step 4: apply PPMI

# In[8]:


M1P = scipy.sparse.lil_matrix((len(WORDS), len(WORDS)))
M1sum = M1.sum()
rs, cs = M1.nonzero()
for r, c in zip(rs, cs):
  Pw1 = fdist[int_to_word[r]] / len(brown_words)
  Pw2 = fdist[int_to_word[c]] / len(brown_words)
  PJoint = M1[r, c] / M1sum
  M1P[r, c] = max(0, math.log(PJoint / (Pw1 * Pw2)))


# ## Step 5: apply PCA

# In[9]:


NDIM = 300
svd = sklearn.decomposition.TruncatedSVD(n_components=NDIM, random_state=12345)
M2 = svd.fit_transform(M1P)


# ## Step 7: calculate cosine similarity

# In[13]:


# Change this to specify which matrix to use
MTX = M2

def get_cosine_similarity(row):
  w1_ix = word_to_int[row['word1']]
  w2_ix = word_to_int[row['word2']]
  
  # Discard ones that occur in less than 3 bigrams
  if M1[w1_ix].sum() <= 2 or M1[w2_ix].sum() <= 2:
    return None
  
  if scipy.sparse.issparse(MTX):
    return 1 - scipy.spatial.distance.cosine(MTX[w1_ix].todense(), MTX[w2_ix].todense())
  else:
    return 1 - scipy.spatial.distance.cosine(MTX[w1_ix], MTX[w2_ix])

rg65['lsa_similarity'] = rg65.apply(get_cosine_similarity, axis=1)


# ## Step 8: Pearson correlation
# 
# Results:
# ```
# M1: 0.21639005014068902
# M1P: 0.41548942500838
# M2_10: 0.1333050985847491
# M2_100: 0.329345887621373
# M2_300: 0.41080191606630956
# ```

# In[14]:


rg_notnull = rg65.dropna()
scipy.stats.pearsonr(rg_notnull.similarity, rg_notnull.lsa_similarity)


# ## Serialize vectors

# In[15]:


with open('lsa-exercise.pkl', 'wb') as f:
  pickle.dump({'WORDS': WORDS, 'M2': M2}, f)

