#!/usr/bin/env python
# coding: utf-8

# # Align Vectors

# In[1]:


import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gensim

import src.procrustes

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


vecs_classical = gensim.models.KeyedVectors.load_word2vec_format('../data/glove/classical-chinese/vectors.txt', binary=False)
vecs_modern = gensim.models.KeyedVectors.load_word2vec_format('../data/glove/modern-chinese/vectors.txt', binary=False)


# In[3]:


print(len(vecs_classical.vocab))
print(len(vecs_modern.vocab))


# In[4]:


vecs_classical.most_similar('也')


# In[5]:


vecs_modern.most_similar('也')


# In[6]:


vecs_modern = src.procrustes.smart_procrustes_align_gensim(vecs_classical, vecs_modern)


# ## Print amount of change

# In[7]:


for c in list('之不也而以其子曰人者有為'):
  print(c, np.linalg.norm(vecs_classical[c] - vecs_modern[c]))

