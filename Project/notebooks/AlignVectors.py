#!/usr/bin/env python
# coding: utf-8

# # Align Vectors

# In[1]:


import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
import gensim
import glob

import src.procrustes
import src.ud_corpus

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


# In[7]:


shared_vocab = set(vecs_classical.vocab.keys()) & set(vecs_modern.vocab.keys())


# ## Calculate amount of change for shared vocab

# In[8]:


ud_data = src.ud_corpus.POSCorpus.create_from_ud(glob.glob('../data/UD_Classical_Chinese-Kyoto/*.conllu'))


# In[9]:


summary_data = ud_data.get_nv_stats().sort_values('total_count', ascending=False)
summary_data = summary_data[summary_data.total_count >= 5]
summary_data = summary_data[summary_data.char.isin(shared_vocab)]
summary_data['semantic_change'] = summary_data.char.apply(
  lambda c: np.linalg.norm(vecs_classical[c] - vecs_modern[c])
)
summary_data


# ## Print words with most and least change

# In[10]:


summary_data.sort_values("semantic_change", ascending=False).head(15)


# In[11]:


summary_data.sort_values("semantic_change").head(15)


# In[12]:


sns.regplot(summary_data.noun_ratio, summary_data.semantic_change)


# In[13]:


scipy.stats.pearsonr(summary_data.noun_ratio, summary_data.semantic_change)

