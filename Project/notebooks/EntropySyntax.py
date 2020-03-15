#!/usr/bin/env python
# coding: utf-8

# # POS Entropy in different syntactic positions

# In[1]:


import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats
import glob
import collections

import src.ud_corpus

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ud_data_classical = src.ud_corpus.POSCorpus.create_from_ud(glob.glob('../data/UD_Classical_Chinese-Kyoto/*.conllu'))
ud_data_modern = src.ud_corpus.POSCorpus.create_from_ud(glob.glob('../data/UD_Chinese-GSD/*.conllu'))


# ## Find most common particles

# In[3]:


PARTICLE_POS = ['AUX', 'PART', 'CCONJ', 'SCONJ', 'AUX', 'ADP']

def get_most_common_particles(ud):
  ctr = collections.Counter()
  for sentence in ud.sentences:
    for tok in sentence:
      if tok['pos'] in PARTICLE_POS:
        ctr[tok['char']] += 1
  return ctr


# In[4]:


ctr = get_most_common_particles(ud_data_classical)
ctr.most_common(10)


# In[5]:


ctr = get_most_common_particles(ud_data_modern)
ctr.most_common(10)

