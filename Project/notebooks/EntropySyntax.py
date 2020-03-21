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
import src.syntax_entropy

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ud_data_classical = src.ud_corpus.POSCorpus.create_from_ud(glob.glob('../data/UD_Classical_Chinese-Kyoto/*.conllu'), split_chars=False)
ud_data_modern = src.ud_corpus.POSCorpus.create_from_ud(glob.glob('../data/UD_Chinese-GSD/*.conllu'), split_chars=False)


# ## Find most common particles

# In[3]:


PARTICLE_POS = ['AUX', 'PART', 'CCONJ', 'SCONJ', 'AUX', 'ADP']

def get_most_common_particles(ud):
  ctr = collections.Counter()
  for sentence in ud.sentences:
    for tok in sentence:
      if tok['pos'] in PARTICLE_POS:
        ctr[tok['word']] += 1
  return ctr


# In[4]:


ctr = get_most_common_particles(ud_data_classical)
ctr.most_common(10)


# In[5]:


ctr = get_most_common_particles(ud_data_modern)
ctr.most_common(10)


# ## Split into segments without punctuation

# In[6]:


segments_classical = src.syntax_entropy.split_into_segments(ud_data_classical.sentences)
segments_modern = src.syntax_entropy.split_into_segments(ud_data_modern.sentences)


# ## POS distribution in various sentence positions

# In[7]:


src.syntax_entropy.display_entropy(src.syntax_entropy.get_start_distribution(segments_classical))


# In[8]:


src.syntax_entropy.display_entropy(src.syntax_entropy.get_start_distribution(segments_modern))


# In[9]:


src.syntax_entropy.display_entropy(src.syntax_entropy.get_end_distribution(segments_classical))


# In[10]:


src.syntax_entropy.display_entropy(src.syntax_entropy.get_end_distribution(segments_modern))


# In[11]:


for ch, _ in get_most_common_particles(ud_data_classical).most_common(5):
  print('Before', ch)
  src.syntax_entropy.display_entropy(src.syntax_entropy.get_before_distribution(segments_classical, ch))
  print('After', ch)
  src.syntax_entropy.display_entropy(src.syntax_entropy.get_after_distribution(segments_classical, ch))
  print()


# In[12]:


for ch, _ in get_most_common_particles(ud_data_modern).most_common(5):
  print('Before', ch)
  src.syntax_entropy.display_entropy(src.syntax_entropy.get_before_distribution(segments_modern, ch))
  print('After', ch)
  src.syntax_entropy.display_entropy(src.syntax_entropy.get_after_distribution(segments_modern, ch))
  print()

