#!/usr/bin/env python
# coding: utf-8

# # Basic Frequency Stats

# In[1]:


import sys
sys.path.append('../')

import re
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import src.utils

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load data
# 
# Filter out lines that don't contain chinese characters.
# 
# Split sentences by period mark.

# In[2]:


TEXT_NAME = '../data/24hist/01_shiji_full.txt'
#TEXT_NAME = '../data/24hist/24_mingshi_full.txt'
sentences = src.utils.load_sentences(TEXT_NAME)


# In[3]:


len(sentences)


# In[4]:


sentences[:10]


# ## Most frequent, sentence length distribution

# In[5]:


c = Counter()
for l in sentences:
  for ch in l:
    c[ch] += 1
c.most_common(20)


# In[6]:


sentence_lengths = np.array([len(sent) for sent in sentences])
sns.distplot(sentence_lengths)

