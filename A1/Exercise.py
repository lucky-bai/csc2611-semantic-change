#!/usr/bin/env python
# coding: utf-8

# # Exercise: Meaning construction from text

# In[1]:


import nltk
import pandas as pd


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


len(WORDS)


# ## Step 3: construct word-context matrix

# In[ ]:




