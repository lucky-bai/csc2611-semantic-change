#!/usr/bin/env python
# coding: utf-8

# # Concreteness Experiments
# 
# Attempt to correlate N/V percentage with concreteness norms from Yao et al. (2017).
# 
# Yao, Zhao, et al. "Norms of valence, arousal, concreteness, familiarity, imageability, and context availability for 1,100 Chinese words." Behavior research methods 49.4 (2017): 1374-1385.

# In[1]:


from collections import defaultdict
import numpy as np
import pandas as pd


# In[2]:


# Extracted from Yao et al. (2017) PDF using tabula, with manual corrections
yao2017 = pd.read_csv('../data/yao2017_norms.csv')
yao2017.head()


# ## Break down by character

# In[3]:


D = defaultdict(list)
for ix, row in yao2017.iterrows():
  for ch in row['Chinese words']:
    D[ch].append(row['CON_M'])


# In[4]:


concreteness_df = []
for ch, ratings in D.items():
  concreteness_df.append(pd.Series({
    'character': ch,
    'compounds': len(ratings),
    'concreteness': np.mean(np.array(ratings))
  }))
concreteness_df = pd.DataFrame(concreteness_df)
concreteness_df.head()

