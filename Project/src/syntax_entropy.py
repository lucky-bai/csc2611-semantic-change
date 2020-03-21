from collections import Counter
import scipy.stats
import numpy as np


# Split into segments without punctuation
def split_into_segments(sentences):
  new_sentences = []
  for sentence in sentences:
    new_sentence = []
    
    for tok in sentence:
      if tok['pos'] == 'PUNCT':
        if new_sentence != []:
          new_sentences.append(new_sentence)
        new_sentence = []
      else:
        new_sentence.append(tok)
        
    if new_sentence != []:
      new_sentences.append(new_sentence)
    
  return new_sentences


# Merge everything that's not noun or verb
def postprocess_distribution(ctr_fn):
  def inner(*args):
    ctr = ctr_fn(*args)
    new_ctr = Counter()
    for k, v in ctr.most_common():
      if k in ['NOUN', 'VERB']:
        new_ctr[k] += v
      else:
        new_ctr['OTHER'] += v
    return new_ctr
  return inner



@postprocess_distribution
def get_start_distribution(segments):
  ctr = Counter()
  for seg in segments:
    ctr[seg[0]['pos']] += 1
  return ctr


@postprocess_distribution
def get_end_distribution(segments):
  ctr = Counter()
  for seg in segments:
    ctr[seg[-1]['pos']] += 1
  return ctr


@postprocess_distribution
def get_before_distribution(segments, target_ch):
  ctr = Counter()
  for seg in segments:
    for i, ch in enumerate(seg):
      if i != len(seg) - 1 and seg[i+1]['word'] == target_ch:
        ctr[ch['pos']] += 1
  return ctr


@postprocess_distribution
def get_after_distribution(segments, target_ch):
  ctr = Counter()
  for seg in segments:
    for i, ch in enumerate(seg):
      if i != 0 and seg[i-1]['word'] == target_ch:
        ctr[ch['pos']] += 1
  return ctr


def display_entropy(dist):
  print(dist)
  vals = np.array(list(dist.values()))
  entropy = scipy.stats.entropy(vals)
  print(entropy)
