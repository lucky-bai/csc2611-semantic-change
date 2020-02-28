import re

CHINESE_PUNCT = "。！？"

def contains_chinese_char(text):
  return re.search("[\u4e00-\u9FFF]", text) is not None

def load_sentences(fname):
  with open(fname) as f:
    text_lines = f.read().split('\n')
    text_lines = [l for l in text_lines if contains_chinese_char(l)]

  sentences = []
  for line in text_lines:
    last_ix = -1
    for ix, ch in enumerate(line):
      if ch in CHINESE_PUNCT:
        sentences.append(line[last_ix+1:ix+1])
        last_ix = ix
  return sentences

