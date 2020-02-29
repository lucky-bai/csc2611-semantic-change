# Concatenate all of 24hist files into one file, remove metadata
import glob
import src.utils

OUTFILE = './24hist.txt'

with open(OUTFILE, 'w') as outf:
  for f in glob.glob('data/24hist/*.txt'):
    sentences = src.utils.load_sentences(f)
    for s in sentences:
      outf.write(s + '\n')
