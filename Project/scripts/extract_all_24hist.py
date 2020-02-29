"""
Concatenate all of 24hist files into one file, remove metadata.
Usage:
  PYTHONPATH=. python scripts/extract_all_24hist.py
"""
import glob
import argparse
import src.utils

parser = argparse.ArgumentParser()
parser.add_argument('--in_dir', default='data/24hist')
parser.add_argument('--out_file', default='./24hist.txt')
parser.add_argument('--spaces', action='store_true') # Needed for GloVe
args = parser.parse_args()
print(args)

with open(args.out_file, 'w') as outf:
  for f in glob.glob(args.in_dir + '/*.txt'):
    sentences = src.utils.load_sentences(f)
    for s in sentences:
      if args.spaces:
        s = ' '.join(list(s))
      outf.write(s + '\n')
