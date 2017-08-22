#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from numpy import dot, prod, sum as np_sum
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-vec", "--vector_file", help="binary vector file")
parser.add_argument("-sim", "--sim_file", help="word similarity file")
parser.add_argument("-nor", "--normalized", help="normalize the vector", action="store_true")
parser.add_argument("-el", "--expected_likelihood", help="using expected likelihood to calculate similarity", action="store_true")
parser.add_argument("-cate_n", "--cate_n", help="category variable num", type=int, default=0)
parser.add_argument("-cate_k", "--cate_k", help="category variable size", type=int, default=0)
args = parser.parse_args()

def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000, 
                        case_insensitive=True, dummy4unknown=False, normalized=True,
                        el=False, cate_n=0, cate_k=0):
  ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
  ok_vocab = dict((w.upper(), v) for w, v in reversed(ok_vocab)) if case_insensitive else dict(ok_vocab)

  similarity_gold = []
  similarity_model = []
  oov = 0

  original_vocab = self.vocab
  self.vocab = ok_vocab

  for line_no, line in enumerate(utils.smart_open(pairs)):
    line = utils.to_unicode(line)
    if line.startswith('#'):
      # May be a comment
      continue
    else:
      try:
        if case_insensitive:
          a, b, sim = [word.upper() for word in line.split(delimiter)]
        else:
          a, b, sim = [word for word in line.split(delimiter)]
        sim = float(sim)
      except:
        print('skipping invalid line #%d in %s', line_no, pairs)
        continue
      if a not in ok_vocab or b not in ok_vocab:
        oov += 1
        if dummy4unknown:
          similarity_model.append(0.0)
          similarity_gold.append(sim)
          continue
        else:
          print('skipping line #%d with OOV words: %s', line_no, line.strip())
          continue
      similarity_gold.append(sim)  # Similarity from the dataset
      if el:
        similarity_model.append(prod(np_sum((self[a] * self[b]).reshape(cate_n, cate_k), axis=1))) # Expected likelihood similarity from the model
      else:
        similarity_model.append(self.similarity(a, b) if normalized else dot(self[a], self[b]))  # Similarity from the model
  self.vocab = original_vocab
  spearman = stats.spearmanr(similarity_gold, similarity_model)
  pearson = stats.pearsonr(similarity_gold, similarity_model)
  oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

  print(
    'Pearson correlation coefficient against %s: %f with p-value %f',
    pairs, pearson[0], pearson[1])
  print(
    'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
    pairs, spearman[0], spearman[1])
  print('Pairs with unknown words: %d' % oov)
  self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
  return pearson, spearman, oov_ratio

if __name__ == '__main__':
  if not args.vector_file or not args.sim_file:
    print('-vec and -sim are needed')
    exit()
  print('Normalize vector' if args.normalized else "No normalization")
  w2v_emb = KeyedVectors.load_word2vec_format(args.vector_file, binary=True)
  r = evaluate_word_pairs(w2v_emb, args.sim_file, case_insensitive=False, dummy4unknown=False,
                          normalized=args.normalized, el=args.expected_likelihood, cate_n=args.cate_n, cate_k=args.cate_k)
  print(r)