#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors

parser = argparse.ArgumentParser()
parser.add_argument("-vec", "--vector_file", help="binary vector file")
parser.add_argument("-cate_n", "--cate_n", help="category variable num", type=int, default=0)
parser.add_argument("-cate_k", "--cate_k", help="category variable size", type=int, default=0)
parser.add_argument("-top", "--top", help="top words to output", type=int, default=20)
parser.add_argument("-file", "--filename", help="output file name (csv)")
args = parser.parse_args()

if __name__ == '__main__':
  w2v_emb = KeyedVectors.load_word2vec_format(args.vector_file, binary=True)
  words_in_order = np.array(w2v_emb.index2word)
  words = {}
  result = []
  for n in range(args.cate_n):
    for k in range(args.cate_k):
      c = n * args.cate_k + k
      rank = np.argsort(-w2v_emb.syn0[:, c])[:args.top]
      for w in words_in_order[rank]:
        if w not in words:
          words[w] = 0
        words[w] += 1
      result.append(list(zip(words_in_order[rank], w2v_emb.syn0[:, c][rank])))
  print('total {}/{} unique words'.format(len(words), args.top * args.cate_n * args.cate_k))
  freq_words = [(w[0], w[1], w2v_emb.vocab[w[0]].index) for w in sorted(words.items(), key=lambda x:-x[1])]
  print('frequence words are {}'.format(freq_words[:10]))
  data = np.concatenate([d for d in result], axis=1)
  column = np.array([['{},{}'.format(k // args.cate_k, k % args.cate_k)] * 2 for k in range(args.cate_n * args.cate_k)]).flatten()
  df = pd.DataFrame(data, columns=column)
  df.to_csv(args.filename + '.csv', index=False)