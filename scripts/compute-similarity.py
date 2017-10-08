#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from numpy import dot, prod, sum as np_sum, stack, exp as np_exp
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim import utils, matutils
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument("-vec", "--vector_file", help="binary vector file")
parser.add_argument("-context_vec", "--context_vector_file", help="context binary vector file")
parser.add_argument("-sim", "--sim_file", help="word similarity file")
parser.add_argument("-nor", "--normalized", help="normalize the vector", action="store_true")
parser.add_argument("-el", "--expected_likelihood", help="using expected likelihood to calculate similarity", action="store_true")
parser.add_argument("-cate_n", "--cate_n", help="category variable num", type=int, default=0)
parser.add_argument("-cate_k", "--cate_k", help="category variable size", type=int, default=0)
parser.add_argument("-top", "--top", help="evaluate only top words", type=int, default=100000000)
args = parser.parse_args()

def softmax_prob(logits, cate_n, cate_k):
  logits = logits.reshape(cate_n, cate_k)
  probs = np_exp(logits)
  return (probs / np_sum(probs, axis=1, keepdims=True)).reshape(-1)

def parse_sim_line(line, delimiter='\t', case_insensitive=True, mode=0):
  if mode == 0:
    a, b, sim = line.split(delimiter)
  elif mode == 1:
    ind, a, ap, b, bp, asen, bsen, sim = line.split(delimiter)[:-10]
    asen = asen.split()
    bsen = bsen.split()
    aind = asen.index('<b>')
    bind = bsen.index('<b>')
    asen = [w.lower() if case_insensitive else w for w in asen if w != '<b>' and w != '</b>']
    bsen = [w.lower() if case_insensitive else w for w in bsen if w != '<b>' and w != '</b>']
  if case_insensitive:
    a = a.lower()
    b = b.lower()
  sim = float(sim)
  if mode == 1:
    return a, b, sim, asen, bsen, aind, bind
  return a, b, sim, None, None, None, None

def evaluate_word_pairs(self, pairs, delimiter='\t', restrict_vocab=300000, 
                        case_insensitive=True, dummy4unknown=False, normalized=True,
                        el=False, cate_n=0, cate_k=0, mode=0, window=0, context_emb=None):
  ok_vocab = [(w, self.vocab[w]) for w in self.index2word[:restrict_vocab]]
  ok_vocab = dict((w.lower(), v) for w, v in reversed(ok_vocab)) if case_insensitive else dict(ok_vocab)
  original_vocab = self.vocab
  self.vocab = ok_vocab
  if context_emb != None:
    context_ok_vocab = [(w, context_emb.vocab[w]) for w in context_emb.index2word[:restrict_vocab]]
    context_ok_vocab = dict((w.lower(), v) for w, v in reversed(context_ok_vocab)) if case_insensitive else dict(context_ok_vocab)
    context_original_vocab = context_emb.vocab
    context_emb.vocab = context_ok_vocab

  similarity_gold = []
  similarity_model = []
  oov = 0

  for line_no, line in enumerate(utils.smart_open(pairs)):
    line = utils.to_unicode(line)
    if line.startswith('#'):
      # May be a comment
      continue
    else:
      try:
        a, b, sim, asen, bsen, aind, bind = parse_sim_line(line, delimiter=delimiter, case_insensitive=case_insensitive, mode=mode)
      except:
        #print('skipping invalid line #%d in %s' % (line_no, pairs))
        continue
      if a not in ok_vocab or b not in ok_vocab:
        oov += 1
        #print('unknown', a, b)
        if dummy4unknown:
          similarity_model.append(0.0)
          similarity_gold.append(sim)
          continue
        else:
          #print('skipping line #%d with OOV words: %s' % (line_no, line.strip()))
          continue
      similarity_gold.append(sim)  # Similarity from the dataset
      if mode == 1 and window >= 0 and context_emb != None:
        aemb = np_sum(stack([self[a]] + [context_emb[w] for w in (asen[aind-window:aind] + asen[aind+1:aind+window+1]) 
                                        if w in context_emb and self.vocab[a].index <= restrict_vocab]), axis=0)
        bemb = np_sum(stack([self[b]] + [context_emb[w] for w in (bsen[bind-window:bind] + bsen[bind+1:bind+window+1])
                                        if w in context_emb and self.vocab[b].index <= restrict_vocab]), axis=0)
      else:
        aemb = self[a]
        bemb = self[b]
      if el:
        similarity_model.append(prod(np_sum((aemb * bemb).reshape(cate_n, cate_k), axis=1))) # Expected likelihood similarity from the model
      else:
        '''
        post = softmax_prob(aemb, cate_n, cate_k).reshape(cate_n, cate_k)
        prior = softmax_prob(self[a], cate_n, cate_k).reshape(cate_n, cate_k)
        print(a, len([context_emb[w] for w in (asen[aind-window:aind] + asen[aind+1:aind+window+1]) if w in context_emb]),
              b, len([context_emb[w] for w in (bsen[bind-window:bind] + bsen[bind+1:bind+window+1]) if w in context_emb]))
        print(asen[aind-window:aind] + asen[aind+1:aind+window+1])
        print(bsen[bind-window:bind] + bsen[bind+1:bind+window+1])
        print(np.stack([prior, post], axis=1)[:10])
        input()
        '''
        similarity_model.append(dot(matutils.unitvec(aemb), matutils.unitvec(bemb)) if normalized else dot(aemb, bemb))  # Similarity from the model
        #similarity_model.append(dot(matutils.unitvec(softmax_prob(aemb, cate_n, cate_k)), matutils.unitvec(softmax_prob(bemb, cate_n, cate_k))) if normalized else dot(aemb, bemb))  # Similarity from the model
  self.vocab = original_vocab
  if context_emb != None:
    context_emb.vocab = context_original_vocab
  spearman = stats.spearmanr(similarity_gold, similarity_model)
  pearson = stats.pearsonr(similarity_gold, similarity_model)
  oov_ratio = float(oov) / (len(similarity_gold) + oov) * 100

  #print(
  #  'Pearson correlation coefficient against %s: %f with p-value %f',
  #  pairs, pearson[0], pearson[1])
  #print(
  #  'Spearman rank-order correlation coefficient against %s: %f with p-value %f',
  #  pairs, spearman[0], spearman[1])
  print('Pairs with unknown words: %d' % oov)
  self.log_evaluate_word_pairs(pearson, spearman, oov_ratio, pairs)
  return pearson, spearman, oov_ratio

if __name__ == '__main__':
  if not args.vector_file or not args.sim_file:
    print('-vec and -sim are needed')
    exit()
  mode = 1 if 'SCWS' in args.sim_file else 0
  print('Evaluate %d' % args.top)
  print('Normalize vector' if args.normalized else "No normalization")
  print('Context similarity file' if mode else "No context similarity file")
  w2v_emb = KeyedVectors.load_word2vec_format(args.vector_file, binary=True)
  if args.context_vector_file:
    context_emb = KeyedVectors.load_word2vec_format(args.context_vector_file, binary=True)
    r = evaluate_word_pairs(w2v_emb, args.sim_file, case_insensitive=True, dummy4unknown=False, restrict_vocab=args.top,
                            normalized=args.normalized, el=args.expected_likelihood, cate_n=args.cate_n, cate_k=args.cate_k, 
                            mode=mode, window=5, context_emb=context_emb)
  else:
    r = evaluate_word_pairs(w2v_emb, args.sim_file, case_insensitive=True, dummy4unknown=False, restrict_vocab=args.top,
                            normalized=args.normalized, el=args.expected_likelihood, cate_n=args.cate_n, cate_k=args.cate_k,
                            mode=mode)
  print(r)