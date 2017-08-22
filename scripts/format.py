#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.keyedvectors import KeyedVectors
import sys
import numpy as np

def softmax(x, cate_n, cate_k):
    x = x.reshape(cate_n, cate_k)
    e_x = np.exp(x)
    return (e_x / e_x.sum(axis=1, keepdims=True)).reshape(-1)

if __name__ == '__main__':
    emb_path = sys.argv[1]
    pemb_path = sys.argv[2]
    cate_n = int(sys.argv[3])
    cate_k = int(sys.argv[4])
    emb = KeyedVectors.load_word2vec_format(emb_path, binary=True)
    with open(pemb_path, 'w') as fp:
        fp.write('{} {}\n'.format(len(emb.vocab), emb.vector_size))
        for w in emb.vocab:
            nemb = softmax(emb[w], cate_n, cate_k)
            fp.write('{} {}\n'.format(w, ' '.join(str(v) for v in nemb)))
    pemb = KeyedVectors.load_word2vec_format(pemb_path, binary=False)
    pemb.save_word2vec_format(pemb_path, binary=True)