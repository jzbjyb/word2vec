#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from gensim.models.keyedvectors import KeyedVectors

parser = argparse.ArgumentParser()
parser.add_argument("-vec", "--vector_file", help="binary vector file")
parser.add_argument("-sim", "--sim_file", help="word similarity file")
args = parser.parse_args()

if __name__ == '__main__':
    if not args.vector_file or not args.sim_file:
        print('-vec and -sim are needed')
        exit()
    w2v_emb = KeyedVectors.load_word2vec_format(args.vector_file, binary=True)
    r = w2v_emb.evaluate_word_pairs(args.sim_file, case_insensitive=False, dummy4unknown=False)
    print(r)
