#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VECTOR_DATA=../vec/wiki_size10-10_epoch5_vi_nokl_thread12.prob.bin

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim ../../wordsim353_sim_rel/wordsim_similarity_goldstandard.txt -el -cate_n 10 -cate_k 10
$BIN_DIR/compute-accuracy $VECTOR_DATA 1 30000 < $DATA_DIR/questions-words.txt