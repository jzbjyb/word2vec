#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VECTOR_DATA=../vec/text8_w2v_100.bin

python eval.py -vec $VECTOR_DATA -sim ../../wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
$BIN_DIR/compute-accuracy $VECTOR_DATA 30000 < $DATA_DIR/questions-words.txt