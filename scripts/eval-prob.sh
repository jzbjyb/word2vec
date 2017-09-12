#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
#VECTOR_DATA=../vec/wiki_size50-2_epoch5_novi_free.bin
VECTOR_DATA=../vec/test.bin
CONTEXT_VECTOR_DATA=../vec/wiki_size10-10_epoch5_realvi_kl.context.bin
SIM_DATA=../../wordsim_eval/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
#SIM_DATA=../../wordsim_eval/SCWS/ratings.txt

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim $SIM_DATA -nor #-el -cate_n 10 -cate_k 10
#python compute-similarity.py -vec $VECTOR_DATA -context_vec $CONTEXT_VECTOR_DATA -sim $SIM_DATA -nor
$BIN_DIR/compute-accuracy $VECTOR_DATA 1 30000 < $DATA_DIR/questions-words.txt