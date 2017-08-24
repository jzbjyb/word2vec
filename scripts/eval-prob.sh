#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
#VECTOR_DATA=../vec/wiki_size10-10_epoch5_vi_nokl_thread12.bin
VECTOR_DATA=../vec/text8_size50-2_epoch1_novi.bin
CONTEXT_VECTOR_DATA=../vec/wiki_size10-10_epoch5_realvi_kl.context.bin
SIM_DATA=../../wordsim_eval/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
#SIM_DATA=../../wordsim_eval/SCWS/ratings.txt

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim $SIM_DATA -nor
#python compute-similarity.py -vec $VECTOR_DATA -context_vec $CONTEXT_VECTOR_DATA -sim $SIM_DATA -nor
#$BIN_DIR/compute-accuracy $VECTOR_DATA 1 30000 < $DATA_DIR/questions-words.txt