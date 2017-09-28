#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec
VEC_FILE=text8_size100-5_epoch40_posterior_lr05_s5_m10.bin
VECTOR_DATA=$VEC_DIR/$VEC_FILE
CONTEXT_VECTOR_DATA=$VEC_DIR/text8_size100-5_epoch40_posterior_lr05_s5_m10.context.bin
SIM_DATA=../../wordsim_eval/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
#SIM_DATA=../../wordsim_eval/SCWS/ratings.txt

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim $SIM_DATA -nor #-el -cate_n 10 -cate_k 10
#python compute-similarity.py -vec $VECTOR_DATA -context_vec $CONTEXT_VECTOR_DATA -sim $SIM_DATA -nor
$BIN_DIR/compute-accuracy $VECTOR_DATA 1 0 < $DATA_DIR/questions-words.txt > ../eval/$VEC_FILE.analogy