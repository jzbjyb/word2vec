#!/bin/bash

NGRAM_HOME=../../ngram2vec
export PATH=$NGRAM_HOME:$PATH
DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec
VEC_FILE=wiki_w2v_300_s5_m10_epoch5.bin
VECTOR_DATA=$VEC_DIR/$VEC_FILE
CONTEXT_VECTOR_DATA=$VEC_DIR/wiki_w2v_100_s5_m10_epoch5.bin
SIM_DATA=../../wordsim_eval/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
#SIM_DATA=../../wordsim_eval/SCWS/ratings.txt

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim $SIM_DATA -nor
#python compute-similarity.py -vec $VECTOR_DATA -context_vec $CONTEXT_VECTOR_DATA -sim $SIM_DATA -nor
#$BIN_DIR/compute-accuracy $VECTOR_DATA 1 0 < $DATA_DIR/questions-words.txt #> ../eval/$VEC_FILE.analogy
if [ ! -e $VEC_DIR/$VEC.words.npy ]; then
  text2numpy $VEC_DIR/$VEC.words
fi
analogy_eval SGNS $VEC_DIR/$VEC ${NGRAM_HOME}/testsets/analogy/google.txt