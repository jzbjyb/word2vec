#!/bin/bash

NGRAM_HOME=/home/v-zhjia2/exp/ngram2vec
export PATH=$NGRAM_HOME:$PATH
DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec
#VEC=wiki_context_size100-5_epoch10_s5_m10
#VEC=wiki_size100-5_epoch20_s5_m10
#VEC=text8_size100-5_epoch30_posterior_lr05_s5_m10
#VEC=text8_context_size100-5_epoch10_s5_m10
#VEC=wiki_context_size100-5_epoch50_ds8_s5_m10_lr25
VEC=wiki_context_interaction_size100-5_epoch1_s5_m10
VEC_FILE=$VEC.bin
VECTOR_DATA=$VEC_DIR/$VEC_FILE
#VECTOR_DATA=/home/v-zhjia2/exp/multi-sense-baseline/multi-sense-skipgram/vectors-MSSG-wiki.gz
CONTEXT_VECTOR_DATA=$VEC_DIR/$VEC.context.bin
SIM_DATA=../../wordsim_eval/wordsim353_sim_rel/wordsim_similarity_goldstandard.txt
#SIM_DATA=../../wordsim_eval/SCWS/ratings.txt

pushd ${SRC_DIR} && make; popd

python compute-similarity.py -vec $VECTOR_DATA -sim $SIM_DATA -nor #-el -cate_n 10 -cate_k 10
#python compute-similarity.py -vec $VECTOR_DATA -context_vec $CONTEXT_VECTOR_DATA -sim $SIM_DATA -nor -cate_n 100 -cate_k 5
#$BIN_DIR/compute-accuracy $VECTOR_DATA 1 0 < $DATA_DIR/questions-words.txt #> ../eval/$VEC_FILE.analogy
#if [ ! -e $VEC_DIR/$VEC.words.npy ]; then
  text2numpy $VEC_DIR/$VEC.words
#fi
analogy_eval SGNS $VEC_DIR/$VEC ${NGRAM_HOME}/testsets/analogy/google.txt