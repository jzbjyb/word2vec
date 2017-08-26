#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec

#TEXT_DATA=$DATA_DIR/text8
#ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
#VECTOR_DATA=$DATA_DIR/text8-vector.bin

#TEXT_DATA=$DATA_DIR/text8
TEXT_DATA=/home/v-zhjia2/exp/data/WestburyLab.wikicorp.201004.txt.clean
ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
VECTOR_DATA=$VEC_DIR/wiki_size50-2_epoch5_novi_free.bin
PROB_VECTOR_DATA=$VEC_DIR/wiki_size50-2_epoch5_novi_free.prob.bin
CONTEXT_VECTOR_DATA=$VEC_DIR/wiki_size50-2_epoch5_novi_free.context.bin
PREDICT_VECTOR_DATA=$VEC_DIR/wiki_size50-2_epoch5_novi_free.predict.bin
CATE_N=50
CATE_K=2

pushd ${SRC_DIR} && make; popd
  
if [ ! -e $TEXT_DATA ]; then
  if [ ! -e $ZIPPED_TEXT_DATA ]; then
    wget http://mattmahoney.net/dc/text8.zip -O $ZIPPED_TEXT_DATA
  fi
  unzip $ZIPPED_TEXT_DATA
  mv text8 $TEXT_DATA
fi
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/word2vec-prob -train $TEXT_DATA -output $VECTOR_DATA -context-output $CONTEXT_VECTOR_DATA -predict_output $PREDICT_VECTOR_DATA \
-cbow 0 -cate-n $CATE_N -cate-k $CATE_K -tau 1 -window 5 -negative 0 -hs 1 -sample 1e-3 \
-threads 12 -binary 1 -report-period 0 -alpha 0.25 -eval ./eval-prob.sh -epoch 5 -kl 1 -posterior 0 -freedom 1

$BIN_DIR/format $VECTOR_DATA $PROB_VECTOR_DATA $CATE_N $CATE_K

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $VECTOR_DATA
