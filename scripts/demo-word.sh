#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec

#TEXT_DATA=$DATA_DIR/text8
#ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
#VECTOR_DATA=$DATA_DIR/text8-vector.bin

#TEXT_DATA=$DATA_DIR/text8
TEXT_DATA=../../data/WestburyLab.wikicorp.201004.txt.clean
ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
VECTOR=wiki_w2v_500_s5_m10_epoch5
VECTOR_DATA=$VEC_DIR/${VECTOR}.bin
PROB_VECTOR_DATA=$VEC_DIR/${VECTOR}.prob.bin
SIZE=500

pushd ${SRC_DIR} && make; popd

#if [ ! -e $VECTOR_DATA ]; then
  
if [ ! -e $TEXT_DATA ]; then
  if [ ! -e $ZIPPED_TEXT_DATA ]; then
    wget http://mattmahoney.net/dc/text8.zip -O $ZIPPED_TEXT_DATA
  fi
  unzip $ZIPPED_TEXT_DATA
  mv text8 $TEXT_DATA
fi
echo -----------------------------------------------------------------------------------------------------
echo -- Training vectors...
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size $SIZE -window 5 -negative 0 -hs 1 -sample 1e-5 -min-count 10 \
-threads 12 -binary 1 -report-period 0 -eval ./eval.sh -epoch 5 -alpha 0.025 

#fi

$BIN_DIR/format $VECTOR_DATA $PROB_VECTOR_DATA $SIZE 2 0 1

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
