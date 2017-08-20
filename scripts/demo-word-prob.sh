#!/bin/bash

DATA_DIR=../data
BIN_DIR=../bin
SRC_DIR=../src
VEC_DIR=../vec

#TEXT_DATA=$DATA_DIR/text8
#ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
#VECTOR_DATA=$DATA_DIR/text8-vector.bin

TEXT_DATA=$DATA_DIR/text8
ZIPPED_TEXT_DATA="${TEXT_DATA}.zip"
VECTOR_DATA=$VEC_DIR/text8_size10-10_epoch20_fastlog.bin

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
time $BIN_DIR/word2vec-prob -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -cate-n 10 -cate-k 10 -tau 1 -window 3 \
-negative 0 -hs 1 -sample 1e-3 -threads 1 -binary 1 -report-period 0 -alpha 0.25 -eval ./eval-prob.sh -epoch 20

#echo -----------------------------------------------------------------------------------------------------
#echo -- distance...

#$BIN_DIR/distance $VECTOR_DATA
