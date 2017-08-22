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
VECTOR_DATA=$VEC_DIR/wiki_w2v_100.bin

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
time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 100 -window 5 -negative 0 -hs 1 -sample 1e-3 \
-threads 12 -binary 1 -report-period 0 -eval ./eval.sh
  
#fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $DATA_DIR/$VECTOR_DATA
