SCRIPTS_DIR=../scripts
BIN_DIR=../bin

CC = gcc
#The -Ofast might not work with older versions of gcc; in that case, use -O2
CFLAGS = -lm -pthread -O2 -Wall -funroll-loops -Wno-unused-result

all: word2vec word2vec-prob word2vec-context-prob word2phrase distance distance-context word-analogy compute-accuracy format binary-to-text

word2vec : word2vec.cpp
	$(CC) word2vec.cpp -o ${BIN_DIR}/word2vec $(CFLAGS)
word2vec-prob : word2vec-prob.cpp
	$(CC) word2vec-prob.cpp -o ${BIN_DIR}/word2vec-prob $(CFLAGS)
word2vec-context-prob : word2vec-context-prob.cpp
	$(CC) word2vec-context-prob.cpp -o ${BIN_DIR}/word2vec-context-prob $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o ${BIN_DIR}/word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o ${BIN_DIR}/distance $(CFLAGS)
distance-context: distance-context.c
	$(CC) distance-context.c -o ${BIN_DIR}/distance-context $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o ${BIN_DIR}/word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o ${BIN_DIR}/compute-accuracy $(CFLAGS)
format: format.c
	$(CC) format.c -o ${BIN_DIR}/format $(CFLAGS)
binary-to-text: binary-to-text.c
	$(CC) binary-to-text.c -o ${BIN_DIR}/binary-to-text $(CFLAGS)	
	chmod +x ${SCRIPTS_DIR}/*.sh

clean:
	pushd ${BIN_DIR} && rm -rf word2vec word2phrase distance word-analogy compute-accuracy; popd
