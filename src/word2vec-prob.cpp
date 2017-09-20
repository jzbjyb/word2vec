//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
//#include "sse2.h"

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

//uint32_t *SEED = &(uint32_t){2017};
uint32_t XORSHF_RAND_MAX = (1 << 32) - 1;
const float count_power = 1;
const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
  long long cn;
  real cn_e;
  int *point;
  char *word, *code, codelen;
};

char train_file[MAX_STRING], output_file[MAX_STRING], context_output_file[MAX_STRING], predict_output_file[MAX_STRING], eval_sh[MAX_STRING];
char pre_train_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100, syn1_layer1_size, r_layer1_size;
long long cate_n = 1, cate_k = 100, r_cate_k;
long long train_words = 0, word_count_actual = 0, last_word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, alpha_decay = 1.0 / 3.0, sample = 0, tau = 1, min_tau = 0.2, starting_tau, all_prob = 0, var_scale = 1;
real *syn0, *syn0eoe, *syn1, *syn1p, *syn1neg, *expTable;
real *adam_m_syn0, *adam_v_syn0, *adam_m_syn1, *adam_v_syn1, *adam_m_syn1p, *adam_v_syn1p, \
     adam_beta1 = 0.9, adam_beta1p = 0.1, adam_beta2 = 0.999, adam_beta2p = 0.001, adam_eps = 1e-8, \
     *all_grad;
clock_t start;

int hs = 1, negative = 0;
int report_period = 10, num_epoch = 1, kl = 0, ent = 0, rollback = 0, posterior = 0, freedom = 0, adam = 0, pre_train = 0, 
    binary_one = 0, hard_sigm = 0, eoe = 0;
const int table_size = 1e8;
int *table;

inline double fast_exp(double x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  /*
  // TODO: small x result in free bug
  if (abs(x) < 1e-5) {
    if(x >= 0) x = 1e-5;
    else x = -1e-5;
  } */
  return x;
}

inline float fast_log2 (float val) {
   int * const exp_ptr = reinterpret_cast <int *> (&val);
   int x = *exp_ptr;
   const int log_2 = ((x >> 23) & 255) - 128;
   x &= ~(255 << 23);
   x += 127 << 23;
   *exp_ptr = x;
   val = ((-1.0f/3) * val + 2) * val - 2.0f/3;
   return (val + log_2);
} 

inline float fast_log (const float &val) {
   return (fast_log2 (val) * 0.69314718f);
}

static unsigned long x=123456789, y=362436069, z=521288629;

unsigned long xorshf96(void) { //period 2^96-1
  unsigned long t;
  x ^= x << 16;
  x ^= x >> 5;
  x ^= x << 1;
  t = x;
  x = y;
  y = z;
  z = t ^ x ^ y;
  return z;
}

/*
uint32_t xorshift32(uint32_t state[static 1])
{
  uint32_t x = state[0];
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  state[0] = x;
  return x;
}
*/

int fastrand() {
  int g_seed = (214013*g_seed+2531011); 
  return (g_seed>>16)&0x7FFF; 
} 

void InitUnigramTable() {
  int a, i;
  long long train_words_pow = 0;
  real d1, power = 0.75;
  table = (int *)malloc(table_size * sizeof(int));
  if (table == NULL) {
    fprintf(stderr, "cannot allocate memory for the table\n");
    exit(1);
  }
  for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
  i = 0;
  d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
  for (a = 0; a < table_size; a++) {
    table[a] = i;
    if (a / (real)table_size > d1) {
      i++;
      d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
    }
    if (i >= vocab_size) i = vocab_size - 1;
  }
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
  int a = 0, ch;
  while (!feof(fin)) {
    ch = fgetc(fin);
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
  unsigned long long a, hash = 0;
  for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
  hash = hash % vocab_hash_size;
  return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
  unsigned int hash = GetWordHash(word);
  while (1) {
    if (vocab_hash[hash] == -1) return -1;
    if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
    hash = (hash + 1) % vocab_hash_size;
  }
  return -1;
}

// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
  char word[MAX_STRING];
  ReadWord(word, fin);
  if (feof(fin)) return -1;
  return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
  unsigned int hash, length = strlen(word) + 1;
  if (length > MAX_STRING) length = MAX_STRING;
  vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
  strcpy(vocab[vocab_size].word, word);
  vocab[vocab_size].cn = 0;
  vocab_size++;
  // Reallocate memory if needed
  if (vocab_size + 2 >= vocab_max_size) {
    vocab_max_size += 1000;
    vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
  }
  hash = GetWordHash(word);
  while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
  vocab_hash[hash] = vocab_size - 1;
  return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

void DestroyVocab() {
  int a;

  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      free(vocab[a].word);
    }
    if (vocab[a].code != NULL) {
      free(vocab[a].code);
    }
    if (vocab[a].point != NULL) {
      free(vocab[a].point);
    }
  }
  free(vocab[vocab_size].word);
  free(vocab);
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
  int a, size;
  unsigned int hash;
  // Sort the vocabulary and keep </s> at the first position
  qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  size = vocab_size;
  train_words = 0;
  for (a = 1; a < size; a++) { // Skip </s>
    // Words occuring less than min_count times will be discarded from the vocab
    if (vocab[a].cn < min_count) {
      vocab_size--;
      free(vocab[a].word);
      vocab[a].word = NULL;
    } else {
      // Hash will be re-computed, as after the sorting it is not actual
      hash=GetWordHash(vocab[a].word);
      while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
      vocab_hash[hash] = a;
      train_words += vocab[a].cn;
      vocab[a].cn_e = pow(vocab[a].cn, count_power);
    }
  }
  vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
  // Allocate memory for the binary tree construction
  for (a = 0; a < vocab_size; a++) {
    vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
    vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
  }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
  int a, b = 0;
  unsigned int hash;
  for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
    vocab[b].cn = vocab[a].cn;
    vocab[b].word = vocab[a].word;
    b++;
  } else free(vocab[a].word);
  vocab_size = b;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  for (a = 0; a < vocab_size; a++) {
    // Hash will be re-computed, as it is not actual
    hash = GetWordHash(vocab[a].word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = a;
  }
  fflush(stdout);
  min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
void CreateBinaryTree() {
  long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
  char code[MAX_CODE_LENGTH];
  long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
  for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
  for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
  pos1 = vocab_size - 1;
  pos2 = vocab_size;
  // Following algorithm constructs the Huffman tree by adding one node at a time
  for (a = 0; a < vocab_size - 1; a++) {
    // First, find two smallest nodes 'min1, min2'
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min1i = pos1;
        pos1--;
      } else {
        min1i = pos2;
        pos2++;
      }
    } else {
      min1i = pos2;
      pos2++;
    }
    if (pos1 >= 0) {
      if (count[pos1] < count[pos2]) {
        min2i = pos1;
        pos1--;
      } else {
        min2i = pos2;
        pos2++;
      }
    } else {
      min2i = pos2;
      pos2++;
    }
    count[vocab_size + a] = count[min1i] + count[min2i];
    parent_node[min1i] = vocab_size + a;
    parent_node[min2i] = vocab_size + a;
    binary[min2i] = 1;
  }
  // Now assign binary code to each vocabulary word
  for (a = 0; a < vocab_size; a++) {
    b = a;
    i = 0;
    while (1) {
      code[i] = binary[b];
      point[i] = b;
      i++;
      b = parent_node[b];
      if (b == vocab_size * 2 - 2) break;
    }
    vocab[a].codelen = i;
    vocab[a].point[0] = vocab_size - 2;
    for (b = 0; b < i; b++) {
      vocab[a].code[i - b - 1] = code[b];
      vocab[a].point[i - b] = point[b] - vocab_size;
    }
  }
  free(count);
  free(binary);
  free(parent_node);
}

void LearnVocabFromTrainFile() {
  char word[MAX_STRING];
  FILE *fin;
  long long a, i;
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  vocab_size = 0;
  AddWordToVocab((char *)"</s>");
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    train_words++;
    if ((debug_mode > 1) && (train_words % 100000 == 0)) {
      printf("%lldK%c", train_words / 1000, 13);
      fflush(stdout);
    }
    i = SearchVocab(word);
    if (i == -1) {
      a = AddWordToVocab(word);
      vocab[a].cn = 1;
    } else vocab[i].cn++;
    if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  file_size = ftell(fin);
  fclose(fin);
}

void SaveVocab() {
  long long i;
  FILE *fo = fopen(save_vocab_file, "wb");
  for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
  fclose(fo);
}

void SaveVec() {
  FILE *fo;
  long a, b;
  real s;
  // save word embedding
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    exit(1);
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      fprintf(fo, "%s ", vocab[a].word);
    }
    if (binary) for (b = 0; b < layer1_size; b++) {
      s = syn0[a * layer1_size + b] - all_grad[a * layer1_size + b];
      fwrite(&s, sizeof(real), 1, fo);
    }
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  // save context embedding
  fo = fopen(context_output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", context_output_file);
    exit(1);
  }
  fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
  for (a = 0; a < vocab_size; a++) {
    if (vocab[a].word != NULL) {
      fprintf(fo, "%s ", vocab[a].word);
    }
    if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn1p[a * layer1_size + b], sizeof(real), 1, fo);
    else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn1p[a * layer1_size + b]);
    fprintf(fo, "\n");
  }
  fclose(fo);
  // save predict (output) embedding
  // TODO: hierarchy softmax is hard to store
  if (predict_output_file[0] != 0) {
    fo = fopen(predict_output_file, "wb");
    if (fo == NULL) {
      fprintf(stderr, "Cannot open %s: permission denied\n", predict_output_file);
      exit(1);
    }
    fprintf(fo, "%lld %lld\n", vocab_size, syn1_layer1_size);
    for (a = 0; a < vocab_size; a++) {
      if (vocab[a].word != NULL) {
        fprintf(fo, "%s ", vocab[a].word);
      }
      if (binary) for (b = 0; b < syn1_layer1_size; b++) fwrite(&syn1[a * syn1_layer1_size + b], sizeof(real), 1, fo);
      else for (b = 0; b < syn1_layer1_size; b++) fprintf(fo, "%lf ", syn1[a * syn1_layer1_size + b]);
      fprintf(fo, "\n");
    }
    fclose(fo);
  }
}

void ReadVocab() {
  long long a, i = 0;
  char c;
  char word[MAX_STRING];
  FILE *fin = fopen(read_vocab_file, "rb");
  if (fin == NULL) {
    printf("Vocabulary file not found\n");
    exit(1);
  }
  for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
  vocab_size = 0;
  while (1) {
    ReadWord(word, fin);
    if (feof(fin)) break;
    a = AddWordToVocab(word);
    fscanf(fin, "%lld%c", &vocab[a].cn, &c);
    i++;
  }
  SortVocab();
  if (debug_mode > 0) {
    printf("Vocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
  }
  fin = fopen(train_file, "rb");
  if (fin == NULL) {
    printf("ERROR: training data file not found!\n");
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
}

void InitNet() {
  long long a, b;
  FILE *fin;
  a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
  if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
  if (hs) {
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * syn1_layer1_size * sizeof(real));
    if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < syn1_layer1_size; b++) for (a = 0; a < vocab_size; a++)
      syn1[a * syn1_layer1_size + b] = 0;
    a = posix_memalign((void **)&syn1p, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1p == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
      syn1p[a * layer1_size + b] = 0;
      //syn1p[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
    a = posix_memalign((void **)&all_grad, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (all_grad == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
      all_grad[a * layer1_size + b] = 0;
    if (adam) {
      a = posix_memalign((void **)&adam_m_syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
      a = posix_memalign((void **)&adam_v_syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
      a = posix_memalign((void **)&adam_m_syn1, 128, (long long)vocab_size * syn1_layer1_size * sizeof(real));
      a = posix_memalign((void **)&adam_v_syn1, 128, (long long)vocab_size * syn1_layer1_size * sizeof(real));
      a = posix_memalign((void **)&adam_m_syn1p, 128, (long long)vocab_size * layer1_size * sizeof(real));
      a = posix_memalign((void **)&adam_v_syn1p, 128, (long long)vocab_size * layer1_size * sizeof(real));
      if (adam_m_syn0 == NULL || adam_v_syn0 == NULL || adam_m_syn1 == NULL || adam_v_syn1 == NULL || adam_m_syn1p == NULL || adam_v_syn1p == NULL) {
        printf("Memory allocation failed\n"); exit(1);
      }
      for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++) {
        adam_m_syn0[a * layer1_size + b] = adam_v_syn0[a * layer1_size + b] = 0;
        adam_m_syn1p[a * layer1_size + b] = adam_v_syn1p[a * layer1_size + b] = 0;
      }
      for (b = 0; b < syn1_layer1_size; b++) for (a = 0; a < vocab_size; a++)
        adam_m_syn1[a * syn1_layer1_size + b] = adam_v_syn1[a * syn1_layer1_size + b] = 0;
    }
    if (eoe > 0) {
      a = posix_memalign((void **)&syn0eoe, 128, (long long)layer1_size * eoe * sizeof(real));
      if (syn0eoe == NULL) {printf("Memory allocation failed\n"); exit(1);}
      if (eoe == 1) for (a = 0; a < cate_n; a++) for (b = 0; b < cate_k; b++) 
        syn0eoe[a * cate_k + b] = -1.0 + 2.0 / (cate_k - 1) * b;
      else for (b = 0; b < eoe; b++) for (a = 0; a < layer1_size; a++)
        syn0eoe[a * eoe + b] = (rand() / (real)RAND_MAX - 0.5) / syn1_layer1_size;
      //for (b = 0; b < eoe; b++) for (a = 0; a < layer1_size; a++)
      //  syn0eoe[a * eoe + b] = 0;
      //for (a = 0; a < cate_n; a++) for (b = 0; b < cate_k; b++) for(c = 0; c < eoe; c++) {
      //  if (b == c) syn0eoe[a * cate_k * eoe + b * eoe + c] = 1;
      //  else syn0eoe[a * cate_k * eoe + b * eoe + c] = 0;
      //}
    }
  }
  if (negative>0) {
    a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
     syn1neg[a * layer1_size + b] = 0;
  }
  if (pre_train_file[0] == 0 || (fin = fopen(pre_train_file, "rb")) == NULL)
    for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
      syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
      //syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / 5;
  else {
    long long pre_vocab_size, pre_layer1_size;
    char pre_ch, pre_word[MAX_STRING];
    fin = fopen(pre_train_file, "rb");
    printf("initialize from file [%s]\n", pre_train_file);
    fscanf(fin, "%lld", &pre_vocab_size);
    fscanf(fin, "%lld", &pre_layer1_size);
    if (pre_vocab_size != vocab_size || pre_layer1_size != layer1_size) {
      printf("pre vector file not compatible\n");
      exit(1);
    }
    for (a = 0; a < vocab_size; a++) {
      fscanf(fin, "%s%c", pre_word, &pre_ch);
      for (b = 0; b < layer1_size; b++) fread(&syn0[a * layer1_size + b], sizeof(float), 1, fin);
    }
    fclose(fin);
  }
  CreateBinaryTree();
}

void DestroyNet() {
  if (syn0 != NULL) {
    free(syn0);
  }
  if (syn1 != NULL) {
    free(syn1);
  }
  if (syn1p != NULL) {
    free(syn1p);
  }
  if (syn1neg != NULL) {
    free(syn1neg);
  }
  if (all_grad != NULL) free(all_grad);
  if (adam_m_syn0 != NULL) free(adam_m_syn0);
  if (adam_v_syn0 != NULL) free(adam_v_syn0);
  if (adam_m_syn1 != NULL) free(adam_m_syn1);
  if (adam_v_syn1 != NULL) free(adam_v_syn1);
  if (adam_m_syn1p != NULL) free(adam_m_syn1p);
  if (adam_v_syn1p != NULL) free(adam_v_syn1p);
  if (syn0eoe != NULL) free(syn0eoe);
}

real Clip(real x) {
  x = x < 1e-5 ? 1e-5 : x;
  x = x > (real)(1 - 1e-5) ? (real)(1 - 1e-5) : x;
  return x;
}

real HardSigm(real x) {
  x = (x + 1) / 2;
  return Clip(x);
}

real HardSigmGrad(real x) {
  if ((x > 1e-5) && (x < (real)(1 - 1e-5))) return 0.5;
  return 0;
}

void HardSigmProb(real *vec1, real *vec2, real *prob) {
  long long c1;
  real x;
  for (c1 = 0; c1 < cate_n; c1++) {
    x = vec1[c1] + (vec2 != NULL ? vec2[c1] : 0);
    prob[c1 * r_cate_k] = HardSigm(x);
    prob[c1 * r_cate_k + 1] = HardSigm(-x);
  }
}

void Softmax(real *vec1, real *vec2, real *prob) {
  real prob_sum, p;
  long long c1, c2;
  for (c1 = 0; c1 < cate_n; c1++) {
    prob_sum = prob[c1 * r_cate_k + r_cate_k - 1] = freedom;
    for (c2 = 0; c2 < cate_k; c2++) {
      if (vec2 == NULL) p = fast_exp(vec1[c1 * cate_k + c2]);
      else p = fast_exp(vec1[c1 * cate_k + c2] + vec2[c1 * cate_k + c2]);
      prob_sum += p;
      prob[c1 * r_cate_k + c2] = p;
    }
    for (c2 = 0; c2 < r_cate_k; c2++) {
      prob[c1 * r_cate_k + c2] /= prob_sum;
    }
  }
}

void HardSigmSample(real *vec1, real *vec2, unsigned long long *next_random, unsigned int *rr, 
                    real *prob_app, real *cate, long long *pos) {
  real unit, x;
  bool act;
  long long c1;
  for (c1 = 0; c1 < cate_n; c1++) {
    x = vec1[c1] + (vec2 != NULL ? vec2[c1] : 0);
    unit = (real)rand_r(rr) / (real)RAND_MAX;
    act = unit <= HardSigm(x);
    if (act) cate[c1] = 1;
    else if (binary_one) cate[c1] = -1;
    else cate[c1] = 0;
    if (pos != NULL) pos[c1] = act ? 0 : 1;
    prob_app[c1] = Clip(0.5 / tau * (x - 2 * unit + 1) + 0.5);
  }
}

void GumbelSoftmax(real *vec1, real *vec2, unsigned long long *next_random, unsigned int *rr, 
                   real *prob_app, real *cate, long long *pos) {
  real maxi, gumbel, cur, prob_sum;
  long long c1, c2, argmaxi = -1;
  for (c1 = 0; c1 < cate_n; c1++) {
    maxi = -1e15;
    argmaxi = -1;
    prob_sum = freedom;
    for (c2 = 0; c2 < cate_k; c2++) {
      gumbel = -fast_log(-fast_log((real)rand_r(rr) / (real)RAND_MAX + 1e-15) + 1e-15);
      //gumbel = -fast_log(-fast_log((real)rand() / (real)RAND_MAX + 1e-15) + 1e-15);
      //*(next_random) = *(next_random) * (unsigned long long)25214903917 + 11;
      //gumbel = -fast_log(-fast_log((real)(*(next_random) % RAND_MAX) / (real)RAND_MAX + 1e-15) + 1e-15);
      //gumbel = -log(-log((real)xorshift32(SEED) / (real)XORSHF_RAND_MAX + 1e-15) + 1e-15);
      //gumbel = -log(-log((real)fastrand() / (real)RAND_MAX + 1e-15) + 1e-15);
      //gumbel = -log(-log((real)rand_sse() / (real)RAND_MAX + 1e-15) + 1e-15);
      //gumbel = -log(-log((real)rand() / (real)RAND_MAX + 1e-15) + 1e-15);
      cur = gumbel + vec1[c1 * cate_k + c2];
      if (vec2 != NULL) cur += vec2[c1 * cate_k + c2];
      if (cur > maxi) {
        maxi =  cur;
        argmaxi = c2;
      }
      cate[c1 * cate_k + c2] = 0;
      prob_app[c1 * cate_k + c2] = fast_exp((cur - gumbel) / tau);
      prob_sum += prob_app[c1 * cate_k + c2];
    }
    for (c2 = 0; c2 < cate_k; c2++) prob_app[c1 * cate_k + c2] /= prob_sum;
    if (freedom) {
      gumbel = -fast_log(-fast_log((real)rand_r(rr) / (real)RAND_MAX + 1e-15) + 1e-15);
      if (gumbel > maxi) {
        maxi = gumbel;
        argmaxi = cate_k;
      }
    }
    if (argmaxi < cate_k) cate[c1 * cate_k + argmaxi] = 1;
    else if (binary_one) cate[c1 * cate_k] = -1;
    if (pos != NULL) pos[c1] = argmaxi;
  }
}

real AdamGrad(real *m, real *v, real grad) {
  *m = adam_beta1 * *m + (1 - adam_beta1) * grad;
  *v = adam_beta2 * *v + (1 - adam_beta2) * grad * grad;
  return (alpha * *m / adam_beta1p) / (sqrt(*v / adam_beta2p) + adam_eps);
}

void *TrainModelThread(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, last_report_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l11, l2, c, target, label;
  long long c1, c2, c3;
  real ag1, ag2;
  unsigned long long next_random = (long long)id;
  unsigned int rr = *((int*)(&id));
  real f, g, this_prob = 0;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(syn1_layer1_size, sizeof(real));
  real *neu1e_ae = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e_prob = (real *)calloc(r_layer1_size, sizeof(real));
  real *gs_syn0 = (real *)calloc(layer1_size, sizeof(real));
  real *gs_syn0_eoe = (real *)calloc(syn1_layer1_size, sizeof(real));
  long long *pos = (long long *)calloc(cate_n, sizeof(long long));
  real *prob_syn1p_app = (real *)calloc(layer1_size, sizeof(real));
  real *prob_syn1p = (real *)calloc(r_layer1_size, sizeof(real));
  real *prob_syn0 = (real *)calloc(r_layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  if (fi == NULL) {
    fprintf(stderr, "no such file or directory: %s", train_file);
    exit(1);
  }
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (report_period > 0 && word_count - last_report_word_count > train_words / num_threads / report_period) {
      last_report_word_count = word_count;
      SaveVec();
      printf("--- eval ---");
      system(eval_sh);
    }
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Tau: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha, tau,
         word_count_actual / (real)(train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      // adjust adam parameter
      if (adam) {
        adam_beta1p = 1 - (1 - adam_beta1p) * adam_beta1;
        adam_beta2p = 1 - (1 - adam_beta2p) * adam_beta2;
      }
      // adjust learning rate
      if (!adam) {
        alpha = starting_alpha * (1 - word_count_actual / (real)(train_words /** num_epoch*/ + 1));
        //if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
        // if (alpha < starting_alpha * 0.3) alpha = starting_alpha * 0.3;
        if (alpha < starting_alpha * 0.3) alpha = starting_alpha * 0.3 * \
          (1 - (word_count_actual - 0.7 * train_words) / (real)(train_words * num_epoch + 1 - 0.7 * train_words));
      }
      //if (!adam) {
      //  alpha = starting_alpha * (1 - alpha_decay) * (1 - (word_count_actual - last_word_count_actual) / (real)(train_words + 1)) + starting_alpha * alpha_decay;
      //}
      // adjust Gumbel-Max temperature
      //tau = starting_tau * (1 - word_count_actual / (real)(train_words * 5 + 1));
      //if (tau < min_tau) tau = min_tau;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < syn1_layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
      }
      if (hs) for (d = 0; d < vocab[word].codelen; d++) {
        f = 0;
        l2 = vocab[word].point[d] * layer1_size;
        // Propagate hidden -> output
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
        if (f <= -MAX_EXP) continue;
        else if (f >= MAX_EXP) continue;
        else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
        // 'g' is the gradient multiplied by the learning rate
        g = (1 - vocab[word].code[d] - f) * alpha;
        // Propagate errors output -> hidden
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
        // Learn weights hidden -> output
        for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
      }
      // NEGATIVE SAMPLING
      if (negative > 0) for (d = 0; d < negative + 1; d++) {
        if (d == 0) {
          target = word;
          label = 1;
        } else {
          next_random = next_random * (unsigned long long)25214903917 + 11;
          target = table[(next_random >> 16) % table_size];
          if (target == 0) target = next_random % (vocab_size - 1) + 1;
          if (target == word) continue;
          label = 0;
        }
        l2 = target * layer1_size;
        f = 0;
        for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
        if (f > MAX_EXP) g = (label - 1) * alpha;
        else if (f < -MAX_EXP) g = (label - 0) * alpha;
        else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
        for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
        for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
      }
      // hidden -> in
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        l1 = last_word * layer1_size;
        l11 = word * layer1_size;
        for (c = 0; c < syn1_layer1_size; c++) neu1e[c] = 0;
        // HIERARCHICAL SOFTMAX
        if (hs) {
          if (pre_train <= 0) {
            if (posterior) {
              // real posterior
              if (hard_sigm) HardSigmSample(syn0 + l1, syn1p + l11, &next_random, &rr, prob_syn1p_app, gs_syn0, pos);
              else GumbelSoftmax(syn0 + l1, syn1p + l11, &next_random, &rr, prob_syn1p_app, gs_syn0, pos);
            } else {
              // fake posterior (which is actually equal to prior)
              if (hard_sigm) HardSigmSample(syn0 + l1, NULL, &next_random, &rr, prob_syn1p_app, gs_syn0, pos);
              else GumbelSoftmax(syn0 + l1, NULL, &next_random, &rr, prob_syn1p_app, gs_syn0, pos);
            }
            if (eoe > 0) {
              // embedding of embedding
              if (binary_one) {
                for (c1 = 0; c1 < cate_n; c1++) for (c2 = 0; c2 < eoe; c2++) 
                  gs_syn0_eoe[c1 * eoe + c2] = syn0eoe[c1 * eoe + c2] * gs_syn0[c1];
              }
              else for (c1 = 0; c1 < cate_n; c1++) {
                if (pos[c1] < cate_k) for (c2 = 0; c2 < eoe; c2++) 
                  gs_syn0_eoe[c1 * eoe + c2] = syn0eoe[c1 * cate_k * eoe + pos[c1] * eoe + c2];
                else for (c2 = 0; c2 < eoe; c2++) gs_syn0_eoe[c1 * eoe + c2] = 0;
              }
            }
          }
          for (d = 0; d < vocab[word].codelen; d++) {
            f = 0;
            l2 = vocab[word].point[d] * syn1_layer1_size;
            // Propagate hidden -> output
            if (pre_train > 0) for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
            else {
              if (eoe > 0) {
                for (c = 0; c < syn1_layer1_size; c++) f += gs_syn0_eoe[c] * syn1[c + l2];
              } else {
                if (binary_one) for (c = 0; c < layer1_size; c++) f += gs_syn0[c] * syn1[c + l2];
                else for (c1 = 0; c1 < cate_n; c1++) if (pos[c1] < cate_k) f += syn1[c1 * cate_k + pos[c1] + l2];
                //if (next_random % 100 > 95) printf("the f is %lf\n", f);
              }
              f /= var_scale;
            }
            if (vocab[word].code[d]) this_prob += fast_log(1 / (1 + fast_exp(f)));
            else this_prob += fast_log(fast_exp(f) / (1 + fast_exp(f)));
            if (f <= -MAX_EXP) continue;
            else if (f >= MAX_EXP) continue;
            else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient not multiplied by the learning rate
            g = 1 - vocab[word].code[d] - f;
            // Propagate errors output -> hidden
            for (c = 0; c < syn1_layer1_size; c++) neu1e[c] += g * syn1[c + l2];
            // Learn weights hidden -> output
            if (pre_train > 0) for (c = 0; c < layer1_size; c++) syn1[c + l2] += alpha * g * syn0[c + l1];
            else {
              if (eoe) {
                // TODO: no adam for eoe
                for (c = 0; c < syn1_layer1_size; c++) syn1[c + l2] += alpha * g * gs_syn0_eoe[c] / var_scale;
              } else {
                if (adam) {
                  for (c = 0; c < layer1_size; c++) adam_m_syn1[c + l2] = adam_beta1 * adam_m_syn1[c + l2] + (1 - adam_beta1) * g * gs_syn0[c] / var_scale;
                  for (c = 0; c < layer1_size; c++) adam_v_syn1[c + l2] = adam_beta1 * adam_v_syn1[c + l2] + (1 - adam_beta1) * g * g * gs_syn0[c] * gs_syn0[c] / var_scale / var_scale;
                }
                if (adam) for (c = 0; c < layer1_size; c++) syn1[c + l2] += alpha * adam_m_syn1[c + l2] / adam_beta1p / (sqrt(adam_v_syn1[c + l2] / adam_beta2p) + adam_eps); //AdamGrad(adam_m_syn1 + c + l2, adam_v_syn1 + c + l2, g * gs_syn0[c]);
                else if (binary_one) for (c = 0; c < layer1_size; c++) syn1[c + l2] += alpha * g * gs_syn0[c] / var_scale;
                else for (c1 = 0; c1 < cate_n; c1++) if (pos[c1] < cate_k) syn1[c1 * cate_k + pos[c1] + l2] += alpha * g / var_scale;
              }
            }
          }
        }
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
            target = word;
            label = 1;
          } else {
            next_random = next_random * (unsigned long long)25214903917 + 11;
            target = table[(next_random >> 16) % table_size];
            if (target == 0) target = next_random % (vocab_size - 1) + 1;
            if (target == word) continue;
            label = 0;
          }
          l2 = target * layer1_size;
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
        }
        // Learn weights input -> hidden
        if (negative > 0) for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
        if (hs && pre_train > 0) for (c = 0; c < layer1_size; c++) syn0[c + l1] += alpha * neu1e[c];
        if (hs && pre_train <= 0) {
          if (eoe > 1) {
            // derivative of the eoe
            if (binary_one) for (c1 = 0; c1 < cate_n; c1++) for (c2 = 0; c2 < eoe; c2++)
              syn0eoe[c1 * eoe + c2] += alpha * gs_syn0[c1] * neu1e[c1 * eoe + c2] / var_scale;
            else for (c1 = 0; c1 < cate_n; c1++) {
              if (pos[c1] < cate_k) for (c2 = 0; c2 < eoe; c2++)
                syn0eoe[c1 * cate_k * eoe + pos[c1] * eoe + c2] += alpha * neu1e[c1 * eoe + c2] / var_scale;
            }
          }
          // derivative of reconstruction error and KL divergence
          if (posterior && kl) {
            if (hard_sigm) {
              HardSigmProb(syn0 + l1, syn1p + l11, prob_syn1p);
              HardSigmProb(syn0 + l1, NULL, prob_syn0);
            } else {
              Softmax(syn0 + l1, syn1p + l11, prob_syn1p);
              Softmax(syn0 + l1, NULL, prob_syn0);
            }
          } else if (!posterior && ent) {
            if (hard_sigm) HardSigmProb(syn0 + l1, NULL, prob_syn1p);
            else Softmax(syn0 + l1, NULL, prob_syn1p);
          }          
          if (eoe) {
            for (c1 = 0; c1 < cate_n; c1++) for (c2 = 0; c2 < cate_k; c2++) {
              ag1 = 0;
              for (c3 = 0; c3 < eoe; c3++) ag1 += neu1e[c1 * eoe + c3] * syn0eoe[c1 * cate_k * eoe + c2 * eoe + c3];
              neu1e_ae[c1 * cate_k + c2] = ag1;
            }
          } else for (c = 0; c < layer1_size; c++) neu1e_ae[c] = neu1e[c];
          for (c1 = 0; c1 < cate_n; c1++) {
            for (c2 = 0; c2 < cate_k; c2++) neu1e_ae[c1 * cate_k + c2] = (binary_one + 1) * neu1e_ae[c1 * cate_k + c2] / tau / var_scale;
            if (posterior && kl) for (c2 = 0; c2 < r_cate_k; c2++) neu1e_prob[c1 * r_cate_k + c2] = -fast_log(prob_syn1p[c1 * r_cate_k + c2] / prob_syn0[c1 * r_cate_k + c2]) + 1;
            if (!posterior && ent) for (c2 = 0; c2 < r_cate_k; c2++) neu1e_prob[c1 * r_cate_k + c2] = -(fast_log(prob_syn1p[c1 * r_cate_k + c2]) + 1) / vocab[last_word].cn_e;
          }
          for (c1 = 0; c1 < cate_n; c1++)
            for (c2 = 0; c2 < cate_k; c2++) {
              ag1 = ag2 = 0;
              if (hard_sigm) {
                ag1 += neu1e_ae[c1] * HardSigmGrad(prob_syn1p_app[c1]);
                if ((posterior && kl) || (!posterior && ent)) ag1 += (neu1e_prob[c1 * r_cate_k] - neu1e_prob[c1 * r_cate_k + 1]) * HardSigmGrad(prob_syn1p[c1 * r_cate_k]);
                //if (posterior && kl) ag2 += (prob_syn1p[c1 * r_cate_k] / prob_syn0[c1 * r_cate_k] - prob_syn1p[c1 * r_cate_k + 1] / prob_syn0[c1 * r_cate_k + 1]) * HardSigmGrad(prob_syn0[c1 * r_cate_k]);
              } else {
                for (c3 = 0; c3 < cate_k; c3++) 
                  ag1 += neu1e_ae[c1 * cate_k + c3] * prob_syn1p_app[c1 * cate_k + c3] * ((c2 == c3 ? 1 : 0) - prob_syn1p_app[c1 * cate_k + c2]);
                if ((posterior && kl) || (!posterior && ent)) for (c3 = 0; c3 < r_cate_k; c3++)
                  ag1 += neu1e_prob[c1 * r_cate_k + c3] * prob_syn1p[c1 * r_cate_k + c3] * ((c2 == c3 ? 1 : 0) - prob_syn1p[c1 * r_cate_k + c2]);
                if (posterior && kl) for (c3 = 0; c3 < r_cate_k; c3++) 
                  ag2 += prob_syn1p[c1 * r_cate_k + c3] * ((c2 == c3 ? 1 : 0) - prob_syn0[c1 * r_cate_k + c2]);
              }
              if (adam) syn0[l1 + c1 * cate_k + c2] += AdamGrad(adam_m_syn0 + l1 + c1 * cate_k + c2, adam_v_syn0 + l1 + c1 * cate_k + c2, ag1 + ag2);
              else if (rollback) {
                syn0[l1 + c1 * cate_k + c2] += 2 * alpha * (ag1 + ag2);
                all_grad[l1 + c1 * cate_k + c2] += alpha * ag1;
              }
              else syn0[l1 + c1 * cate_k + c2] += alpha * (ag1 + ag2);
              if (posterior) {
                if (adam) syn1p[l11 + c1 * cate_k + c2] += AdamGrad(adam_m_syn1p + l1 + c1 * cate_k + c2, adam_v_syn1p + l1 + c1 * cate_k + c2, ag1);
                else syn1p[l11 + c1 * cate_k + c2] += alpha * ag1;
              }
            }
        }
      }
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  all_prob += this_prob;
  fclose(fi);
  free(neu1);
  free(neu1e);
  free(neu1e_ae);
  free(neu1e_prob);
  free(gs_syn0);
  free(gs_syn0_eoe);
  free(pos);
  free(prob_syn1p);
  free(prob_syn1p_app);
  free(prob_syn0);
  pthread_exit(NULL);
}

void TrainModel() {
  long a, b, c, d, e;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  if (pt == NULL) {
    fprintf(stderr, "cannot allocate memory for threads\n");
    exit(1);
  }
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  starting_tau = tau;
  if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
  if (save_vocab_file[0] != 0) SaveVocab();
  if (output_file[0] == 0) return;
  if (context_output_file[0] == 0) return;
  InitNet();
  if (negative > 0) InitUnigramTable();
  start = clock();
  for (e = 0; e < num_epoch; e++) {
    printf("\n--- epoch %ld ---\n", e);
    for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
    for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
    printf("probability: %lf\n", all_prob / train_words);
    all_prob = 0;
    pre_train -= 1;
    //last_word_count_actual += word_count_actual;
    //starting_alpha *= alpha_decay;
  }
  for (a = 0; a < layer1_size * eoe; a++) printf("%lf,", syn0eoe[a]);
  if (classes == 0) {
    // Save the word vectors
    SaveVec();
  } else {
    FILE *fo;
    fo = fopen(output_file, "wb");
    if (fo == NULL) {
      fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
      exit(1);
    }
    // Run K-means on the word vectors
    int clcn = classes, iter = 10, closeid;
    int *centcn = (int *)malloc(classes * sizeof(int));
    if (centcn == NULL) {
      fprintf(stderr, "cannot allocate memory for centcn\n");
      exit(1);
    }
    int *cl = (int *)calloc(vocab_size, sizeof(int));
    real closev, x;
    real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
    for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
    for (a = 0; a < iter; a++) {
      for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
      for (b = 0; b < clcn; b++) centcn[b] = 1;
      for (c = 0; c < vocab_size; c++) {
        for (d = 0; d < layer1_size; d++) {
          cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
          centcn[cl[c]]++;
        }
      }
      for (b = 0; b < clcn; b++) {
        closev = 0;
        for (c = 0; c < layer1_size; c++) {
          cent[layer1_size * b + c] /= centcn[b];
          closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
        }
        closev = sqrt(closev);
        for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
      }
      for (c = 0; c < vocab_size; c++) {
        closev = -10;
        closeid = 0;
        for (d = 0; d < clcn; d++) {
          x = 0;
          for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
          if (x > closev) {
            closev = x;
            closeid = d;
          }
        }
        cl[c] = closeid;
      }
    }
    // Save the K-means classes
    for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
    free(centcn);
    free(cent);
    free(cl);
    fclose(fo);
  }
  free(table);
  free(pt);
  DestroyVocab();
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  srand(2017);
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-window <int>\n");
    printf("\t\tSet max skip length between words; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
    printf("\t-hs <int>\n");
    printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    printf("\t-min-count <int>\n");
    printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-debug <int>\n");
    printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-save-vocab <file>\n");
    printf("\t\tThe vocabulary will be saved to <file>\n");
    printf("\t-read-vocab <file>\n");
    printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
    printf("\t-cbow <int>\n");
    printf("\t\tUse the continuous back of words model; default is 0 (skip-gram model)\n");
    printf("\nExamples:\n");
    printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
    return 0;
  }
  pre_train_file[0] = 0;
  output_file[0] = 0;
  context_output_file[0] = 0;
  predict_output_file[0] = 0;
  save_vocab_file[0] = 0;
  read_vocab_file[0] = 0;
  //if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cate-n", argc, argv)) > 0) cate_n = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cate-k", argc, argv)) > 0) cate_k = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) num_epoch = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-report-period", argc, argv)) > 0) report_period = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-tau", argc, argv)) > 0) tau = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-eval", argc, argv)) > 0) strcpy(eval_sh, argv[i + 1]);
  if ((i = ArgPos((char *)"-kl", argc, argv)) > 0) kl = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-ent", argc, argv)) > 0) ent = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-rollback", argc, argv)) > 0) rollback = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-posterior", argc, argv)) > 0) posterior = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-freedom", argc, argv)) > 0) freedom = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-adam", argc, argv)) > 0) adam = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-pre", argc, argv)) > 0) pre_train = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary-one", argc, argv)) > 0) binary_one = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-hard-sigm", argc, argv)) > 0) hard_sigm = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-eoe", argc, argv)) > 0) eoe = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-pre-vec", argc, argv)) > 0) strcpy(pre_train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-context-output", argc, argv)) > 0) strcpy(context_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-predict_output", argc, argv)) > 0) strcpy(predict_output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  r_cate_k = cate_k;
  if (freedom) cate_k -= 1;
  layer1_size  = cate_n * cate_k;
  syn1_layer1_size = layer1_size;
  if (eoe > 0) syn1_layer1_size = cate_n * eoe;
  if (pre_train > 0 && eoe != cate_k) {
    printf("use pre train but eoe != cate_k");
    exit(1);
  }
  r_layer1_size = layer1_size + cate_n * freedom;
  var_scale = cate_n / 10;
  if (var_scale < 1) var_scale = 1;
  if (eoe > 1) var_scale = 1;
  binary_one = cate_k == 1 && freedom && binary_one; // only use in binary case
  hard_sigm = cate_k == 1 && freedom && hard_sigm; // only use in binary case
  printf("%lld category variable with %lld value, freedom: %d\n", cate_n, cate_k, freedom);
  printf("binary one: %d, hard sigm: %d, eoe: %d\n", binary_one, hard_sigm, eoe);
  vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
  vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (expTable == NULL) {
    fprintf(stderr, "out of memory\n");
    exit(1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  DestroyNet();
  free(vocab_hash);
  free(expTable);
  return 0;
}