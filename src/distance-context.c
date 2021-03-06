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
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x


const long long max_size = 2000;         // max length of strings
const long long N = 25;                  // number of closest words that will be shown
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv) {
  FILE *f, *cf, *inf;
  char st1[max_size];
  char bestw[N][max_size];
  char file_name[max_size], context_file_name[max_size], interaction_file_name[max_size], interaction_mf_file_name[max_size], st[100][max_size];
  float dist, len, gate, bestd[N], vec[max_size], context[max_size];
  long long words, size, cate_n, cate_k, a, b, c, d, cn, bi[100], norm = 0, mf_gate_cate_k = 0, interaction_mf_size = 0;
  char ch;
  float *M, *Mn, *C, *inter, *mf_matrix;
  char *vocab;
  interaction_file_name[0] = 0;

  if (argc < 3) {
    printf("Usage: ./distance-context <FILE> <CONTEXT_FILE>\nwhere FILE and CONTEXT_FILE contains word projections in the BINARY FORMAT\n");
    return 0;
  }
  strcpy(file_name, argv[1]);
  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  strcpy(context_file_name, argv[2]);
  cf = fopen(context_file_name, "rb");
  if (cf == NULL) {
    printf("Input context file not found\n");
    return -1;
  }
  if (argc > 3) norm = atoi(argv[3]);
  if (argc > 4) {
    strcpy(interaction_file_name, argv[4]);
    cate_n = atoi(argv[5]);
    cate_k = atoi(argv[6]);
  }
  interaction_file_name[0] = 0;
  if (argc > 7) {
    mf_gate_cate_k = atoi(argv[7]);
    interaction_mf_size = atoi(argv[8]);
    strcpy(interaction_mf_file_name, argv[9]);
  }
  if (norm) printf("use norm\n");
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  vocab = (char *)malloc((long long)words * max_w * sizeof(char));
  M = (float *)malloc((long long)words * (long long)size * sizeof(float));
  Mn = (float *)malloc((long long)words * (long long)size * sizeof(float));
  C = (float *)malloc((long long)words * (long long)size * sizeof(float));
  if (interaction_file_name[0] != 0) {
    inf = fopen(interaction_file_name, "rb");
    if (inf == NULL) {
      printf("Interaction file not found\n");
      return -1;
    }
    inter = (float *)malloc(cate_n * cate_k * cate_k * sizeof(float));
    for (a = 0; a < cate_n * cate_k; a++) {
      for (b = 0; b < cate_k; b++) fscanf(inf, "%f ", &inter[a * cate_k + b]);
      fscanf(inf, "%c", &ch);
    }
    fclose(inf);
  }
  if (interaction_mf_file_name[0] != 0) {
    inf = fopen(interaction_mf_file_name, "rb");
    fscanf(inf, "%lld", &words);
    fscanf(inf, "%lld", &mf_gate_cate_k);
    mf_matrix = (float *)malloc(words * mf_gate_cate_k * sizeof(float));
    for (b = 0; b < words; b++) {
      fscanf(inf, "%s%c", &vocab[b * max_w], &ch);
      for (a = 0; a < mf_gate_cate_k; a++) fread(&mf_matrix[a + b * mf_gate_cate_k], sizeof(float), 1, inf);
      if (norm) {
        len = 0;
        for (a = 0; a < mf_gate_cate_k; a++) len += mf_matrix[a + b * mf_gate_cate_k] * mf_matrix[a + b * mf_gate_cate_k];
        len = sqrt(len);
        for (a = 0; a < mf_gate_cate_k; a++) mf_matrix[a + b * mf_gate_cate_k] /= len;
      }
    }
    fclose(inf);
  }
  if (M == NULL || Mn == NULL || C == NULL) {
    printf("Cannot allocate memory: %lld MB    %lld  %lld\n", (long long)words * size * sizeof(float) / 1048576, words, size);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    len = 0;
    for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
    len = sqrt(len);
    for (a = 0; a < size; a++) Mn[a + b * size] = M[a + b * size] / len;
    if (norm) for (a = 0; a < size; a++) M[a + b * size] = Mn[a + b * size];
  }
  fclose(f);
  fscanf(cf, "%lld", &words);
  fscanf(cf, "%lld", &size);
  for (b = 0; b < words; b++) {
    fscanf(cf, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&C[a + b * size], sizeof(float), 1, cf);
    if (norm) {
      len = 0;
      for (a = 0; a < size; a++) len += C[a + b * size] * C[a + b * size];
      len = sqrt(len);
      for (a = 0; a < size; a++) C[a + b * size] /= len;
    }
  }
  fclose(cf);
  while (1) {
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    printf("Enter word or sentence (EXIT to break): ");
    a = 0;
    while (1) {
      st1[a] = fgetc(stdin);
      if ((st1[a] == '\n') || (a >= max_size - 1)) {
        st1[a] = 0;
        break;
      }
      a++;
    }
    if (!strcmp(st1, "EXIT")) break;
    cn = 0;
    b = 0;
    c = 0;
    while (1) {
      if (st1[c] == 0) break;
      st[cn][b] = st1[c];
      b++;
      c++;
      st[cn][b] = 0;
      if (st1[c] == 0) break;
      if (st1[c] == ' ') {
        cn++;
        b = 0;
        c++;
      }
    }
    cn++;
    printf("cn:%lld\n", cn);
    for (a = 0; a < cn; a++) {
      for (b = 0; b < words; b++) if (!strcmp(&vocab[b * max_w], st[a])) break;
      if (b == words) b = -1;
      bi[a] = b;
      printf("\nWord: %s  Position in vocabulary: %lld\n", st[a], bi[a]);
      //if (b == -1) {
      //  printf("Out of dictionary word!\n");
      //  break;
      //}
    }
    //if (b == -1) continue;
    printf("\n                                              Word       Cosine distance\n------------------------------------------------------------------------\n");
    for (a = 0; a < size; a++) vec[a] = 0;
    for (a = 0; a < size; a++) vec[a] += M[a + bi[0] * size];
    if (interaction_file_name[0] != 0) {
      for (a = 0; a < size; a++) context[a] = 0;
      if (mf_gate_cate_k) {
        for (b = 1; b < cn; b++) {
          if (bi[b] == -1) continue;
          gate = 0;
          for (a = 0; a < mf_gate_cate_k; a++) gate += M[a + bi[0] * size] * C[a + bi[b] * size];
          for (a = 0; a < size; a++) context[a] += gate * C[a + bi[b] * size];
          printf("\ninteraction: %lf\n", gate);
        }
        for (a = 0; a < size; a++) vec[a] += context[a];
      } else {
        for (b = 1; b < cn; b++) {
          if (bi[b] == -1) continue;
          for (a = 0; a < size; a++) context[a] += C[a + bi[b] * size];
        }
        gate = 0;
        printf("\ninteraction: %lf\n", gate);
        for (a = 0; a < cate_n; a++) for (b = 0; b < cate_k; b++) for (c = 0; c < cate_k; c++) {
          printf("%lf, ", inter[a * cate_k * cate_k + b * cate_k + c]);
          gate += context[a * cate_k + b] * M[a * cate_k + c + bi[0] * size] * inter[a * cate_k * cate_k + b * cate_k + c];
        }
        for (a = 0; a < size; a++) vec[a] += context[a] * gate;
        printf("\ninteraction: %lf\n", gate);
      }
    } else if (interaction_mf_file_name[0] != 0) {
      for (a = 0; a < size; a++) context[a] = 0;
      for (b = 1; b < cn; b++) {
        if (bi[b] == -1) continue;
        gate = 0;
        if (bi[b] <= interaction_mf_size && bi[0] <= interaction_mf_size) {
          gate = 0;
          for (a = 0; a < mf_gate_cate_k; a++) gate += mf_matrix[a + bi[0] * mf_gate_cate_k] * mf_matrix[a + bi[b] * mf_gate_cate_k];
          gate = 1 / (1 + exp(-gate));
          printf("\ninteraction: %lf\n", gate);
        }
        
        for (a = 0; a < size; a++) context[a] += gate * C[a + bi[b] * size];
      }
      for (a = 0; a < size; a++) vec[a] += context[a];
    } else for (b = 1; b < cn; b++) {
      if (bi[b] == -1) continue;
      for (a = 0; a < size; a++) vec[a] += C[a + bi[b] * size];
    }
    len = 0;
    for (a = 0; a < size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < size; a++) vec[a] /= len;
    // vis
    for (a = 0; a < 5; a++) {
      for (b = 0; b < 5; b++) printf("%7.3f", Mn[a * 5 + b + bi[0] * size]);
      printf("\n");
      for (b = 0; b < 5; b++) printf("%7.3f", vec[a * 5 + b]);
      printf("\n----\n");
    }
    for (a = 0; a < N; a++) bestd[a] = 0;
    for (a = 0; a < N; a++) bestw[a][0] = 0;
    for (c = 0; c < words; c++) {
      a = 0;
      for (b = 0; b < cn; b++) if (bi[b] == c) a = 1;
      if (a == 1) continue;
      dist = 0;
      for (a = 0; a < size; a++) dist += vec[a] * Mn[a + c * size];
      for (a = 0; a < N; a++) {
        if (dist > bestd[a]) {
          for (d = N - 1; d > a; d--) {
            bestd[d] = bestd[d - 1];
            strcpy(bestw[d], bestw[d - 1]);
          }
          bestd[a] = dist;
          strcpy(bestw[a], &vocab[c * max_w]);
          break;
        }
      }
    }
    for (a = 0; a < N; a++) printf("%50s\t\t%f\n", bestw[a], bestd[a]);
  }
  free(M);
  free(Mn);
  free(C);
  if (inter != NULL) free(inter);
  if (mf_matrix != NULL) free(inter);
  return 0;
}
