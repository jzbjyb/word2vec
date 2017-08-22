#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x
#include <ctype.h>

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries
int binary = 1;

inline double fast_exp(double x) {
  x = 1.0 + x / 256.0;
  x *= x; x *= x; x *= x; x *= x;
  x *= x; x *= x; x *= x; x *= x;
  return x;
}

int main(int argc, char **argv)
{
  FILE *f;
  char file_name[max_size], output_file[max_size], ch;
  long long words, size, a, b, c, d, cate_n, cate_k;
  float *M, prob;
  char *vocab;

  strcpy(file_name, argv[1]);
  strcpy(output_file, argv[2]);
  cate_n = atoi(argv[3]);
  cate_k = atoi(argv[4]);

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  if (cate_n * cate_k != size) {
      printf("Size not correct\n");
      return -1;
  }
  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    for (c = 0; c < cate_n; c++) {
        prob = 0;
        for (d = 0; d < cate_k; d++) {
            M[b * size + c * cate_k + d] = fast_exp(M[b * size + c * cate_k + d]);
            prob += M[b * size + c * cate_k + d];
        }
        for (d = 0; d < cate_k; d++) M[b * size + c * cate_k + d] /= prob;
    }
  }
  fclose(f);
  FILE *fo;
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    return -1;
  }
  fprintf(fo, "%lld %lld\n", words, size);
  for (b = 0; b < words; b++) {
    fprintf(fo, "%s ", vocab + b * max_w);
    if (binary) for (a = 0; a < size; a++) fwrite(&M[b * size + a], sizeof(float), 1, fo);
    else for (a = 0; a < size; a++) fprintf(fo, "%lf ", M[b * size + a]);
    fprintf(fo, "\n");
  }
  fclose(fo);
}