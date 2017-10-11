#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdlib.h> // mac os x
#include <ctype.h>

const long long max_size = 2000;         // max length of strings
const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv)
{
  FILE *f, *fo;
  char file_name[max_size], output_file[max_size], ch;
  long long words, size, a, b;
  float *M;
  char *vocab;

  strcpy(file_name, argv[1]);
  strcpy(output_file, argv[2]);

  f = fopen(file_name, "rb");
  if (f == NULL) {
    printf("Input file not found\n");
    return -1;
  }
  fo = fopen(output_file, "wb");
  if (fo == NULL) {
    fprintf(stderr, "Cannot open %s: permission denied\n", output_file);
    return -1;
  }
  fscanf(f, "%lld", &words);
  fscanf(f, "%lld", &size);
  fprintf(fo, "%lld %lld\n", words, size);

  vocab = (char *)malloc(words * max_w * sizeof(char));
  M = (float *)malloc(words * size * sizeof(float));
  if (M == NULL) {
    printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
    return -1;
  }
  for (b = 0; b < words; b++) {
    fscanf(f, "%s%c", &vocab[b * max_w], &ch);
    fprintf(fo, "%s ", vocab + b * max_w);
    for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
    for (a = 0; a < size; a++) fprintf(fo, "%lf ", M[a + b * size]);
    fprintf(fo, "\n");
  }
  fclose(f);
  fclose(fo);
}