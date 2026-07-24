// Golden testbench for the FIR task. Exit code 0 = PASS (csim gate).
#include <cstdio>
#include <cstdlib>

#define N 128
#define TAPS 8

void fir(const int x[N], const int c[TAPS], int y[N]);

int main() {
  int x[N], c[TAPS], y[N], ref[N];

  srand(42);
  for (int i = 0; i < N; i++) x[i] = rand() % 201 - 100;
  for (int t = 0; t < TAPS; t++) c[t] = rand() % 21 - 10;

  // Golden: causal convolution with zero-initial state.
  for (int i = 0; i < N; i++) {
    long acc = 0;
    for (int t = 0; t < TAPS; t++)
      if (i - t >= 0) acc += (long)c[t] * x[i - t];
    ref[i] = (int)acc;
  }

  for (int i = 0; i < N; i++) y[i] = 0;
  fir(x, c, y);

  int errs = 0;
  for (int i = 0; i < N; i++) {
    if (y[i] != ref[i]) {
      if (errs < 10) printf("MISMATCH i=%d got=%d want=%d\n", i, y[i], ref[i]);
      errs++;
    }
  }
  if (errs) {
    printf("FAIL: %d mismatches\n", errs);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
