// Reference (baseline) implementation of the FIR task.
// This file defines the QoR baseline; LLM-generated candidates replace it
// and are scored against the numbers this version synthesizes to.

#define N 128
#define TAPS 8

void fir(const int x[N], const int c[TAPS], int y[N]) {
  int shift_reg[TAPS] = {0};

SAMPLE:
  for (int i = 0; i < N; i++) {
    int acc = 0;
  SHIFT_MAC:
    for (int t = TAPS - 1; t > 0; t--) {
      shift_reg[t] = shift_reg[t - 1];
      acc += shift_reg[t] * c[t];
    }
    shift_reg[0] = x[i];
    acc += shift_reg[0] * c[0];
    y[i] = acc;
  }
}
