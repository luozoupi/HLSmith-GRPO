# Task: FIR filter (8-tap, 128 samples) — Alveo U280 target

Write a Vitis HLS C++ implementation of a causal 8-tap FIR filter.

Required exact signature (the testbench links against it):

```cpp
#define N 128
#define TAPS 8
void fir(const int x[N], const int c[TAPS], int y[N]);
```

Semantics: `y[i] = sum over t in [0, TAPS) of c[t] * x[i-t]`, treating `x[j] = 0` for `j < 0`
(zero initial state). Integer arithmetic, no saturation.

Target: part xcu280-fsvh2892-2L-e (Alveo U280) at a 3.33 ns clock. Optimize worst-case latency
(and initiation interval) using HLS pragmas (PIPELINE, ARRAY_PARTITION, UNROLL, ...) while
staying within the resource budgets. Respond with a single complete C++ source in a ```cpp
fenced block. Do not include a main() function or the testbench.
