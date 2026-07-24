Apply UNROLL optimization to the following HLS code.

Unroll means:
- Add `#pragma HLS UNROLL factor=N` to inner loops where parallelism is beneficial
- Increase array partitioning factors to match unroll factors
- The unroll factor should be a power of 2 (2, 4, 8) and divide the loop bound evenly
- Focus on the dimension/feature loops that can benefit from data parallelism

Do NOT change the algorithmic structure. Only add unroll pragmas and adjust array partitioning.

Current synthesis report:
  lat_worst: 524701
  ii_max: 524702
  lut: 14551
  ff: 25792
  dsp: 11
  bram: 4
  clk_est_ns: 2.673

Header:
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define TYPE double
#define row_size 64
#define col_size 64
#define N (row_size * col_size)

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]);

struct bench_args_t {
  TYPE m1[N];
  TYPE m2[N];
  TYPE prod[N];
};

```

Current HLS code:
```cpp
#include "gemm.h"

void gemm(TYPE m1[N], TYPE m2[N], TYPE prod[N]) {
    int i, j, k;
    int k_col, i_col;
    TYPE mult;

    outer: for (i = 0; i < row_size; i++) {
        middle: for (j = 0; j < col_size; j++) {
            i_col = i * col_size;
            TYPE sum = 0;
            inner: for (k = 0; k < row_size; k++) {
                k_col = k * col_size;
                mult = m1[i_col + k] * m2[k_col + j];
                sum += mult;
            }
            prod[i_col + j] = sum;
        }
    }
}

extern "C" {
void workload(TYPE* m1, TYPE* m2, TYPE* prod) {
#pragma HLS INTERFACE m_axi port=m1 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=m2 offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=prod offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=m1 bundle=control
#pragma HLS INTERFACE s_axilite port=m2 bundle=control
#pragma HLS INTERFACE s_axilite port=prod bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    gemm(m1, m2, prod);
}
}

```

Provide the complete unroll-optimized code in a ```cpp code fence.