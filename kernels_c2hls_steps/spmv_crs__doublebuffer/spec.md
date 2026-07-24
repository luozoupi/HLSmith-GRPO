Apply DOUBLE BUFFERING optimization to the following HLS code.

Double buffering means:
- Create TWO copies of each local buffer (e.g., buffer_A_1 and buffer_A_2)
- In the outer loop, alternate between buffer pairs: when loading into buffer_1, compute from buffer_2, and vice versa
- Use a flag (e.g., `(iteration/tile_size) % 2`) to select which buffer set to use
- This allows the load and compute phases to overlap in time

The load() and compute() functions should accept a flag parameter to select buffers.
Keep all existing pipeline/partition pragmas.

Current synthesis report:
  lat_worst: -1
  ii_max: -1
  lut: 6332
  ff: 6091
  dsp: 11
  bram: 21
  clk_est_ns: 2.431

Header:
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define NNZ 1666
#define N 494
#define TYPE double

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N + 1],
          TYPE vec[N], TYPE out[N]);

struct bench_args_t {
    TYPE val[NNZ];
    int32_t cols[NNZ];
    int32_t rowDelimiters[N + 1];
    TYPE vec[N];
    TYPE out[N];
};

```

Current HLS code:
```cpp
#include "spmv.h"

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N + 1],
          TYPE vec[N], TYPE out[N]) {
    int i, j;
    TYPE sum, Si;

    spmv_1: for (i = 0; i < N; i++) {
        sum = 0; Si = 0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i + 1];
        spmv_2: for (j = tmp_begin; j < tmp_end; j++) {
            Si = val[j] * vec[cols[j]];
            sum = sum + Si;
        }
        out[i] = sum;
    }
}

extern "C" {
void workload(TYPE* val, int32_t* cols, int32_t* rowDelimiters,
              TYPE* vec, TYPE* out) {
#pragma HLS INTERFACE m_axi port=val offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=cols offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=rowDelimiters offset=slave bundle=gmem1
#pragma HLS INTERFACE m_axi port=vec offset=slave bundle=gmem2
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem3
#pragma HLS INTERFACE s_axilite port=val bundle=control
#pragma HLS INTERFACE s_axilite port=cols bundle=control
#pragma HLS INTERFACE s_axilite port=rowDelimiters bundle=control
#pragma HLS INTERFACE s_axilite port=vec bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_val[NNZ];
    int32_t l_cols[NNZ];
    int32_t l_rowDelimiters[N + 1];
    TYPE l_vec[N];
    TYPE l_out[N];
    int i;

    for (i = 0; i < NNZ; i++) { l_val[i] = val[i]; l_cols[i] = cols[i]; }
    for (i = 0; i < N + 1; i++) l_rowDelimiters[i] = rowDelimiters[i];
    for (i = 0; i < N; i++) l_vec[i] = vec[i];

    spmv(l_val, l_cols, l_rowDelimiters, l_vec, l_out);

    for (i = 0; i < N; i++) out[i] = l_out[i];
}
}

```

Provide the complete double-buffer-optimized code in a ```cpp code fence.