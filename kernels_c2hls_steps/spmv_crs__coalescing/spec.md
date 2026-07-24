Apply MEMORY COALESCING optimization to the following HLS code.

Memory coalescing means:
- Change pointer arguments in the workload() function to use wide bus types: `ap_uint<512>*` (or `ap_uint<LARGE_BUS>*`)
- Use wide bus read/write helper functions (memcpy_wide_bus_read_float, memcpy_wide_bus_write_float, etc.)
- Include the wide bus header: `#include "../../../common/mc.h"` (defines LARGE_BUS=512, MARS_WIDE_BUS_TYPE, and provides helper functions)
- Update INTERFACE pragmas to use the wide bus pointer types
- Increase burst lengths where possible (max_read_burst_length=256, max_write_burst_length=256)
- Add cyclic array partitioning with appropriate factors for local buffers

Keep all existing double-buffering, pipeline, and unroll optimizations.

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

Provide the complete coalescing-optimized code in a ```cpp code fence.