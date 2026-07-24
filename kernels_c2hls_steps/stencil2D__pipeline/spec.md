Apply PIPELINE optimization to the following HLS code.

Pipeline means:
- Add `#pragma HLS PIPELINE II=1` to the innermost compute loops
- Add `#pragma HLS ARRAY_PARTITION` on local arrays that need parallel access within the pipeline
- Add `#pragma HLS DEPENDENCE variable=X inter false` where loop-carried dependencies are false
- Add `#pragma HLS LOOP_TRIPCOUNT min=N max=N` for variable-bound loops

Do NOT change the algorithmic structure. Only add pipeline/partition/dependence pragmas.

Current synthesis report:
  lat_worst: 24265
  ii_max: 24266
  lut: 4246
  ff: 4658
  dsp: 36
  bram: 42
  clk_est_ns: 2.48

Header:
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define col_size 64
#define row_size 128
#define f_size 9
#define TYPE int32_t

void stencil(TYPE orig[row_size * col_size],
             TYPE sol[row_size * col_size],
             TYPE filter[f_size]);

struct bench_args_t {
    TYPE orig[row_size * col_size];
    TYPE sol[row_size * col_size];
    TYPE filter[f_size];
};

```

Current HLS code:
```cpp
#include "stencil.h"

void stencil(TYPE orig[row_size * col_size],
             TYPE sol[row_size * col_size],
             TYPE filter[f_size]) {
    int r, c, k1, k2;
    TYPE temp, mul;

    stencil_label1: for (r = 0; r < row_size - 2; r++) {
        stencil_label2: for (c = 0; c < col_size - 2; c++) {
            temp = (TYPE)0;
            stencil_label3: for (k1 = 0; k1 < 3; k1++) {
                stencil_label4: for (k2 = 0; k2 < 3; k2++) {
                    mul = filter[k1 * 3 + k2] * orig[(r + k1) * col_size + c + k2];
                    temp += mul;
                }
            }
            sol[(r * col_size) + c] = temp;
        }
    }
}

extern "C" {
void workload(TYPE* orig, TYPE* sol, TYPE* filter) {
#pragma HLS INTERFACE m_axi port=orig offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=sol offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=filter offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=orig bundle=control
#pragma HLS INTERFACE s_axilite port=sol bundle=control
#pragma HLS INTERFACE s_axilite port=filter bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_orig[row_size * col_size];
    TYPE l_sol[row_size * col_size];
    TYPE l_filter[f_size];
    int i;

    for (i = 0; i < row_size * col_size; i++) l_orig[i] = orig[i];
    for (i = 0; i < f_size; i++) l_filter[i] = filter[i];

    stencil(l_orig, l_sol, l_filter);

    for (i = 0; i < row_size * col_size; i++) sol[i] = l_sol[i];
}
}

```

Provide the complete pipeline-optimized code in a ```cpp code fence.