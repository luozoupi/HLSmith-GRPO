Apply PIPELINE optimization to the following HLS code.

Pipeline means:
- Add `#pragma HLS PIPELINE II=1` to the innermost compute loops
- Add `#pragma HLS ARRAY_PARTITION` on local arrays that need parallel access within the pipeline
- Add `#pragma HLS DEPENDENCE variable=X inter false` where loop-carried dependencies are false
- Add `#pragma HLS LOOP_TRIPCOUNT min=N max=N` for variable-bound loops

Do NOT change the algorithmic structure. Only add pipeline/partition/dependence pragmas.

Current synthesis report:
  lat_worst: -1
  ii_max: -1
  lut: 2577
  ff: 1903
  dsp: 0
  bram: 10
  clk_est_ns: 2.431

Header:
```cpp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

#define SIZE 2048
#define TYPE int32_t
#define TYPE_MAX INT32_MAX

void ms_mergesort(TYPE a[SIZE]);

struct bench_args_t {
    TYPE a[SIZE];
};

```

Current HLS code:
```cpp
#include "sort.h"

void merge(TYPE a[SIZE], int start, int m, int stop) {
    TYPE temp[SIZE];
    int i, j, k;

    merge_label1: for (i = start; i <= m; i++) {
        temp[i] = a[i];
    }
    merge_label2: for (j = m + 1; j <= stop; j++) {
        temp[m + 1 + stop - j] = a[j];
    }

    i = start;
    j = stop;

    merge_label3: for (k = start; k <= stop; k++) {
        TYPE tmp_j = temp[j];
        TYPE tmp_i = temp[i];
        if (tmp_j < tmp_i) {
            a[k] = tmp_j;
            j--;
        } else {
            a[k] = tmp_i;
            i++;
        }
    }
}

void ms_mergesort(TYPE a[SIZE]) {
    int start, stop;
    int i, m, from, mid, to;

    start = 0;
    stop = SIZE;

    mergesort_label1: for (m = 1; m < stop - start; m += m) {
        mergesort_label2: for (i = start; i < stop; i += m + m) {
            from = i;
            mid = i + m - 1;
            to = i + m + m - 1;
            if (to < stop) {
                merge(a, from, mid, to);
            } else {
                merge(a, from, mid, stop);
            }
        }
    }
}

extern "C" {
void workload(TYPE* a) {
#pragma HLS INTERFACE m_axi port=a offset=slave bundle=gmem
#pragma HLS INTERFACE s_axilite port=a bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    TYPE l_a[SIZE];
    int i;
    for (i = 0; i < SIZE; i++) l_a[i] = a[i];
    ms_mergesort(l_a);
    for (i = 0; i < SIZE; i++) a[i] = l_a[i];
}
}

```

Provide the complete pipeline-optimized code in a ```cpp code fence.