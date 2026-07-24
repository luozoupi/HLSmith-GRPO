Convert the following plain C/C++ kernel code into Xilinx Vitis HLS-optimized code.

Requirements:
1. Use a top-level `workload()` function wrapped in `extern "C" { }`.
   If the input already contains a `workload()` wrapper, preserve and upgrade that wrapper instead of creating a second wrapper.
   If there is no wrapper yet, add one that calls the kernel.
2. Add HLS INTERFACE pragmas to the workload function:
   - `#pragma HLS INTERFACE m_axi port=<ptr> offset=slave bundle=gmem` for pointer arguments
   - `#pragma HLS INTERFACE s_axilite port=<arg> bundle=control` for all arguments
   - `#pragma HLS INTERFACE s_axilite port=return bundle=control`
3. Add performance pragmas to the kernel:
   - `#pragma HLS PIPELINE` on innermost loops where appropriate
   - `#pragma HLS UNROLL` where beneficial for parallelism
   - `#pragma HLS ARRAY_PARTITION` for arrays that need parallel access
4. Keep the original algorithm logic UNCHANGED
5. Include the original header file
6. Do NOT copy or re-declare structs, typedefs, constants, or function prototypes that already exist in the header; include the header once and reuse its declarations
7. The code must be synthesizable with Vitis HLS targeting the Alveo U280 (vivado flow)

Benchmark-specific guidance:
The top-level function must be `workload`.

Checklist before returning:
- Include the header exactly once.
- Reuse existing function names and signatures from the plain input when possible.
- Preserve the plain-input helper and wrapper structure unless a change is required for valid Vitis HLS pragmas.
- Prefer minimal edits to the plain input over creative rewrites.
- Do not redeclare header-owned structs/types like `bench_args_t`.
- Do not invent undeclared helper arrays or buffers like `l_*`; if a local buffer is needed, declare it and fill it explicitly.
- Keep every `#pragma HLS` inside a function body or loop body, never at global scope.

Here is the header:
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

Here is the plain C kernel:
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


void workload(TYPE* a) {

    TYPE l_a[SIZE];
    int i;
    for (i = 0; i < SIZE; i++) l_a[i] = a[i];
    ms_mergesort(l_a);
    for (i = 0; i < SIZE; i++) a[i] = l_a[i];
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.