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

Here is the plain C kernel:
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


void workload(TYPE* orig, TYPE* sol, TYPE* filter) {

    TYPE l_orig[row_size * col_size];
    TYPE l_sol[row_size * col_size];
    TYPE l_filter[f_size];
    int i;

    for (i = 0; i < row_size * col_size; i++) l_orig[i] = orig[i];
    for (i = 0; i < f_size; i++) l_filter[i] = filter[i];

    stencil(l_orig, l_sol, l_filter);

    for (i = 0; i < row_size * col_size; i++) sol[i] = l_sol[i];
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.