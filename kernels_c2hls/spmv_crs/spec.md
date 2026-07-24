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
Use the existing kernel interface `spmv(val, cols, rowDelimiters, vec, out)` from the header.
Keep the workload wrapper very close to the plain input: preserve the existing local arrays `l_val`, `l_cols`, `l_rowDelimiters`, `l_vec`, and `l_out` plus their copy-in/copy-out loops.
Do not invent new helper buffers beyond the existing plain-input locals unless they are clearly necessary and fully declared.
Keep the wrapper ports aligned with the reference AXI-visible arrays: `val`, `cols`, `rowDelimiters`, `vec`, and `out`.
Do not collapse the wrapper into a direct pointer call to `spmv`; the plain input already gives the intended wrapper structure.
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

Here is the plain C kernel:
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


void workload(TYPE* val, int32_t* cols, int32_t* rowDelimiters,
              TYPE* vec, TYPE* out) {

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
```

Provide the complete HLS-optimized code in a ```cpp code fence.