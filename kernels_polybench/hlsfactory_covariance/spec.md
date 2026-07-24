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
- Benchmark name: `hlsfactory_covariance`.
- Required HLS wrapper top function: `kernel_covariance`.
- Include `covariance.h` exactly once and reuse its declarations.
- Header-declared functions available for reuse: `kernel_covariance`.
- Functions already defined in the plain input whose names/signatures should be preserved unless wrapping is required: `kernel_covariance`.

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
#pragma once
// >>> c2hls auto-macro guards (do not edit between markers)
#ifndef N
#define N 100
#endif
#ifndef M
#define M 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_covariance( 
		       double float_n,
		       double data[ N + 0][M + 0],
		       double cov[ M + 0][M + 0],
		       double mean[ M + 0]);
}
```

Here is the plain C kernel:
```cpp
#include "covariance.h"


void kernel_covariance( 
		       double float_n,
		       double data[ N + 0][M + 0],
		       double cov[ M + 0][M + 0],
		       double mean[ M + 0])
{


    const int n = N;
    const int m = M;

  int i, j, k;

  for (j = 0; j < m; j++)
    {
      mean[j] = 0.0;
      for (i = 0; i < n; i++)
        mean[j] += data[i][j];
      mean[j] /= float_n;
    }

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      data[i][j] -= mean[j];

  for (i = 0; i < m; i++)
    for (j = i; j < m; j++)
      {
        cov[i][j] = 0.0;
        for (k = 0; k < n; k++)
	  cov[i][j] += data[k][i] * data[k][j];
        cov[i][j] /= (float_n - 1.0);
        cov[j][i] = cov[i][j];
      }

}
```

Provide the complete HLS-optimized code in a ```cpp code fence.