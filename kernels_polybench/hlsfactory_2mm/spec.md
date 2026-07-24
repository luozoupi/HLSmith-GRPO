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
- Benchmark name: `hlsfactory_2mm`.
- Required HLS wrapper top function: `kernel_2mm`.
- Include `2mm.h` exactly once and reuse its declarations.
- Header-declared functions available for reuse: `kernel_2mm`.
- Functions already defined in the plain input whose names/signatures should be preserved unless wrapping is required: `kernel_2mm`.

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
#ifndef NI
#define NI 40
#endif
#ifndef NJ
#define NJ 50
#endif
#ifndef NK
#define NK 70
#endif
#ifndef NL
#define NL 80
#endif
// <<< c2hls auto-macro guards
#include <cmath>


extern "C" {
void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0]);
}
```

Here is the plain C kernel:
```cpp
#include "2mm.h"


void kernel_2mm(   
		double alpha,
		double beta,
		double tmp[ NI + 0][NJ + 0],
		double A[ NI + 0][NK + 0],
		double B[ NK + 0][NJ + 0],
		double C[ NJ + 0][NL + 0],
		double D[ NI + 0][NL + 0])
{


    const int ni = NI;
    const int nj = NJ;
    const int nk = NK;
    const int nl = NL;

  int i, j, k;


  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      {
	tmp[i][j] = 0.0;
	for (k = 0; k < nk; ++k)
	  tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
  for (i = 0; i < ni; i++)
    for (j = 0; j < nl; j++)
      {
	D[i][j] *= beta;
	for (k = 0; k < nj; ++k)
	  D[i][j] += tmp[i][k] * C[k][j];
      }

}
```

Provide the complete HLS-optimized code in a ```cpp code fence.