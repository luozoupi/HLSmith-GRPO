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
#include <string.h>

//-----------------------------------------------
//Original
// #define ROWS 16384
// #define COLS 16384
//-----------------------------------------------
//-----------------------------------------------
//Alec-added
#define ROWS 1024
#define COLS 1024
//-----------------------------------------------
#define TYPE int32_t
#define MIN(a,b) ((a)<=(b) ? (a) : (b))

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]);

////////////////////////////////////////////////////////////////////////////////
// Test harness interface code.

struct bench_args_t {
  int32_t J[ROWS*COLS];
  int32_t Jout[COLS];
};

```

Here is the plain C kernel:
```cpp
#include "pathfinder.h"

void pathfinder_kernel(int32_t J[ROWS*COLS], int32_t Jout[COLS]){
	int32_t temp;
	int32_t i, t, tt, n;
	int32_t min, before;

	KERNEL2: for(t = 0; t < ROWS-1 ;t++){
		for(n = 0; n < COLS; n++){
			min = Jout[n];
			
			if(n > 0){
				min = MIN(min,before);
			}
	
			if(n < COLS-1){
				min = MIN(min,Jout[n+1]);
			}
			before = Jout[n];
			
			Jout[n] = J[(t+1) * COLS + n]+min;
		}
	}
}


void workload(int32_t J[ROWS * COLS], int32_t Jout[COLS]) {
  
	
	int32_t dst[COLS], src[COLS];
	int32_t i, t, tt, n;
	int32_t min;

	memcpy(dst,J,sizeof(int32_t) * COLS);

	KERNEL_OUTER: for(t = 0; t < ROWS-1 ;t++){
		KERNEL_INNER: for(n = 0; n < COLS; n++){
			min = dst[n];
			
			if(n > 0){
				min = MIN(min,dst[n-1]);
			}
	
			if(n < COLS-1){
				min = MIN(min,dst[n+1]);
			}
			
			src[n] = J[(t+1) * COLS + n]+min;
		}
		memcpy(dst,src,sizeof(int32_t) * COLS);
	}  	
	memcpy(Jout,dst,sizeof(int32_t) * COLS);
	return;
							  
}
```

Provide the complete HLS-optimized code in a ```cpp code fence.